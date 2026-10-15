"""Reproducible, group-aware image classification training and evaluation."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .datasets import group_split
from .metrics import classification_metrics


class ManifestImageDataset(Dataset):
    def __init__(self, rows, indices, labels, image_size=224, grayscale=False, pretrained=False):
        from torchvision import transforms

        self.rows = [rows[i] for i in indices]
        self.to_label = {label: i for i, label in enumerate(labels)}
        self.mode = "L" if grayscale else "RGB"
        steps = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        if pretrained:
            steps.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert(self.mode))
        return tensor, self.to_label[row["label"]]


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_experiment(
    config, rows, output_dir="artifacts", progress=None, cancel=None, resume_from=None
):
    """Train by validation macro-F1; use the final test partition once at the end.

    The manifest must identify patient/group IDs. Dataset license, source, and version must
    be recorded in config by the caller. This function never fetches or invents data.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from .models import build_classical_model, build_hybrid_model

    if len(rows) < 6 or len({r["patient_id"] for r in rows}) < 6:
        raise ValueError("at least six independent patient groups are required for train/val/test")
    label_names = sorted({r["label"] for r in rows})
    if len(label_names) < 2:
        raise ValueError("classification requires at least two classes")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    if config.split_strategy == "official_five_fold":
        if any("official_fold" not in row for row in rows):
            raise ValueError(
                "official five-fold strategy requires official_fold in every manifest row"
            )
        fold_by_patient = {}
        for row in rows:
            fold = int(row["official_fold"])
            if fold not in range(1, 6):
                raise ValueError("official fold values must be between 1 and 5")
            previous = fold_by_patient.setdefault(row["patient_id"], fold)
            if previous != fold:
                raise ValueError(f"patient {row['patient_id']} appears in multiple official folds")
        validation_fold = config.test_fold % 5 + 1
        splits = {
            "train": [
                i
                for i, row in enumerate(rows)
                if row["official_fold"] not in {config.test_fold, validation_fold}
            ],
            "validation": [
                i for i, row in enumerate(rows) if row["official_fold"] == validation_fold
            ],
            "test": [i for i, row in enumerate(rows) if row["official_fold"] == config.test_fold],
        }
    else:
        splits = group_split([r["patient_id"] for r in rows], config.seed)
    if any(not splits[name] for name in ("train", "validation", "test")):
        raise ValueError(
            "patient split produced an empty partition; provide more independent groups"
        )
    for name, indices in splits.items():
        present = {rows[i]["label"] for i in indices}
        if present != set(label_names):
            raise ValueError(
                f"{name} split is missing classes; provide more balanced patient groups"
            )

    family = getattr(config, "model_family", "classical")
    is_hybrid = family == "hybrid"
    pretrained = config.training_mode == "transfer"
    if is_hybrid and pretrained:
        raise ValueError("transfer mode is currently available for classical models only")
    torchvision_version = None
    pretrained_checkpoint = None
    if not is_hybrid:
        import torchvision

        torchvision_version = torchvision.__version__
        if pretrained:
            weight_types = {
                "resnet18": torchvision.models.ResNet18_Weights,
                "efficientnet_b0": torchvision.models.EfficientNet_B0_Weights,
            }
            weight_type = weight_types.get(config.architecture)
            if weight_type is not None:
                selected_weights = weight_type.DEFAULT
                pretrained_checkpoint = {
                    "identifier": f"torchvision.models.{type(selected_weights).__name__}.{selected_weights.name}",
                    "url": selected_weights.url,
                }
    size = 64 if is_hybrid else 224
    datasets = {
        name: ManifestImageDataset(rows, indices, label_names, size, is_hybrid, pretrained)
        for name, indices in splits.items()
    }
    model = (
        build_hybrid_model(
            len(label_names), config.qubits, config.circuit_depth, config.quantum_backend
        )
        if is_hybrid
        else build_classical_model(config.architecture, len(label_names), pretrained)
    )
    # PennyLane's current default.qubit/lightning.qubit backends are CPU simulators.
    if is_hybrid and config.device_preference == "cuda":
        raise ValueError("the configured PennyLane hybrid simulators currently run on CPU")
    if config.device_preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if not is_hybrid
        and (
            config.device_preference == "cuda"
            or (config.device_preference == "auto" and torch.cuda.is_available())
        )
        else "cpu"
    )
    mixed_precision_enabled = bool(config.mixed_precision and device.type == "cuda")
    loaders = {
        name: DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=name == "train",
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
            **({"prefetch_factor": 2} if config.num_workers > 0 else {}),
        )
        for name, ds in datasets.items()
    }
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision_enabled)

    def compute_context():
        if mixed_precision_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{int(time.time())}-{config.seed}"
    checkpoint = out / f"{run_id}-best.pt"
    history, best_score, best_epoch, stale = [], -1.0, -1, 0
    resumed_from_epoch = None
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        saved_config = state.get("config", {}).copy()
        current_config = config.to_dict()
        saved_config.pop("epochs", None)
        current_config.pop("epochs", None)
        if saved_config != current_config:
            raise ValueError("resume checkpoint configuration does not match this experiment")
        if state.get("labels") != label_names:
            raise ValueError("resume checkpoint class labels do not match the dataset")
        if not all(key in state for key in ("best_epoch", "best_score", "history")):
            raise ValueError("checkpoint predates safe warm-start recovery and cannot be resumed")
        model.load_state_dict(state["model"])
        best_epoch = int(state["best_epoch"])
        best_score = float(state["best_score"])
        history = state["history"]
        if best_epoch > config.epochs:
            raise ValueError("resume epoch exceeds the configured epoch limit")
        resumed_from_epoch = best_epoch
        state["config"] = config.to_dict()
        torch.save(state, checkpoint)
    started = time.perf_counter()

    def evaluate(loader):
        model.eval()
        true, pred, score_rows, total_loss, count = [], [], [], 0.0, 0
        with torch.inference_mode():
            for images, targets in loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                targets = targets.to(device, non_blocking=device.type == "cuda")
                with compute_context():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                count += targets.size(0)
                true.extend(label_names[i] for i in targets.cpu().tolist())
                probabilities = outputs.softmax(dim=1)
                pred.extend(label_names[i] for i in probabilities.argmax(1).cpu().tolist())
                score_rows.extend(probabilities.cpu().tolist())
        return classification_metrics(true, pred, label_names, score_rows), total_loss / count

    for epoch in range(resumed_from_epoch or 0, config.epochs):
        if cancel and cancel():
            raise InterruptedError("experiment cancelled")
        model.train()
        total_loss, count = 0.0, 0
        for images, targets in loaders["train"]:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with compute_context():
                loss = criterion(model(images), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)
        validation, val_loss = evaluate(loaders["validation"])
        scheduler.step(validation["macro_f1"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / count,
                "validation_loss": val_loss,
                "validation": validation,
            }
        )
        if validation["macro_f1"] > best_score:
            best_score, best_epoch, stale = validation["macro_f1"], epoch + 1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": label_names,
                    "config": config.to_dict(),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "history": history,
                },
                checkpoint,
            )
        else:
            stale += 1
        if progress:
            progress(epoch + 1, config.epochs, history[-1])
        if stale >= 5:
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    test_metrics, test_loss = evaluate(loaders["test"])
    metadata = {
        "status": "completed",
        "dataset": config.dataset_path,
        "dataset_version": getattr(config, "dataset_version", "unspecified"),
        "dataset_kind": getattr(config, "dataset_kind", "unverified"),
        "experiment_kind": config.experiment_kind,
        "class_names": label_names,
        "sample_count": len(rows),
        "group_count": len({r["patient_id"] for r in rows}),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_strategy": config.split_strategy,
        "test_fold": config.test_fold if config.split_strategy == "official_five_fold" else None,
        "config": config.to_dict(),
        "seed": config.seed,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision_version,
        },
        "hardware": {
            "device": str(device),
            "cuda": torch.version.cuda if device.type == "cuda" else None,
        },
        "performance": {
            "device_preference": config.device_preference,
            "num_workers": config.num_workers,
            "pin_memory": device.type == "cuda",
            "mixed_precision_requested": config.mixed_precision,
            "mixed_precision_enabled": mixed_precision_enabled,
        },
        "git_revision": _git_revision(),
        "history": history,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "test_metrics": test_metrics,
        "test_loss": test_loss,
        "checkpoint": str(checkpoint),
        "duration_seconds": time.perf_counter() - started,
        "resumed_from_checkpoint": str(resume_from) if resume_from is not None else None,
        "resumed_from_epoch": resumed_from_epoch,
        "resume_state": "optimizer_and_scheduler_reset" if resume_from is not None else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "pretrained_weights": (
            pretrained_checkpoint["identifier"] if pretrained_checkpoint is not None else None
        ),
        "pretrained_checkpoint": pretrained_checkpoint,
        "quantum": (
            {
                "qubits": config.qubits,
                "depth": config.circuit_depth,
                "backend": config.quantum_backend,
            }
            if is_hybrid
            else None
        ),
    }
    (out / f"{run_id}-report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, metadata
