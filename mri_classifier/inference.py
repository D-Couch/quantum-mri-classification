"""Local prediction using a previously saved research checkpoint."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import ExperimentConfig
from .models import build_classical_model, build_hybrid_model


def load_checkpoint(path: str | Path, device: str | None = None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**state["config"])
    labels = state["labels"]
    if device is None:
        use_cuda = torch.cuda.is_available() and (
            config.model_family != "hybrid" or config.quantum_backend == "lightning.gpu"
        )
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = torch.device(device)
    if config.model_family == "hybrid":
        gpu_quantum = config.quantum_backend == "lightning.gpu"
        if gpu_quantum and device.type != "cuda":
            raise ValueError("lightning.gpu checkpoints require CUDA for inference")
        if not gpu_quantum and device.type != "cpu":
            raise ValueError("CPU PennyLane backends require CPU inference")
    model = (
        build_hybrid_model(len(labels), config.qubits, config.circuit_depth, config.quantum_backend)
        if config.model_family == "hybrid"
        else build_classical_model(
            config.architecture,
            len(labels),
            pretrained=False,
            mask_auxiliary=getattr(config, "mask_auxiliary", False),
        )
    )
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, labels, config, device


def predict_image(checkpoint: str | Path, image, device: str | None = None) -> dict:
    model, labels, config, device = load_checkpoint(checkpoint, device)
    grayscale = config.model_family == "hybrid"
    mode = "L" if grayscale else "RGB"
    size = 64 if grayscale else 224
    pipeline = [transforms.Resize((size, size)), transforms.ToTensor()]
    if config.training_mode == "transfer" and not grayscale:
        pipeline.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    with Image.open(image) as source:
        tensor = transforms.Compose(pipeline)(source.convert(mode)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().tolist()
    return {
        "prediction": labels[max(range(len(probabilities)), key=probabilities.__getitem__)],
        "probabilities": dict(zip(labels, probabilities)),
        "experiment_kind": config.experiment_kind,
        "dataset_kind": config.dataset_kind,
        "research_notice": "Research output only; not for clinical diagnosis.",
    }


def predict_patient(checkpoints: list[str | Path], images: list, device: str | None = None) -> dict:
    """Pool slice probabilities per patient, then average across compatible checkpoints."""
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if not images:
        raise ValueError("at least one image is required")

    import torch

    probability_sum = None
    expected_labels = None
    experiment_kinds, dataset_kinds = set(), set()
    for checkpoint in checkpoints:
        model, labels, config, model_device = load_checkpoint(checkpoint, device)
        if config.model_family == "hybrid":
            raise ValueError("patient pooling currently supports classical image models only")
        if expected_labels is not None and labels != expected_labels:
            raise ValueError("ensemble checkpoints must use the same class order")
        expected_labels = labels
        experiment_kinds.add(config.experiment_kind)
        dataset_kinds.add(config.dataset_kind)
        grayscale = config.model_family == "hybrid"
        mode = "L" if grayscale else "RGB"
        size = 64 if grayscale else 224
        pipeline = [transforms.Resize((size, size)), transforms.ToTensor()]
        if config.training_mode == "transfer" and not grayscale:
            pipeline.append(
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )
        preprocess = transforms.Compose(pipeline)
        checkpoint_probabilities = []
        for image in images:
            if hasattr(image, "seek"):
                image.seek(0)
            with Image.open(image) as source:
                tensor = preprocess(source.convert(mode)).unsqueeze(0).to(model_device)
            with torch.inference_mode():
                checkpoint_probabilities.append(
                    model(tensor).softmax(dim=1)[0].cpu().tolist()
                )
        patient_probabilities = [
            sum(probabilities[index] for probabilities in checkpoint_probabilities)
            / len(checkpoint_probabilities)
            for index in range(len(labels))
        ]
        if probability_sum is None:
            probability_sum = patient_probabilities
        else:
            probability_sum = [
                total + probability
                for total, probability in zip(probability_sum, patient_probabilities)
            ]
        del model

    probabilities = [value / len(checkpoints) for value in probability_sum]
    return {
        "prediction": expected_labels[max(range(len(probabilities)), key=probabilities.__getitem__)],
        "probabilities": dict(zip(expected_labels, probabilities)),
        "slice_count": len(images),
        "model_count": len(checkpoints),
        "experiment_kind": sorted(experiment_kinds),
        "dataset_kind": sorted(dataset_kinds),
        "research_notice": "Research output only; not for clinical diagnosis.",
    }
