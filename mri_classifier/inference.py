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
        else build_classical_model(config.architecture, len(labels), pretrained=False)
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
