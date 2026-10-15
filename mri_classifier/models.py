"""Model constructors load optional ML frameworks lazily."""


def build_classical_model(architecture: str, num_classes: int, pretrained: bool = False):
    try:
        from torch import nn
        from torchvision import models
    except Exception as exc:
        raise RuntimeError(
            "Install compatible torch and torchvision to train classical models"
        ) from exc
    constructors = {"resnet18": models.resnet18, "efficientnet_b0": models.efficientnet_b0}
    if architecture not in constructors:
        raise ValueError(f"unsupported classical architecture: {architecture}")
    model = constructors[architecture](weights="DEFAULT" if pretrained else None)
    if architecture == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_hybrid_model(
    num_classes: int, n_qubits: int = 4, depth: int = 2, backend: str = "default.qubit"
):
    """Trainable image feature extractor plus variational quantum circuit."""
    try:
        import pennylane as qml
        from torch import nn
    except Exception as exc:
        raise RuntimeError("Install compatible torch and pennylane to train hybrid models") from exc
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(
        dev, interface="torch", diff_method="backprop" if "lightning" not in backend else "adjoint"
    )
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    quantum = qml.qnn.TorchLayer(circuit, {"weights": (depth, n_qubits, 3)})

    class HybridClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(16, n_qubits),
                nn.Tanh(),
            )
            self.quantum, self.head = quantum, nn.Linear(n_qubits, num_classes)

        def forward(self, x):
            return self.head(self.quantum(self.features(x) * 3.141592653589793))

    return HybridClassifier()
