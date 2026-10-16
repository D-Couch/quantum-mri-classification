"""Model constructors load optional ML frameworks lazily."""


def build_classical_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = False,
    mask_auxiliary: bool = False,
):
    try:
        from torch import nn
        from torchvision import models
    except Exception as exc:
        raise RuntimeError(
            "Install compatible torch and torchvision to train classical models"
        ) from exc
    constructors = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
    }
    if architecture not in constructors:
        raise ValueError(f"unsupported classical architecture: {architecture}")
    if mask_auxiliary:
        if architecture != "resnet18":
            raise ValueError("tumor-mask supervision is currently implemented for ResNet18")
        return _build_mask_aux_resnet18(nn, constructors[architecture], num_classes)
    constructor, default_weights = constructors[architecture]
    model = constructor(weights=default_weights if pretrained else None)
    if architecture == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def _build_mask_aux_resnet18(nn, constructor_and_weights, num_classes: int):
    """ResNet18 classifier with a mask-supervised decoder and predicted ROI pooling."""
    import torch
    from torch.nn import functional

    constructor, weights = constructor_and_weights

    class MaskAuxResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = constructor(weights=weights)
            self.encoder.fc = nn.Identity()
            channels = (64, 128, 256, 512)
            self.lateral = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in channels)
            self.mask_head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
            )
            self.classifier = nn.Linear(1024, num_classes)

        def forward_with_mask(self, images):
            encoder = self.encoder
            x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(images))))
            level1 = encoder.layer1(x)
            level2 = encoder.layer2(level1)
            level3 = encoder.layer3(level2)
            level4 = encoder.layer4(level3)

            pyramid = self.lateral[3](level4)
            for index, level in ((2, level3), (1, level2), (0, level1)):
                pyramid = self.lateral[index](level) + functional.interpolate(
                    pyramid, size=level.shape[-2:], mode="bilinear", align_corners=False
                )
            mask_logits = functional.interpolate(
                self.mask_head(pyramid),
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            roi_weights = functional.adaptive_avg_pool2d(
                mask_logits.sigmoid(), level4.shape[-2:]
            )
            roi_features = (level4 * roi_weights).sum(dim=(2, 3)) / roi_weights.sum(
                dim=(2, 3)
            ).clamp_min(1e-6)
            global_features = functional.adaptive_avg_pool2d(level4, 1).flatten(1)
            logits = self.classifier(torch.cat((global_features, roi_features), dim=1))
            return logits, mask_logits

        def forward(self, images):
            logits, _ = self.forward_with_mask(images)
            return logits

    return MaskAuxResNet18()


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
