"""Validated experiment configuration."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_path: str
    architecture: str = "resnet18"
    training_mode: str = "scratch"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    primary_metric: str = "macro_f1"
    quantum_backend: str = "default.qubit"
    qubits: int = 4
    circuit_depth: int = 2

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs and batch size must be positive and learning rate > 0")
        if self.training_mode not in {"scratch", "transfer"}:
            raise ValueError("training_mode must be scratch or transfer")
        if self.qubits < 1 or self.circuit_depth < 1:
            raise ValueError("quantum qubits and circuit depth must be positive")

    def to_dict(self):
        return asdict(self)
