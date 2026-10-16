"""Validated experiment configuration."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_path: str
    manifest_path: str = "data/manifest.csv"
    dataset_version: str = "unspecified"
    dataset_kind: str = "unverified"
    model_family: str = "classical"
    split_strategy: str = "group_holdout"
    test_fold: int = 5
    experiment_kind: str = "research"
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
    device_preference: str = "auto"
    num_workers: int = 0
    mixed_precision: bool = False
    augment_training: bool = False

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs and batch size must be positive and learning rate > 0")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int):
            raise TypeError("num_workers must be an integer")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not isinstance(self.mixed_precision, bool):
            raise TypeError("mixed_precision must be a boolean")
        if not isinstance(self.augment_training, bool):
            raise TypeError("augment_training must be a boolean")
        if self.training_mode not in {"scratch", "transfer"}:
            raise ValueError("training_mode must be scratch or transfer")
        if self.model_family not in {"classical", "hybrid"}:
            raise ValueError("model_family must be classical or hybrid")
        if self.device_preference not in {"auto", "cpu", "cuda"}:
            raise ValueError("device_preference must be auto, cpu, or cuda")
        if self.dataset_kind not in {"real", "synthetic", "unverified"}:
            raise ValueError("dataset_kind must be real, synthetic, or unverified")
        if self.split_strategy not in {"group_holdout", "official_five_fold"}:
            raise ValueError("unsupported split strategy")
        if self.test_fold not in range(1, 6):
            raise ValueError("test_fold must be between 1 and 5")
        if self.experiment_kind not in {"research", "smoke"}:
            raise ValueError("experiment_kind must be research or smoke")
        if self.qubits < 1 or self.circuit_depth < 1:
            raise ValueError("quantum qubits and circuit depth must be positive")

    def to_dict(self):
        return asdict(self)
