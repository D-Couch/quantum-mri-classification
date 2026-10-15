import pytest

from mri_classifier.config import ExperimentConfig
from mri_classifier.datasets import group_split
from mri_classifier.metrics import classification_metrics
from mri_classifier.store import ExperimentStore


def test_macro_f1_includes_absent_declared_class():
    result = classification_metrics(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]


def test_group_split_keeps_subjects_together_and_is_deterministic():
    groups = ["p1", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]
    first = group_split(groups, seed=7)
    assert first == group_split(groups, seed=7)
    seen = {}
    for split, indices in first.items():
        for i in indices:
            assert groups[i] not in seen or seen[groups[i]] == split
            seen[groups[i]] = split
    assert sorted(i for indices in first.values() for i in indices) == list(range(len(groups)))


def test_store_persists_and_updates(tmp_path):
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    experiment_id = store.create({"seed": 42})
    store.finish(experiment_id, {"macro_f1": 0.75})
    row = store.list()[0]
    assert row["id"] == experiment_id
    assert row["status"] == "completed"
    assert row["metrics"]["macro_f1"] == 0.75


def test_config_validates_mode_and_ranges():
    assert ExperimentConfig("data").to_dict()["primary_metric"] == "macro_f1"
    with pytest.raises(ValueError):
        ExperimentConfig("data", epochs=0)


def test_hybrid_circuit_has_finite_gradients_and_updates():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    from mri_classifier.models import build_hybrid_model

    model = build_hybrid_model(num_classes=3, n_qubits=2, depth=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    images = torch.rand(2, 1, 8, 8)
    labels = torch.tensor([0, 1])
    initial_weights = model.quantum.weights.detach().clone()

    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    gradient = model.quantum.weights.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    optimizer.step()
    assert not torch.equal(initial_weights, model.quantum.weights.detach())
