from pathlib import Path

import pytest

from mri_classifier.config import ExperimentConfig
from mri_classifier.datasets import group_split, load_manifest
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


def test_manifest_requires_patient_ids_and_rejects_duplicate_content(tmp_path):
    from PIL import Image

    image = tmp_path / "same.png"
    Image.new("RGB", (8, 8), "white").save(image)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "relative_path,label,patient_id\nsame.png,a,p1\nsame.png,b,p2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate image content"):
        load_manifest(manifest, tmp_path)


def test_manifest_rejects_path_traversal(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("relative_path,label,patient_id\n../outside.png,a,p1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(manifest, tmp_path)


def test_figshare_import_preserves_labels_and_patient_ids(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from mri_classifier.figshare import import_figshare_directory

    source = tmp_path / "mat"
    source.mkdir()
    with h5py.File(source / "1.mat", "w") as handle:
        group = handle.create_group("cjdata")
        group.create_dataset("label", data=np.array([[2.0]]))
        group.create_dataset("PID", data=np.array([[49], [50], [51], [52]], dtype=np.uint16))
        group.create_dataset("image", data=np.arange(64).reshape(8, 8))
    folds_path = tmp_path / "cvind.mat"
    with h5py.File(folds_path, "w") as handle:
        handle.create_dataset("cvind", data=np.ones((1, 3064)))
    result = import_figshare_directory(source, tmp_path / "converted", folds_path)
    assert result["classes"] == {1: "meningioma", 2: "glioma", 3: "pituitary"}
    assert result["patient_count"] == 1
    rows = load_manifest(result["manifest"], tmp_path / "converted")
    assert rows[0]["label"] == "glioma"
    assert rows[0]["patient_id"] == "1234"
    assert rows[0]["official_fold"] == 1


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


def test_grouped_hybrid_training_vertical_slice_is_labeled_synthetic(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    from PIL import Image

    from mri_classifier.training import run_experiment

    records = []
    for patient in range(12):
        for label, color in (("class_a", (30, 60, 90)), ("class_b", (200, 180, 160))):
            image_path = tmp_path / f"{patient}-{label}.png"
            Image.new("RGB", (16, 16), color).save(image_path)
            records.append(
                {
                    "path": image_path,
                    "label": label,
                    "patient_id": f"p{patient}",
                    "official_fold": patient % 5 + 1,
                }
            )
    config = ExperimentConfig(
        dataset_path="synthetic-test",
        dataset_kind="synthetic",
        experiment_kind="smoke",
        model_family="hybrid",
        split_strategy="official_five_fold",
        test_fold=5,
        epochs=1,
        batch_size=8,
        qubits=2,
        circuit_depth=1,
    )
    model, report = run_experiment(config, records, tmp_path / "artifacts")
    assert model is not None
    assert report["dataset_kind"] == "synthetic"
    assert report["validation_macro_f1"] >= 0
    assert report["test_metrics"]["accuracy"] >= 0
    assert report["split_strategy"] == "official_five_fold"
    assert report["test_fold"] == 5
    assert (tmp_path / "artifacts" / Path(report["checkpoint"]).name).exists()
