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
    assert result["per_class"]["a"]["specificity"] == pytest.approx(1.0)
    assert result["matthews_correlation_coefficient"] == pytest.approx(0.5)


def test_score_metrics_compute_multiclass_roc_and_pr_auc():
    result = classification_metrics(
        ["a", "b", "c", "a", "b", "c"],
        ["a", "b", "c", "a", "b", "c"],
        ["a", "b", "c"],
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
        ],
    )
    assert result["roc_auc_ovr_macro"] == pytest.approx(1.0)
    assert result["pr_auc_macro"] == pytest.approx(1.0)
    assert result["cohen_kappa"] == pytest.approx(1.0)


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


def test_store_migrates_existing_schema_with_progress(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE experiments (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
            config TEXT NOT NULL, metrics TEXT NOT NULL, error TEXT
        )""")
    store = ExperimentStore(path)
    experiment_id = store.create({"dataset": "x"})
    store.update_progress(experiment_id, {"epoch": 2})
    row = store.list()[0]
    assert row["progress"] == {"epoch": 2}
    assert row["parent_id"] is None
    assert row["resume_checkpoint"] is None


def test_training_manager_recovers_stale_jobs_and_warm_starts_checkpoint(tmp_path, monkeypatch):
    import json

    import torch

    from mri_classifier import jobs
    from mri_classifier.jobs import TrainingManager

    monkeypatch.chdir(tmp_path)
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    config = ExperimentConfig(dataset_path="synthetic", dataset_kind="synthetic")
    dead_pid = 2**30
    parent_id = store.create(config.to_dict(), status="running", worker_pid=dead_pid)
    completed_id = store.create(config.to_dict(), status="running", worker_pid=dead_pid)
    checkpoint = tmp_path / "artifacts" / parent_id / "epoch-best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "model": {},
            "labels": ["a", "b"],
            "config": config.to_dict(),
            "best_epoch": 1,
            "best_score": 0.5,
            "history": [],
        },
        checkpoint,
    )
    completed_checkpoint = tmp_path / "artifacts" / completed_id / "epoch-best.pt"
    completed_checkpoint.parent.mkdir(parents=True)
    completed_checkpoint.write_bytes(b"completed-checkpoint")
    report = {
        "status": "completed",
        "checkpoint": str(completed_checkpoint),
        "validation_macro_f1": 0.8,
        "test_metrics": {"accuracy": 0.85},
    }
    (completed_checkpoint.parent / "final-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    def fake_run(config, rows, output_dir, progress, cancel, resume_from):
        assert resume_from == checkpoint
        return object(), {"validation_macro_f1": 0.7, "test_metrics": {"accuracy": 0.75}}

    monkeypatch.setattr(jobs, "run_experiment", fake_run)
    monkeypatch.setattr(jobs, "load_manifest", lambda *_: [])
    manager = TrainingManager(store)
    try:
        parent = next(row for row in store.list() if row["id"] == parent_id)
        assert parent["status"] == "failed"
        assert "process stopped" in parent["error"].lower()
        completed = next(row for row in store.list() if row["id"] == completed_id)
        assert completed["status"] == "completed"
        assert completed["metrics"]["recovered_after_restart"] is True
        assert manager.can_resume(parent_id)
        resumed_id = manager.resume(parent_id)
        manager.wait(timeout=5)
        resumed = next(row for row in store.list() if row["id"] == resumed_id)
        assert resumed["status"] == "completed"
        assert resumed["parent_id"] == parent_id
        assert resumed["resume_checkpoint"] == str(checkpoint)
    finally:
        manager.shutdown()


def test_background_job_persists_progress_and_result(tmp_path, monkeypatch):
    from threading import Event

    from PIL import Image

    from mri_classifier import jobs
    from mri_classifier.jobs import TrainingManager

    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(image)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("relative_path,label,patient_id\nsample.png,a,p1\n", encoding="utf-8")

    started, release = Event(), Event()

    def fake_run(config, rows, output_dir, progress, cancel):
        started.set()
        progress(1, 1, {"train_loss": 0.2, "validation_loss": 0.3, "validation": {"macro_f1": 0.8}})
        release.wait(timeout=5)
        return object(), {"validation_macro_f1": 0.8, "test_metrics": {"accuracy": 0.75}}

    monkeypatch.setattr(jobs, "run_experiment", fake_run)
    monkeypatch.setattr(jobs, "load_manifest", lambda *_: [])
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    manager = TrainingManager(store)
    config = ExperimentConfig(
        dataset_path=str(tmp_path),
        manifest_path=str(manifest),
        dataset_kind="synthetic",
        experiment_kind="smoke",
    )
    try:
        experiment_id = manager.submit(config)
        assert started.wait(timeout=5)
        with pytest.raises(RuntimeError, match="already active"):
            manager.submit(config)
        release.set()
        manager.wait(timeout=5)
        assert manager.active_experiment_id() is None
        result = store.list()[0]
        assert result["id"] == experiment_id
        assert result["status"] == "completed"
        assert result["progress"]["epoch"] == 1
        assert result["metrics"]["macro_f1"] == 0.8
    finally:
        release.set()
        manager.shutdown()


def test_background_job_can_be_cancelled_between_epochs(tmp_path, monkeypatch):
    from threading import Event
    from time import sleep

    from mri_classifier import jobs
    from mri_classifier.jobs import TrainingManager

    started = Event()

    def fake_run(config, rows, output_dir, progress, cancel):
        started.set()
        while not cancel():
            sleep(0.01)
        raise InterruptedError("experiment cancelled")

    monkeypatch.setattr(jobs, "run_experiment", fake_run)
    monkeypatch.setattr(jobs, "load_manifest", lambda *_: [])
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    manager = TrainingManager(store)
    config = ExperimentConfig(
        dataset_path=str(tmp_path), manifest_path=str(tmp_path / "missing.csv")
    )
    try:
        experiment_id = manager.submit(config)
        assert started.wait(timeout=5)
        assert manager.cancel()
        manager.wait(timeout=5)
        row = store.list()[0]
        assert row["id"] == experiment_id
        assert row["status"] == "cancelled"
        assert row["error"] == "experiment cancelled"
    finally:
        manager.shutdown()


def test_config_validates_mode_and_ranges():
    defaults = ExperimentConfig("data").to_dict()
    assert defaults["primary_metric"] == "macro_f1"
    assert defaults["device_preference"] == "auto"
    assert defaults["num_workers"] == 0
    assert defaults["mixed_precision"] is False
    with pytest.raises(ValueError):
        ExperimentConfig("data", epochs=0)
    with pytest.raises(ValueError):
        ExperimentConfig("data", num_workers=-1)
    with pytest.raises(ValueError):
        ExperimentConfig("data", device_preference="tpu")


def test_patient_cluster_bootstrap_resamples_whole_groups_deterministically():
    import numpy as np

    from mri_classifier.patient_bootstrap import patient_cluster_bootstrap

    labels = {"a": 0, "b": 0, "c": 1, "d": 1}
    perfect = {
        patient: np.array([[1, 0], [0, 0]]) if label == 0 else np.array([[0, 0], [0, 1]])
        for patient, label in labels.items()
    }
    imperfect = {patient: matrix.copy() for patient, matrix in perfect.items()}
    imperfect["b"] = np.array([[0, 1], [0, 0]])
    confusions = {
        "transfer": {seed: perfect for seed in (42, 43, 44)},
        "scratch": {seed: imperfect for seed in (42, 43, 44)},
        "efficientnet_scratch": {seed: imperfect for seed in (42, 43, 44)},
    }

    result = patient_cluster_bootstrap(confusions, labels, ["class_a", "class_b"], 100, 7)

    assert result["transfer"]["pooled_oof_macro_f1_mean_across_seeds"] == 1.0
    assert result["transfer"]["patient_cluster_bootstrap_95_percentile_interval"] == [1.0, 1.0]
    interval = result["paired_differences_vs_transfer"]["scratch"][
        "patient_cluster_bootstrap_95_percentile_interval"
    ]
    assert interval[0] < 0
    assert interval[1] == 0
    assert result["efficientnet_minus_resnet_scratch"][
        "patient_cluster_bootstrap_95_percentile_interval"
    ] == [0.0, 0.0]


def test_research_csv_export_omits_local_paths_and_escapes_formulas():
    import csv
    from io import StringIO

    from mri_classifier.exports import research_results_csv

    content = research_results_csv(
        [
            {
                "id": "run-1",
                "created_at": "2026-10-15T12:00:00+00:00",
                "config": {
                    "dataset_path": "/private/data",
                    "manifest_path": "/private/manifest.csv",
                    "dataset_version": "=2+3",
                    "dataset_kind": "real",
                    "split_strategy": "official_five_fold",
                    "test_fold": 1,
                    "model_family": "classical",
                    "architecture": "resnet18",
                    "training_mode": "scratch",
                    "seed": 42,
                },
                "metrics": {
                    "macro_f1": 0.9,
                    "test_metrics": {"macro_f1": 0.89, "accuracy": 0.9},
                    "report": {
                        "class_names": ["glioma", "meningioma"],
                        "checkpoint": "/private/checkpoint.pt",
                        "hardware": {"device": "cuda"},
                        "git_revision": "abc123",
                    },
                },
            }
        ]
    )
    row = next(csv.DictReader(StringIO(content)))

    assert row["dataset_version"] == "'=2+3"
    assert "dataset_path" not in row
    assert "/private" not in content
    assert "patient_id" not in content


def test_dataset_cohort_identity_uses_source_release_metadata(tmp_path):
    import json

    from mri_classifier.cohorts import dataset_version_identity

    root = tmp_path / "figshare-v5"
    root.mkdir()
    (root / "dataset.json").write_text(
        json.dumps({"source": "https://doi.org/10.6084/m9.figshare.1512427.v5", "version": "5"}),
        encoding="utf-8",
    )
    first = {"dataset_path": str(root), "dataset_version": "figshare-v5"}
    second = {"dataset_path": str(root), "dataset_version": "figshare-1512427-v5"}

    assert dataset_version_identity(first) == dataset_version_identity(second)
    assert "1512427.v5" in dataset_version_identity(first)


def test_near_duplicate_screen_reports_candidates_without_patient_identifiers(tmp_path):
    from PIL import Image, ImageDraw

    from mri_classifier.near_duplicates import screen_near_duplicates

    original = Image.new("L", (128, 128), 0)
    draw = ImageDraw.Draw(original)
    draw.rectangle((24, 24, 96, 96), fill=180)
    draw.ellipse((48, 48, 72, 72), fill=40)
    duplicate = tmp_path / "candidate-a.png"
    original.save(duplicate)
    copied = tmp_path / "candidate-b.png"
    original.copy().save(copied)
    distinct = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(distinct).rectangle((8, 8, 28, 110), fill=255)
    unrelated = tmp_path / "candidate-c.png"
    distinct.save(unrelated)
    rows = [
        {"path": duplicate, "label": "a", "patient_id": "hidden-1", "official_fold": 1},
        {"path": copied, "label": "a", "patient_id": "hidden-2", "official_fold": 2},
        {"path": unrelated, "label": "b", "patient_id": "hidden-3", "official_fold": 3},
    ]

    result = screen_near_duplicates(rows)

    assert result["review_pair_count"] >= 1
    assert result["candidate_pair_counts"]["different_patient_cross_fold"] >= 1
    assert "hidden-1" not in str(result)


def test_dashboard_home_and_live_training_view():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Live Training").run()
    assert not app.exception


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


def test_lightning_gpu_hybrid_forward_backward_when_available():
    torch = pytest.importorskip("torch")
    qml = pytest.importorskip("pennylane")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        qml.device("lightning.gpu", wires=2)
    except (ImportError, RuntimeError, qml.exceptions.DeviceError) as exc:
        pytest.skip(f"PennyLane lightning.gpu is unavailable: {exc}")

    from mri_classifier.models import build_hybrid_model

    model = build_hybrid_model(3, n_qubits=2, depth=1, backend="lightning.gpu").to("cuda")
    inputs = torch.rand(2, 1, 16, 16, device="cuda")
    targets = torch.tensor([0, 2], device="cuda")
    before = model.quantum.weights.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    assert model.quantum.weights.grad is not None
    assert torch.isfinite(model.quantum.weights.grad).all()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert not torch.equal(before, model.quantum.weights.detach())


def test_classical_model_forward_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from mri_classifier.models import build_classical_model

    model = build_classical_model("resnet18", num_classes=3, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.rand(2, 3, 64, 64))
    assert output.shape == (2, 3)


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
    assert report["history"][0]["duration_seconds"] > 0
    assert report["split_strategy"] == "official_five_fold"
    assert report["test_fold"] == 5
    assert (tmp_path / "artifacts" / Path(report["checkpoint"]).name).exists()
    resumed_config = ExperimentConfig(**{**config.to_dict(), "epochs": 2})
    _, resumed_report = run_experiment(
        resumed_config,
        records,
        tmp_path / "resumed-artifacts",
        resume_from=report["checkpoint"],
    )
    assert resumed_report["resumed_from_checkpoint"] == report["checkpoint"]
    assert resumed_report["resumed_from_epoch"] == report["best_epoch"]
    assert resumed_report["resume_state"] == "optimizer_and_scheduler_reset"
    from mri_classifier.inference import predict_image

    prediction = predict_image(report["checkpoint"], records[0]["path"], device="cpu")
    assert prediction["prediction"] in {"class_a", "class_b"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert prediction["research_notice"].startswith("Research output only")
