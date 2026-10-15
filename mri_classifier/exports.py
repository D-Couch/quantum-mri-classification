"""Safe tabular exports for completed research runs."""

import csv
from io import StringIO


def _spreadsheet_safe(value):
    if not isinstance(value, str):
        return value
    if value.lstrip()[:1] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


def research_results_csv(records: list[dict]) -> str:
    """Serialize comparable research-run metrics without local paths or patient data."""
    columns = [
        "experiment_id",
        "created_at",
        "dataset_version",
        "dataset_kind",
        "class_names",
        "split_strategy",
        "test_fold",
        "model_family",
        "architecture",
        "training_mode",
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "qubits",
        "circuit_depth",
        "validation_macro_f1",
        "test_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "duration_seconds",
        "device",
        "num_workers",
        "mixed_precision_enabled",
        "git_revision",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in records:
        config = record.get("config", {})
        metrics = record.get("metrics", {})
        report = metrics.get("report", {})
        test = metrics.get("test_metrics", {})
        performance = report.get("performance", {})
        row = {
            "experiment_id": record.get("id"),
            "created_at": record.get("created_at"),
            "dataset_version": config.get("dataset_version"),
            "dataset_kind": config.get("dataset_kind"),
            "class_names": ", ".join(report.get("class_names", [])),
            "split_strategy": config.get("split_strategy"),
            "test_fold": config.get("test_fold"),
            "model_family": config.get("model_family"),
            "architecture": config.get("architecture"),
            "training_mode": config.get("training_mode"),
            "seed": config.get("seed"),
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "learning_rate": config.get("learning_rate"),
            "qubits": config.get("qubits"),
            "circuit_depth": config.get("circuit_depth"),
            "validation_macro_f1": metrics.get("macro_f1"),
            "test_macro_f1": test.get("macro_f1"),
            "test_accuracy": test.get("accuracy"),
            "test_balanced_accuracy": test.get("balanced_accuracy"),
            "duration_seconds": report.get("duration_seconds"),
            "device": report.get("hardware", {}).get("device"),
            "num_workers": performance.get("num_workers"),
            "mixed_precision_enabled": performance.get("mixed_precision_enabled"),
            "git_revision": report.get("git_revision"),
        }
        writer.writerow({key: _spreadsheet_safe(value) for key, value in row.items()})
    return stream.getvalue()
