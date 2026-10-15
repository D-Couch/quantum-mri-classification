"""Helpers for grouping runs only when their dataset provenance matches."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=64)
def _dataset_source_version(dataset_path: str) -> tuple[str, str] | None:
    metadata_path = Path(dataset_path).expanduser() / "dataset.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source, version = metadata.get("source"), metadata.get("version")
    if not source or not version:
        return None
    return str(source), str(version)


def dataset_version_identity(config: dict) -> str:
    """Use source metadata to reconcile aliases for the same declared dataset release."""
    dataset_path = str(config.get("dataset_path", ""))
    identity = _dataset_source_version(dataset_path)
    if identity:
        source, version = identity
        return f"{source} · version {version}"
    return str(config.get("dataset_version", "unspecified"))
