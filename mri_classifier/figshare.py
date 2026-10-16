"""Importer for Cheng et al.'s Figshare MATLAB 7.3 brain tumor dataset."""

import csv
import hashlib
import json
from pathlib import Path

LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
SOURCE = "https://doi.org/10.6084/m9.figshare.1512427.v5"


def import_figshare_directory(
    source_dir: str | Path, output_dir: str | Path, cvind_path: str | Path | None = None
) -> dict:
    """Convert extracted `.mat` records into images plus a patient-aware CSV manifest.

    Source archives must be obtained under the dataset's CC BY 4.0 terms. This function
    only processes local files and never downloads or embeds source data in the repository.
    """
    try:
        import h5py
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Install h5py, numpy and Pillow to import Figshare MATLAB files"
        ) from exc

    source = Path(source_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    mat_files = sorted(source.rglob("*.mat"))
    if not mat_files:
        raise ValueError(f"no MATLAB .mat files found under {source}")
    fold_values = None
    if cvind_path is not None:
        with h5py.File(cvind_path, "r") as folds_file:
            if "cvind" not in folds_file:
                raise ValueError("cvind file does not contain the published cvind array")
            fold_values = np.asarray(folds_file["cvind"]).squeeze().astype(int)
        if fold_values.shape != (3064,) or not set(fold_values).issubset({1, 2, 3, 4, 5}):
            raise ValueError("cvind must contain one fold ID (1..5) for each of 3064 samples")
    rows, seen_pids, fold_by_pid = [], set(), {}
    mask_presence = set()
    for mat_path in mat_files:
        with h5py.File(mat_path, "r") as handle:
            if "cjdata" not in handle:
                raise ValueError(f"{mat_path.name} has no cjdata record")
            data = handle["cjdata"]
            try:
                label_id = int(np.asarray(data["label"]).squeeze())
                pid_value = np.asarray(data["PID"]).squeeze()
                if pid_value.ndim == 0:
                    scalar = pid_value.item()
                    pid = scalar.decode() if isinstance(scalar, bytes) else str(scalar)
                elif pid_value.dtype.kind in "ui":
                    pid = "".join(chr(int(char)) for char in pid_value.ravel()).strip("\x00")
                else:
                    raise ValueError("unsupported patient ID encoding")
                image = np.asarray(data["image"]).squeeze()
                mask = np.asarray(data["tumorMask"]).squeeze() if "tumorMask" in data else None
            except (KeyError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid Figshare record: {mat_path.name}") from exc
        if label_id not in LABELS or image.ndim != 2 or not pid:
            raise ValueError(f"invalid label, patient ID, or 2D image in {mat_path.name}")
        if image.size == 0 or not np.isfinite(image).all():
            raise ValueError(f"empty or non-finite MRI image: {mat_path.name}")
        mask_presence.add(mask is not None)
        if mask is not None and (mask.shape != image.shape or mask.ndim != 2):
            raise ValueError(f"invalid tumor mask shape in {mat_path.name}")
        image = image.astype(np.float32)
        low, high = float(image.min()), float(image.max())
        image = (
            np.zeros_like(image, dtype=np.uint8)
            if high <= low
            else np.clip((image - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        )
        class_name = LABELS[label_id]
        source_key = mat_path.relative_to(source).as_posix().encode("utf-8")
        suffix = hashlib.sha256(source_key).hexdigest()[:10]
        relative = Path(class_name) / f"{mat_path.stem}-{suffix}.png"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="L").save(target)
        record = [relative.as_posix(), class_name, pid]
        if mask is not None:
            mask_relative = Path("masks") / class_name / f"{mat_path.stem}-{suffix}.png"
            mask_target = destination / mask_relative
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            binary_mask = (mask > 0).astype(np.uint8) * 255
            Image.fromarray(binary_mask, mode="L").save(mask_target)
            record.append(mask_relative.as_posix())
        if fold_values is not None:
            sample_id = int(mat_path.stem)
            if not 1 <= sample_id <= 3064:
                raise ValueError(f"unexpected sample ID {sample_id} in {mat_path}")
            fold = int(fold_values[sample_id - 1])
            if pid in fold_by_pid and fold_by_pid[pid] != fold:
                raise ValueError(f"published folds separate slices for patient {pid}")
            fold_by_pid[pid] = fold
            record.append(fold)
        rows.append(record)
        seen_pids.add(pid)

    if len(mask_presence) > 1:
        raise ValueError("source records must either all include tumor masks or all omit them")
    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["relative_path", "label", "patient_id"]
        if mask_presence == {True}:
            header.append("mask_relative_path")
        if fold_values is not None:
            header.append("official_fold")
        writer.writerow(header)
        writer.writerows(rows)
    metadata = {
        "dataset": "Cheng et al. brain tumor dataset",
        "version": "5",
        "source": SOURCE,
        "license": "CC BY 4.0",
        "classes": LABELS,
        "image_count": len(rows),
        "patient_count": len(seen_pids),
        "mask_count": len(rows) if mask_presence == {True} else 0,
        "manifest": str(manifest),
        "limitation": "Three tumor classes only; no no-tumor class.",
        "split_protocol": "Published patient-disjoint five-fold IDs"
        if fold_values is not None
        else "Patient IDs retained; split protocol not imported",
    }
    (destination / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
