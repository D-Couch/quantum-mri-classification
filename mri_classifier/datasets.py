"""Safe local image-folder inspection and patient/group split helpers."""

import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def inspect_image_folder(root: str | Path) -> dict:
    """Read an ImageFolder-style tree where each immediate subfolder is a class."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset directory does not exist: {root}")
    classes = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            classes[directory.name] = images
    if not classes:
        raise ValueError("expected class subdirectories containing supported image files")
    return {
        "root": str(root),
        "classes": {k: len(v) for k, v in classes.items()},
        "total": sum(map(len, classes.values())),
        "paths": classes,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_split(
    groups: list[str], seed: int = 42, fractions=(0.7, 0.15, 0.15)
) -> dict[str, list[int]]:
    """Assign whole subject/group IDs to splits deterministically."""
    if not groups or len(fractions) != 3 or any(x <= 0 for x in fractions):
        raise ValueError("provide groups and three positive split fractions")
    total_fraction = sum(fractions)
    fractions = tuple(x / total_fraction for x in fractions)
    unique = list(dict.fromkeys(groups))
    unique.sort(key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())
    n = len(unique)
    first = round(n * fractions[0])
    second = round(n * (fractions[0] + fractions[1]))
    assignment = {
        g: "train" if i < first else "validation" if i < second else "test"
        for i, g in enumerate(unique)
    }
    return {
        name: [i for i, group in enumerate(groups) if assignment[group] == name]
        for name in ("train", "validation", "test")
    }
