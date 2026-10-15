"""Read-only perceptual near-duplicate screening for a manifest dataset."""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from .datasets import load_manifest


def _difference_hash(image: Image.Image) -> int:
    pixels = np.asarray(
        image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    result = 0
    for row in range(8):
        for column in range(8):
            result = (result << 1) | int(pixels[row, column] > pixels[row, column + 1])
    return result


def _structural_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mean_first, mean_second = float(first.mean()), float(second.mean())
    variance_first, variance_second = float(first.var()), float(second.var())
    covariance = float(((first - mean_first) * (second - mean_second)).mean())
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (mean_first**2 + mean_second**2 + c1) * (
        variance_first + variance_second + c2
    )
    return numerator / denominator


def screen_near_duplicates(
    rows: list[dict], hash_distance: int = 3, similarity_threshold: float = 0.90
) -> dict:
    """Return perceptual review candidates; never automatically rejects or removes images."""
    if hash_distance < 0 or not 0 <= similarity_threshold <= 1:
        raise ValueError("hash distance must be non-negative and similarity threshold in [0, 1]")
    hashes, thumbnails = [], []
    for row in rows:
        with Image.open(row["path"]) as image:
            hashes.append(_difference_hash(image))
            thumbnails.append(
                np.asarray(
                    image.convert("L").resize((128, 128), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            )

    pair_counts = Counter()
    review_pairs = []
    maximum_cross_patient_similarity = 0.0
    for first, second in combinations(range(len(rows)), 2):
        distance = (hashes[first] ^ hashes[second]).bit_count()
        if distance > hash_distance:
            continue
        a, b = rows[first], rows[second]
        same_patient = a["patient_id"] == b["patient_id"]
        same_fold = a.get("official_fold") == b.get("official_fold")
        relation = f"{'same' if same_patient else 'different'}_patient_"
        relation += f"{'same' if same_fold else 'cross'}_fold"
        pair_counts[relation] += 1
        if same_patient or same_fold:
            continue
        similarity = _structural_similarity(thumbnails[first], thumbnails[second])
        maximum_cross_patient_similarity = max(maximum_cross_patient_similarity, similarity)
        if similarity >= similarity_threshold:
            review_pairs.append(
                {
                    "first_image": Path(a["path"]).name,
                    "first_label": a["label"],
                    "first_fold": a.get("official_fold"),
                    "second_image": Path(b["path"]).name,
                    "second_label": b["label"],
                    "second_fold": b.get("official_fold"),
                    "difference_hash_distance": distance,
                    "structural_similarity": similarity,
                }
            )

    return {
        "images_scanned": len(rows),
        "difference_hash_bits": 64,
        "difference_hash_distance_threshold": hash_distance,
        "structural_similarity_review_threshold": similarity_threshold,
        "candidate_pair_counts": dict(pair_counts),
        "cross_patient_cross_fold_max_similarity": maximum_cross_patient_similarity,
        "review_pair_count": len(review_pairs),
        "review_pairs": review_pairs,
        "interpretation": "Perceptual matches are review candidates, not confirmed duplicates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("artifacts/near-duplicate-review.json"))
    parser.add_argument("--hash-distance", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    args = parser.parse_args()
    rows = load_manifest(args.manifest, args.image_root)
    report = screen_near_duplicates(rows, args.hash_distance, args.similarity_threshold)
    output = args.report
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
