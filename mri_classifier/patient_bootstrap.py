"""Patient-cluster bootstrap analysis for the fixed Figshare five-fold pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mri_classifier.datasets import load_manifest
from mri_classifier.models import build_classical_model, build_hybrid_model
from mri_classifier.store import ExperimentStore
from mri_classifier.training import ManifestImageDataset

REVISION_BY_FOLD = {
    1: "a1f0184",
    2: "75e67ac",
    3: "9f54446",
    4: "a1f0184",
    5: "a1f0184",
}
SEEDS = (42, 43, 44)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261015
EFFICIENTNET_REVISION_BY_FOLD = {
    1: "01d9511",
    2: "ead30d4",
    3: "ead30d4",
    4: "ead30d4",
    5: "ead30d4",
}


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    scores = []
    for index in range(len(matrix)):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def patient_cluster_bootstrap(
    group_confusions: dict[str, dict[int, dict[str, np.ndarray]]],
    group_labels: dict[str, int],
    labels: list[str],
    n_resamples: int = N_BOOTSTRAP,
    random_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Estimate pooled out-of-fold macro-F1 intervals by resampling whole patients.

    Resampling is stratified by patient class and shared across model families, so
    model comparisons use paired patient samples. Intervals condition on the saved
    checkpoints and observed seeds.
    """
    if n_resamples < 1 or not group_labels:
        raise ValueError("bootstrap requires patient groups and at least one resample")
    group_ids = sorted(group_labels)
    expected = set(group_ids)
    classes = set(range(len(labels)))
    if set(group_labels.values()) != classes:
        raise ValueError("each declared class must have patient groups")
    strata = {
        class_id: [patient for patient in group_ids if group_labels[patient] == class_id]
        for class_id in range(len(labels))
    }
    positions_by_group = {patient: index for index, patient in enumerate(group_ids)}
    family_confusions = {}
    for family, by_seed in group_confusions.items():
        if set(by_seed) != set(SEEDS):
            raise ValueError(f"{family}: expected predictions for seeds {SEEDS}")
        family_confusions[family] = {}
        for seed, matrices in by_seed.items():
            if set(matrices) != expected:
                raise ValueError(f"{family}, seed {seed}: patient groups do not match")
            family_confusions[family][seed] = np.stack([matrices[patient] for patient in group_ids])

    rng = np.random.default_rng(random_seed)
    sampled_positions = np.empty((n_resamples, len(group_ids)), dtype=np.int64)
    for iteration in range(n_resamples):
        sampled_groups = np.concatenate(
            [rng.choice(members, size=len(members), replace=True) for members in strata.values()]
        )
        sampled_positions[iteration] = [positions_by_group[p] for p in sampled_groups]

    draws_by_family = {}
    results = {}
    for family, by_seed in family_confusions.items():
        observed = [matrices.sum(axis=0) for matrices in by_seed.values()]
        point_estimate = float(np.mean([macro_f1_from_confusion(matrix) for matrix in observed]))
        draws = np.empty(n_resamples)
        for iteration, positions in enumerate(sampled_positions):
            seed_scores = [
                macro_f1_from_confusion(matrices[positions].sum(axis=0))
                for matrices in by_seed.values()
            ]
            draws[iteration] = np.mean(seed_scores)
        draws_by_family[family] = draws
        results[family] = {
            "pooled_oof_macro_f1_mean_across_seeds": point_estimate,
            "patient_cluster_bootstrap_95_percentile_interval": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }

    if "transfer" in draws_by_family:
        results["paired_differences_vs_transfer"] = {
            family: {
                "mean_macro_f1_difference": results[family][
                    "pooled_oof_macro_f1_mean_across_seeds"
                ]
                - results["transfer"]["pooled_oof_macro_f1_mean_across_seeds"],
                "patient_cluster_bootstrap_95_percentile_interval": [
                    float(np.quantile(draws - draws_by_family["transfer"], 0.025)),
                    float(np.quantile(draws - draws_by_family["transfer"], 0.975)),
                ],
            }
            for family, draws in draws_by_family.items()
            if family != "transfer"
        }
    if "scratch" in draws_by_family and "efficientnet_scratch" in draws_by_family:
        efficientnet = draws_by_family["efficientnet_scratch"]
        scratch = draws_by_family["scratch"]
        results["efficientnet_minus_resnet_scratch"] = {
            "mean_macro_f1_difference": results["efficientnet_scratch"][
                "pooled_oof_macro_f1_mean_across_seeds"
            ]
            - results["scratch"]["pooled_oof_macro_f1_mean_across_seeds"],
            "patient_cluster_bootstrap_95_percentile_interval": [
                float(np.quantile(efficientnet - scratch, 0.025)),
                float(np.quantile(efficientnet - scratch, 0.975)),
            ],
        }
    return results


def select_runs(records):
    selected = {}
    for record in records:
        config = record["config"]
        report = record["metrics"].get("report", {})
        fold, seed = config.get("test_fold"), config.get("seed")
        if (
            record["status"] != "completed"
            or report.get("experiment_kind") != "research"
            or report.get("split_strategy") != "official_five_fold"
            or fold not in REVISION_BY_FOLD
            or seed not in SEEDS
        ):
            continue
        family = config.get("model_family")
        revision = str(report.get("git_revision", ""))
        if family == "hybrid":
            if not revision.startswith(REVISION_BY_FOLD[fold]):
                continue
            arm = "hybrid"
        elif config.get("architecture") == "efficientnet_b0":
            if (
                config.get("training_mode") != "scratch"
                or config.get("dataset_version") != "figshare-v5"
                or not revision.startswith(EFFICIENTNET_REVISION_BY_FOLD[fold])
            ):
                continue
            arm = "efficientnet_scratch"
        else:
            if not revision.startswith(REVISION_BY_FOLD[fold]):
                continue
            arm = config.get("training_mode")
        key = (arm, seed, fold)
        if key in selected:
            raise ValueError(f"duplicate fixed-protocol model run for {key}")
        selected[key] = record
    expected = 4 * len(SEEDS) * 5
    if len(selected) != expected:
        raise ValueError(f"expected {expected} fixed-protocol runs, found {len(selected)}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/figshare-v5"))
    parser.add_argument("--database", type=Path, default=Path("artifacts/experiments.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/patient-cluster-bootstrap.json"))
    parser.add_argument("--resamples", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    repo = Path.cwd()
    root = args.dataset_root.resolve()
    rows = load_manifest(root / "manifest.csv", root)
    labels = sorted({row["label"] for row in rows})
    label_index = {label: index for index, label in enumerate(labels)}
    records = ExperimentStore(args.database).list()
    selected = select_runs(records)
    groups_by_family_seed = {
        family: {seed: {} for seed in SEEDS}
        for family in ("scratch", "transfer", "hybrid", "efficientnet_scratch")
    }
    group_class = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected {len(selected)} models; predicting on {device}; {len(rows)} images", flush=True)

    for ordinal, ((family, seed, fold), record) in enumerate(sorted(selected.items()), start=1):
        config = record["config"]
        report = record["metrics"]["report"]
        checkpoint_path = repo / report["checkpoint"]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        is_hybrid = family == "hybrid"
        if is_hybrid:
            model = build_hybrid_model(
                len(labels), config["qubits"], config["circuit_depth"], config["quantum_backend"]
            )
            model_device = torch.device("cpu")
        else:
            model = build_classical_model(
                config["architecture"], len(labels), config["training_mode"] == "transfer"
            )
            model_device = device
        model.load_state_dict(state["model"])
        model.to(model_device).eval()
        test_indices = [i for i, row in enumerate(rows) if row["official_fold"] == fold]
        dataset = ManifestImageDataset(
            rows,
            test_indices,
            labels,
            64 if is_hybrid else 224,
            is_hybrid,
            (not is_hybrid and config["training_mode"] == "transfer"),
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        matrices = {}
        cursor = 0
        with torch.inference_mode():
            for images, targets in loader:
                outputs = model(images.to(model_device))
                predictions = outputs.argmax(1).cpu().tolist()
                truths = targets.tolist()
                for truth, prediction in zip(truths, predictions):
                    row = dataset.rows[cursor]
                    patient = row["patient_id"]
                    matrix = matrices.setdefault(patient, np.zeros((len(labels), len(labels)), dtype=np.int64))
                    matrix[truth, prediction] += 1
                    observed_class = label_index[row["label"]]
                    if patient in group_class and group_class[patient] != observed_class:
                        raise ValueError(f"patient group has multiple labels: {patient}")
                    group_class[patient] = observed_class
                    cursor += 1
        if cursor != len(dataset):
            raise ValueError("prediction count does not match the test-fold dataset")
        groups_by_family_seed[family][seed].update(matrices)
        print(
            f"{ordinal}/{len(selected)} {family} seed={seed} fold={fold}: "
            f"{len(matrices)} patients",
            flush=True,
        )
        del model, state, dataset, loader
        if model_device.type == "cuda":
            torch.cuda.empty_cache()

    group_ids = sorted(group_class)
    if len(group_ids) != 233:
        raise ValueError(f"expected 233 unique patient groups, got {len(group_ids)}")
    family_results = patient_cluster_bootstrap(
        groups_by_family_seed,
        group_class,
        labels,
        n_resamples=args.resamples,
        random_seed=args.seed,
    )
    strata_counts = {
        label: sum(group_class[patient] == index for patient in group_ids)
        for index, label in enumerate(labels)
    }

    result = {
        "analysis": "stratified patient-cluster bootstrap of pooled out-of-fold predictions",
        "dataset_version": "Figshare v5",
        "folds": 5,
        "seeds": list(SEEDS),
        "models_per_family": 15,
        "patients": len(group_ids),
        "images": len(rows),
        "class_labels": labels,
        "patient_groups_per_class": strata_counts,
        "bootstrap_resamples": args.resamples,
        "bootstrap_seed": args.seed,
        "selection_metric": "validation_macro_f1",
        "test_data_used_for_tuning": False,
        "results": family_results,
        "limitations": [
            "Intervals describe patient sampling conditional on the trained checkpoints and three seeds.",
            "Fold-specific models are combined as pooled out-of-fold predictions.",
            "Patient resampling is stratified by the dataset class assigned to each patient.",
        ],
    }
    path = args.output if args.output.is_absolute() else repo / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
