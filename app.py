"""Local research dashboard; run with `streamlit run app.py`."""

import streamlit as st

from mri_classifier.config import ExperimentConfig
from mri_classifier.datasets import inspect_image_folder
from mri_classifier.jobs import TrainingManager
from mri_classifier.store import ExperimentStore

st.set_page_config(page_title="Quantum MRI Research", page_icon="🧠", layout="wide")
st.title("Brain MRI Classification Research")
st.caption("Local research platform · not for clinical diagnosis")
page = st.sidebar.radio(
    "View",
    [
        "Home",
        "Datasets",
        "Experiment Configuration",
        "Live Training",
        "History",
        "Model Comparison",
        "Best Model",
    ],
)
store = ExperimentStore()
records = store.list()


@st.cache_resource
def training_manager():
    return TrainingManager(store)


if page == "Home":
    a, b, c = st.columns(3)
    a.metric("Saved experiments", len(records))
    b.metric("Datasets", "Not configured")
    c.metric("GPU", "Unverified")
    st.info(
        "Inspect an ImageFolder dataset to begin. No medical performance claims are made until real data has been evaluated."
    )
elif page == "Datasets":
    path = st.text_input("ImageFolder root", "data")
    if st.button("Inspect dataset"):
        try:
            summary = inspect_image_folder(path)
            st.success(f"Found {summary['total']} images in {len(summary['classes'])} classes")
            st.json({"root": summary["root"], "classes": summary["classes"]})
        except ValueError as exc:
            st.error(str(exc))
    st.caption(
        "Layout: dataset/class_name/image.png. Confirm license, provenance and subject identifiers before research use."
    )
elif page == "Experiment Configuration":
    st.info(
        "Training uses a CSV manifest with `relative_path,label,patient_id`; patient IDs are required for leakage-safe splits."
    )
    dataset = st.text_input("Dataset image root", "data")
    manifest = st.text_input("Manifest CSV", "data/manifest.csv")
    version = st.text_input("Dataset version or source identifier", "unspecified")
    kind = st.selectbox("Data status", ["unverified", "real", "synthetic"])
    split_strategy = st.selectbox("Split protocol", ["group_holdout", "official_five_fold"])
    test_fold = st.selectbox("Official test fold", [1, 2, 3, 4, 5], index=4)
    experiment_kind = st.selectbox("Run type", ["research", "smoke"])
    family = st.selectbox("Model family", ["Classical", "Hybrid quantum-classical"])
    architecture = st.selectbox(
        "Architecture",
        ["resnet18", "efficientnet_b0"] if family == "Classical" else ["Variational classifier"],
    )
    mode = st.selectbox("Training mode", ["scratch", "transfer"])
    epochs = st.number_input("Epochs", 1, 500, 10)
    batch_size = st.number_input("Batch size", 1, 512, 16)
    learning_rate = st.number_input(
        "Learning rate", min_value=0.000001, max_value=1.0, value=0.001, format="%.6f"
    )
    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42)
    qubits = st.number_input("Qubits (hybrid)", 1, 12, 4)
    depth = st.number_input("Circuit depth (hybrid)", 1, 12, 2)
    backend = st.text_input("PennyLane simulator (hybrid)", "default.qubit")
    if st.button("Train and evaluate", type="primary"):
        try:
            config = ExperimentConfig(
                dataset_path=dataset,
                manifest_path=manifest,
                dataset_version=version,
                dataset_kind=kind,
                model_family="hybrid" if family.startswith("Hybrid") else "classical",
                split_strategy=split_strategy,
                test_fold=int(test_fold),
                experiment_kind=experiment_kind,
                architecture=architecture if family == "Classical" else "resnet18",
                training_mode=mode,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                seed=int(seed),
                qubits=int(qubits),
                circuit_depth=int(depth),
                quantum_backend=backend,
            )
            experiment_id = training_manager().submit(config)
            st.success(f"Training job {experiment_id} started. Follow progress in Live Training.")
        except Exception as exc:  # noqa: BLE001 - persist any user-visible training failure
            st.error(f"Experiment failed: {exc}")
elif page == "Live Training":

    @st.fragment(run_every="1s")
    def live_training_view():
        latest = store.list()
        active = next((r for r in latest if r["status"] in {"queued", "running"}), None)
        if active is None:
            st.info("No active training job.")
            return
        progress = active.get("progress", {})
        current = progress.get("epoch", 0)
        total = progress.get("epochs", 1)
        st.write(f"Experiment {active['id']} · {active['status']}")
        st.progress(min(current / total, 1.0), text=f"Epoch {current}/{total}")
        if progress:
            st.write(
                f"Train loss {progress['train_loss']:.4f} · validation macro-F1 {progress['validation']['macro_f1']:.4f}"
            )
        if st.button("Cancel training", type="secondary") and training_manager().cancel():
            st.warning("Cancellation requested; it will stop after the current epoch.")

    live_training_view()
elif page in {"History", "Model Comparison", "Best Model"}:
    if page == "Model Comparison":
        st.caption("Compare scores only when dataset, class definitions, and split protocol match.")
    if not records:
        st.info("No experiments saved yet.")
    else:
        st.dataframe(
            [
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "status": r["status"],
                    "dataset": r["config"].get("dataset_path"),
                    "data_kind": r["config"].get("dataset_kind"),
                    "run_kind": r["config"].get("experiment_kind", "research"),
                    "macro_f1": r["metrics"].get("macro_f1"),
                }
                for r in records
            ],
            use_container_width=True,
        )
    if page == "Best Model":
        completed = [
            r
            for r in records
            if r["status"] == "completed"
            and r["config"].get("dataset_kind") == "real"
            and r["config"].get("experiment_kind", "research") == "research"
            and "macro_f1" in r["metrics"]
        ]

        def cohort_key(record):
            config = record["config"]
            report = record["metrics"].get("report", {})
            return (
                config.get("dataset_path"),
                config.get("dataset_version"),
                tuple(report.get("class_names", ())),
                config.get("split_strategy"),
                config.get("test_fold")
                if config.get("split_strategy") == "official_five_fold"
                else None,
            )

        cohorts = sorted({cohort_key(record) for record in completed}, key=str)
        cohort_names = {
            f"{dataset} · {version} · {','.join(classes)} · {split}"
            + (f" test fold {fold}" if fold else ""): key
            for key in cohorts
            for dataset, version, classes, split, fold in [key]
        }
        if len(cohort_names) > 1:
            chosen_cohort = st.selectbox("Comparable dataset and split cohort", list(cohort_names))
            completed = [
                record for record in completed if cohort_key(record) == cohort_names[chosen_cohort]
            ]
        categories = {
            "Best overall": completed,
            "Best classical": [
                r for r in completed if r["config"].get("model_family") == "classical"
            ],
            "Best quantum/hybrid": [
                r for r in completed if r["config"].get("model_family") == "hybrid"
            ],
        }
        columns = st.columns(len(categories))
        for column, (title, candidates) in zip(columns, categories.items()):
            if candidates:
                best = max(candidates, key=lambda r: r["metrics"]["macro_f1"])
                column.metric(title, f"{best['metrics']['macro_f1']:.3f}", help=best["id"])
            else:
                column.metric(title, "Unavailable")
        if completed:
            st.subheader("Inference with best validated overall model")
            best = max(completed, key=lambda r: r["metrics"]["macro_f1"])
            report = best["metrics"].get("report", {})
            uploaded = st.file_uploader(
                "Choose a compatible MRI image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
            )
            if uploaded and report.get("checkpoint"):
                from io import BytesIO

                from mri_classifier.inference import predict_image

                prediction = predict_image(report["checkpoint"], BytesIO(uploaded.getvalue()))
                st.write(f"Predicted class: **{prediction['prediction']}**")
                st.bar_chart(prediction["probabilities"])
                st.caption(prediction["research_notice"])
        else:
            st.caption(
                "Inference is available after a real-data research model has been validated."
            )
    elif page == "Model Comparison":
        comparable = [
            r
            for r in records
            if r["status"] == "completed"
            and r["config"].get("dataset_kind") == "real"
            and r["config"].get("experiment_kind", "research") == "research"
            and "macro_f1" in r["metrics"]
        ]
        if comparable:
            comparison_keys = {
                (
                    r["config"].get("dataset_path"),
                    r["config"].get("dataset_version"),
                    tuple(r["metrics"].get("report", {}).get("class_names", ())),
                    r["config"].get("split_strategy"),
                    r["config"].get("test_fold"),
                )
                for r in comparable
            }
            for key in sorted(comparison_keys, key=str):
                same = [
                    r
                    for r in comparable
                    if (
                        r["config"].get("dataset_path"),
                        r["config"].get("dataset_version"),
                        tuple(r["metrics"].get("report", {}).get("class_names", ())),
                        r["config"].get("split_strategy"),
                        r["config"].get("test_fold"),
                    )
                    == key
                ]
                st.subheader(" · ".join(str(part) for part in key if part is not None))
                st.dataframe(
                    [
                        {
                            "id": r["id"],
                            "model": r["config"].get("model_family"),
                            "architecture": r["config"].get("architecture"),
                            "validation macro-F1": r["metrics"]["macro_f1"],
                            "test accuracy": r["metrics"].get("test_metrics", {}).get("accuracy"),
                        }
                        for r in same
                    ],
                    use_container_width=True,
                )
        else:
            st.info(
                "No real-data research runs are available to compare. Smoke and synthetic runs are excluded."
            )
else:
    st.info(
        "Compare only completed validated experiments; different datasets and protocols are not directly comparable."
    )
