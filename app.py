"""Local research dashboard; run with `streamlit run app.py`."""

import streamlit as st

from mri_classifier.config import ExperimentConfig
from mri_classifier.datasets import inspect_image_folder, load_manifest
from mri_classifier.store import ExperimentStore
from mri_classifier.training import run_experiment

st.set_page_config(page_title="Quantum MRI Research", page_icon="🧠", layout="wide")
st.title("Brain MRI Classification Research")
st.caption("Local research platform · not for clinical diagnosis")
page = st.sidebar.radio(
    "View",
    ["Home", "Datasets", "Experiment Configuration", "History", "Model Comparison", "Best Model"],
)
store = ExperimentStore()
records = store.list()
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
            rows = load_manifest(manifest, dataset)
            config = ExperimentConfig(
                dataset_path=dataset,
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
            experiment_id = store.create(config.to_dict())
            store.set_status(experiment_id, "running")
            bar = st.progress(0.0, text="Preparing experiment")
            status = st.empty()

            def update_progress(epoch, total, record):
                bar.progress(epoch / total, text=f"Epoch {epoch}/{total}")
                status.write(
                    f"Train loss {record['train_loss']:.4f} · validation macro-F1 {record['validation']['macro_f1']:.4f}"
                )

            with st.spinner("Training locally; do not close this page until the run finishes."):
                _, report = run_experiment(
                    config, rows, f"artifacts/{experiment_id}", progress=update_progress
                )
            store.finish(
                experiment_id,
                {
                    "macro_f1": report["validation_macro_f1"],
                    "test_metrics": report["test_metrics"],
                    "report": report,
                },
            )
            st.success(
                f"Completed experiment {experiment_id}; validation macro-F1 {report['validation_macro_f1']:.3f}"
            )
            st.json(report["test_metrics"])
        except Exception as exc:  # noqa: BLE001 - persist any user-visible training failure
            if "experiment_id" in locals():
                store.finish(experiment_id, {}, error=str(exc))
            st.error(f"Experiment failed: {exc}")
elif page in {"History", "Model Comparison", "Best Model"}:
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
        if completed:
            best = max(completed, key=lambda r: r["metrics"]["macro_f1"])
            st.success(
                f"Best overall: {best['id']} (validation macro-F1 {best['metrics']['macro_f1']:.3f})"
            )
        else:
            st.caption(
                "Unavailable until real-data research experiments (not smoke runs) are completed."
            )
else:
    st.info(
        "Compare only completed validated experiments; different datasets and protocols are not directly comparable."
    )
