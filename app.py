"""Local research dashboard; run with `streamlit run app.py`."""

import streamlit as st

from mri_classifier.datasets import inspect_image_folder
from mri_classifier.store import ExperimentStore

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
    st.warning("Training and leakage-safe split execution are not yet wired to this dashboard.")
    dataset = st.text_input("Dataset path", "data")
    family = st.selectbox("Model family", ["Classical", "Hybrid quantum-classical"])
    architecture = st.selectbox(
        "Architecture",
        ["resnet18", "efficientnet_b0"] if family == "Classical" else ["Variational classifier"],
    )
    mode = st.selectbox("Training mode", ["scratch", "transfer"])
    epochs = st.number_input("Epochs", 1, 500, 10)
    if st.button("Save draft configuration"):
        exp = store.create(
            {
                "dataset_path": dataset,
                "family": family,
                "architecture": architecture,
                "training_mode": mode,
                "epochs": int(epochs),
            }
        )
        st.success(f"Draft saved: {exp}")
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
                    "macro_f1": r["metrics"].get("macro_f1"),
                }
                for r in records
            ],
            use_container_width=True,
        )
    if page == "Best Model":
        completed = [
            r for r in records if r["status"] == "completed" and "macro_f1" in r["metrics"]
        ]
        if completed:
            best = max(completed, key=lambda r: r["metrics"]["macro_f1"])
            st.success(
                f"Best overall: {best['id']} (validation macro-F1 {best['metrics']['macro_f1']:.3f})"
            )
        else:
            st.caption("Unavailable until validated experiments are completed.")
else:
    st.info(
        "Compare only completed validated experiments; different datasets and protocols are not directly comparable."
    )
