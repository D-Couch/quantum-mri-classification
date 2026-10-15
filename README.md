# Quantum Brain MRI Classification Platform

Research software for comparing classical, hybrid quantum-classical, and ensemble methods on brain MRI data. It is not intended for clinical diagnosis. The implementation currently provides a local dataset inspector, subject-group split helper, classification metrics, SQLite experiment index, model constructors, and an early Streamlit dashboard. Full training/evaluation orchestration and a verified real dataset are still in progress.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
streamlit run app.py --server.address 127.0.0.1
```

Place an ImageFolder-style dataset under `data/` (one subdirectory per original class). Check its license, provenance, patient identifiers, duplicates, and official evaluation protocol before using it. Data and artifacts are excluded from Git. Do not treat split utility output as safe patient-level splits unless trustworthy subject IDs are provided.

Consult [requirements](docs/REQUIREMENTS.md) and [roadmap](docs/ROADMAP.md) for scope and verification status. The current development runtime had no usable NVIDIA driver and no installed ML/dashboard dependencies; no GPU or real-data result is claimed.
