# Roadmap

## Status (2026-10-15)

### Milestone 1 — Foundation: initial slice verified; broader acceptance remains in progress

- [x] Establish package metadata, engineering instructions, authoritative requirements, and roadmap.
- [x] Create private GitHub repository and establish initial Git remote.
- [x] Install isolated dependencies; test metrics, group partitioning, persistence, and hybrid quantum gradient/update.
- [x] Provide initial dataset inspector and dashboard; dashboard training orchestration remains outstanding.
- [ ] Finish trainer, dashboard background jobs, real dataset support, and full CI verification.

### Milestone 2 — Dataset Pipeline: not started

- [ ] Research licensed dataset sources, provenance, and evaluation protocols.
- [ ] Add registry/import, integrity inspection, deduplication, group-aware partitioning, and disclosure of unknown subject identity.
- [ ] Validate on a real dataset without committing the data.

### Milestone 3 — Classical Training: not started

- [ ] Train/evaluate scratch and transfer learning baselines with validation-only selection.
- [ ] Persist histories, checkpoints, metrics, and visualizations; validate inference.

### Milestone 4 — Quantum Integration: not started

- [ ] Implement trainable hybrid model and gradient/update tests.
- [ ] Profile CPU and compatible accelerated simulators.

### Milestone 5 — Model Selection and Research: not started

- [ ] Compare repeated runs and uncertainty; select best overall/classical/hybrid.
- [ ] Add benchmark references with dataset and protocol qualifiers, exports, and model comparison.

### Milestone 6 — Performance Optimization: not started

- [ ] Evaluate architecture/hyperparameter alternatives, ensembles, augmentation, and GPU pipeline.
- [ ] Measure on the target RTX 4090; document any unavailable verification.

## Current blockers and verification

- Current runtime: Python 3.14.8, Linux x86_64. Installed Git 2.56 and GitHub CLI 2.102.0. Outside-sandbox `gh auth status` confirmed D-Couch keyring authentication; GitHub API is reachable; the private repository has been created. Push/remote verification status is recorded below.
- `nvidia-smi` cannot communicate with an NVIDIA driver. This environment does not establish target workstation capability.
- Dependencies installed in `.venv` on Python 3.14: Streamlit, Pillow, NumPy, PyTorch, torchvision, scikit-learn, PennyLane, PennyLane-Lightning, pytest, and Ruff. Pytest currently passes 5 tests; Ruff and compileall pass. The hybrid model passed CPU forward/backward/weight-update verification.
- No MRI dataset or confirmed data license/provenance is available in this workspace.

## Decisions to investigate

- Select an accessible dataset after confirming license, provenance, subject identifiers, duplicate risks, labels, and published protocol.
- Compare PennyLane Lightning devices against available classical controls empirically.
- Select transfer-learning checkpoints only after confirming license, source, and compatibility.
