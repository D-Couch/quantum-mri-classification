# Roadmap

## Status (2026-10-15)

### Milestone 1 — Foundation: initial slice verified; broader acceptance remains in progress

- [x] Establish package metadata, engineering instructions, authoritative requirements, and roadmap.
- [x] Create private GitHub repository and establish initial Git remote.
- [x] Install isolated dependencies; test metrics, group partitioning, persistence, and hybrid quantum gradient/update.
- [x] Provide dataset inspector and dashboard training/evaluation workflow.
- [x] Add a reusable, manifest-driven trainer and dashboard training/evaluation workflow.
- [x] Add checkpoint loading and in-memory single-image inference; smoke-test probabilities sum to one.
- [x] Restrict best-model selection and comparisons to matched dataset/class/split cohorts; exclude synthetic and smoke runs.
- [ ] Move dashboard training to a cancellable background process and add checkpoint recovery.
- [x] Run dashboard training in a single cancellable background worker with persisted epoch progress and duplicate-job prevention.
- [ ] Add safe recovery/resumption of interrupted jobs from checkpoints.
- [x] Verify GitHub Actions on the CPU-only workflow (latest pushed CI passed).

### Milestone 2 — Dataset Pipeline: real dataset imported and verified

- [x] Research licensed dataset source, provenance, classes, patient identity, and published protocol.
- [x] Add Figshare `.mat` importer, manifest integrity checks, duplicate rejection, and patient-group partitioning.
- [x] Retain official five-fold assignments and check patient-level separation on the downloaded release.
- [x] Validate import and manifest integrity on real data without committing the data.
- [ ] Train a baseline on the real dataset with the official folds.

### Implemented dataset safeguards

- CSV manifest requires `relative_path,label,patient_id`; paths cannot escape the declared image root.
- Byte-identical image content is rejected before splitting; groups are assigned wholly to train/validation/test.
- The trainer refuses small or empty splits and any split missing a declared class. Real-world feasibility and stratification still need evaluation.
- Dataset kind defaults to `unverified`; tests explicitly mark generated fixtures `synthetic`.

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

- Current runtime: Python 3.14.8, Linux x86_64. Installed Git 2.56 and GitHub CLI 2.102.0. Outside-sandbox `gh auth status` confirmed D-Couch keyring authentication; GitHub API is reachable; the private repository and remote are verified.
- `nvidia-smi` cannot communicate with an NVIDIA driver. This environment does not establish target workstation capability.
- Dependencies installed in `.venv` on Python 3.14: Streamlit, Pillow, NumPy, h5py, PyTorch, torchvision, scikit-learn, PennyLane, PennyLane-Lightning, pytest, and Ruff. Current local suite has 15 passing tests covering metrics, splits, persistence/migration, model forward, hybrid gradients, importer, training/inference, background job progress/cancellation, and dashboard views.
- ML frameworks are declared in the `ml` extra; GitHub Actions installs CPU-only PyTorch wheels to avoid pulling CUDA runtime packages onto CI runners.
- Latest implementation has manifest-driven dashboard training, metrics, checkpoint selection, report JSON, and final test evaluation. Synthetic training tests are explicitly labeled. A real-data smoke run executed one CPU epoch on a deliberately small 15-patient subset, marked `experiment_kind=smoke`; its metrics are not research findings and are excluded from best-model selection. Full-dataset repeated baseline experiments remain outstanding.
- The evaluation engine reports macro/weighted F1, accuracy, balanced accuracy, per-class precision/recall/sensitivity/specificity, multiclass ROC-AUC and average precision, MCC, kappa, and confusion matrix. Training now runs off the Streamlit request path and persists epoch progress. Checks currently pass locally; the latest dashboard/test changes have not yet been pushed.
- Figshare v5 archives are in ignored `data/raw/`; imported images/manifests are in ignored `data/figshare-v5/`. Source details, checksum findings, and class limitations are documented in `docs/DATASETS.md`.

## Decisions to investigate

- Evaluate the Figshare importer against future source versions and inspect duplicate/near-duplicate risks.
- Compare PennyLane Lightning devices against available classical controls empirically.
- Select transfer-learning checkpoints only after confirming license, source, and compatibility.
