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
- [x] Move dashboard training to a cancellable background process.
- [x] Run dashboard training in a single cancellable background worker with persisted epoch progress and duplicate-job prevention.
- [x] Reconcile interrupted jobs after restart; recover final reports without rerunning test evaluation, and offer guarded warm-start from a compatible best checkpoint with parent linkage and optimizer/scheduler reset recorded.
- [x] Verify GitHub Actions on the CPU-only workflow (latest pushed CI passed).

### Milestone 2 — Dataset Pipeline: real dataset imported and verified

- [x] Research licensed dataset source, provenance, classes, patient identity, and published protocol.
- [x] Add Figshare `.mat` importer, manifest integrity checks, duplicate rejection, and patient-group partitioning.
- [x] Retain official five-fold assignments and check patient-level separation on the downloaded release.
- [x] Validate import and manifest integrity on real data without committing the data.
- [x] Run a preliminary scratch baseline on one official fold assignment; repeated folds/seeds remain outstanding.

### Implemented dataset safeguards

- CSV manifest requires `relative_path,label,patient_id`; paths cannot escape the declared image root.
- Byte-identical image content is rejected before splitting; groups are assigned wholly to train/validation/test.
- The trainer refuses small or empty splits and any split missing a declared class. Real-world feasibility and stratification still need evaluation.
- Dataset kind defaults to `unverified`; tests explicitly mark generated fixtures `synthetic`.

### Milestone 3 — Classical Training: runner implemented; research baselines outstanding

- [x] Implement scratch and transfer-capable classical constructors, checkpoint selection by validation macro-F1, metric/report persistence, and inference.
- [x] Run a full-dataset one-epoch ResNet18 smoke execution on official folds (CPU, 3,064 images, 233 patients); artifact remains labeled smoke.
- [x] Run one multi-epoch scratch baseline on real data using official folds (single seed/fold; preliminary).
- [x] Run one transfer baseline on a separate official fold assignment (single seed/fold; preliminary).
- [x] Repeat scratch and transfer on one matched official fold with seeds 42–44; record test macro-F1 mean and sample SD.
- [ ] Repeat across additional official held-out folds to estimate fold-level variation.

### Milestone 4 — Quantum Integration: not started

- [x] Implement trainable hybrid model and gradient/update tests.
- [x] Run one matched-fold hybrid pilot against a scratch ResNet18 control; this single seed/fold result is preliminary.
- [x] Repeat matched scratch/hybrid comparison across seeds 42–44 on official fold 3.
- [ ] Repeat across additional folds and profile simulator alternatives.
- [ ] Profile CPU and compatible accelerated simulators.

### Milestone 5 — Model Selection and Research: not started

- [x] Compare three-seed matched-fold results and report mean/sample SD in the dataset registry.
- [ ] Aggregate uncertainty across multiple official folds and select best overall/classical/hybrid.
- [ ] Add benchmark references with dataset and protocol qualifiers, exports, and model comparison.

### Milestone 6 — Performance Optimization: not started

- [ ] Evaluate architecture/hyperparameter alternatives, ensembles, augmentation, and GPU pipeline.
- [x] Verify the exposed RTX 4090 and measure a full-dataset one-epoch CUDA smoke run (10.16 seconds; smoke only).
- [ ] Profile matched CPU/GPU throughput, optimize input pipeline, and test mixed precision.

## Current blockers and verification

- Current runtime: Python 3.14.8, Linux x86_64. Installed Git 2.56 and GitHub CLI 2.102.0. Outside-sandbox `gh auth status` confirmed D-Couch keyring authentication; GitHub API is reachable; the private repository and remote are verified.
- This runtime exposes an NVIDIA GeForce RTX 4090 with driver 615.78.08, CUDA 13.0, and 24,564 MiB total memory. A full 3,064-image one-epoch ResNet18 smoke run on official folds completed on CUDA in 10.16 seconds; it is labeled `experiment_kind=smoke` and is not a research result. Matched CPU/GPU performance profiling remains outstanding.
- Dependencies installed in `.venv` on Python 3.14: Streamlit, Pillow, NumPy, h5py, PyTorch, torchvision, scikit-learn, PennyLane, PennyLane-Lightning, pytest, and Ruff. Current local suite has 16 passing tests covering metrics, splits, persistence/migration, model forward, hybrid gradients, importer, training/inference, background job progress/cancellation/recovery, and dashboard views.
- ML frameworks are declared in the `ml` extra; GitHub Actions installs CPU-only PyTorch wheels to avoid pulling CUDA runtime packages onto CI runners.
- Latest implementation has manifest-driven dashboard training, metrics, checkpoint selection, report JSON, final test evaluation, and guarded checkpoint warm-start recovery. Synthetic training tests are explicitly labeled. A full-dataset one-epoch run is marked `experiment_kind=smoke` and excluded from best-model selection. On official fold 3, three matched seeds yielded mean test macro-F1 ± sample SD of 0.858 ± 0.018 (scratch), 0.933 ± 0.006 (ImageNet transfer), and 0.497 ± 0.072 (hybrid). This quantifies seed variation on one fold; additional held-out folds are needed to estimate fold variation.
- The evaluation engine reports macro/weighted F1, accuracy, balanced accuracy, per-class precision/recall/sensitivity/specificity, multiclass ROC-AUC and average precision, MCC, kappa, and confusion matrix. Training runs off the Streamlit request path and persists epoch progress. The current one-epoch CUDA smoke report and recovery lineage are stored in SQLite. Classical training uses CUDA when available; current PennyLane CPU simulators remain on CPU to avoid device mismatch.
- Figshare v5 archives are in ignored `data/raw/`; imported images/manifests are in ignored `data/figshare-v5/`. Source details, checksum findings, and class limitations are documented in `docs/DATASETS.md`.

## Decisions to investigate

- Evaluate the Figshare importer against future source versions and inspect duplicate/near-duplicate risks.
- Compare PennyLane Lightning devices against available classical controls empirically.
- Select transfer-learning checkpoints only after confirming license, source, and compatibility.
