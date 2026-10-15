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
- [x] Verify GitHub Actions on the CPU-only workflow (recovery implementation passed).

### Milestone 2 — Dataset Pipeline: real dataset imported and verified

- [x] Research licensed dataset source, provenance, classes, patient identity, and published protocol.
- [x] Add Figshare `.mat` importer, manifest integrity checks, duplicate rejection, and patient-group partitioning.
- [x] Retain official five-fold assignments and check patient-level separation on the downloaded release.
- [x] Validate import and manifest integrity on real data without committing the data.
- [x] Run a preliminary scratch baseline on one official fold assignment; expanded matched experiments are tracked in Milestones 3–5.

### Implemented dataset safeguards

- CSV manifest requires `relative_path,label,patient_id`; paths cannot escape the declared image root.
- Byte-identical image content is rejected before splitting; groups are assigned wholly to train/validation/test.
- The trainer refuses small or empty splits and any split missing a declared class. Real-world feasibility and stratification still need evaluation.
- Dataset kind defaults to `unverified`; tests explicitly mark generated fixtures `synthetic`.

### Milestone 3 — Classical Training: matched five-fold pilot complete; broader baselines outstanding

- [x] Implement scratch and transfer-capable classical constructors, checkpoint selection by validation macro-F1, metric/report persistence, and inference.
- [x] Run a full-dataset one-epoch ResNet18 smoke execution on official folds (CPU, 3,064 images, 233 patients); artifact remains labeled smoke.
- [x] Run one multi-epoch scratch baseline on real data using official folds (single seed/fold; preliminary).
- [x] Run one transfer baseline on a separate official fold assignment (single seed/fold; preliminary).
- [x] Repeat scratch and transfer on all five matched official folds with seeds 42–44; record per-fold test macro-F1 mean and sample SD.

### Milestone 4 — Quantum Integration: matched five-fold pilot complete; wider profiling in progress

- [x] Implement trainable hybrid model and gradient/update tests.
- [x] Run one matched-fold hybrid pilot against a scratch ResNet18 control; this single seed/fold result is preliminary.
- [x] Repeat matched scratch/hybrid comparison across seeds 42–44 on all five official folds.
- [x] Profile `default.qubit` and `lightning.qubit` on a matched one-epoch CPU smoke run; results are in `docs/DATASETS.md`.
- [ ] Profile accelerated quantum simulators when a compatible backend is available (`lightning.gpu` is not installed/exposed here).

### Milestone 5 — Model Selection and Research: fixed-protocol five-fold pilot and patient bootstrap complete

- [x] Compare three-seed matched-fold results and report per-fold mean/sample SD for all five official folds.
- [x] Aggregate validation macro-F1 across folds and seeds for validation-based family selection; report test results descriptively without using them for selection.
- [x] Select ImageNet transfer as the leading family in this fixed pilot: it has the highest mean validation macro-F1 on each fold.
- [x] Add patient-cluster bootstrap intervals for pooled out-of-fold predictions across four model families, conditional on the checkpoints and three seeds.
- [x] Provide dashboard best overall/classical/hybrid selection from completed real-data research runs using validation macro-F1.
- [x] Cite Cheng et al. and the Figshare dataset with protocol/class qualifiers; document why the published methods are not a direct comparison.
- [x] Add a CSV export for comparable completed real-data research runs; keep local paths, checkpoints, and patient identifiers out of the export.
- [x] Provide the model comparison view.
- [x] Resolve dataset cohort identity from `dataset.json` source/version metadata so harmless version-label aliases do not split the same release into separate comparisons.

### Milestone 6 — Performance Optimization: RTX 4090 verified; optimization profiling outstanding

- [x] Compare scratch EfficientNet-B0 against matched scratch ResNet18 on all five folds and seeds 42–44; use validation macro-F1 for selection and report the patient bootstrap descriptively.
- [x] Confirm EfficientNet-B0 ImageNet weight source and terms caveat; run a three-seed, one-fold transfer pilot and do not extend it because its validation mean did not improve.
- [ ] Evaluate additional architectures and transfer alternatives with confirmed provenance; then consider ensembles and augmentation.
- [ ] Profile and optimize the GPU input pipeline after repeated warm-up measurements.
- [x] Verify the exposed RTX 4090 and measure a full-dataset one-epoch CUDA smoke run (10.16 seconds; smoke only).
- [x] Add explicit device selection, configurable data-loader workers, opt-in CUDA mixed precision, and persist effective settings in reports.
- [x] Record per-epoch durations in experiment history.
- [x] Run a matched one-epoch CPU/GPU and CUDA input-pipeline/AMP smoke profile; results are in `docs/DATASETS.md` and SQLite.
- [x] Repeat each profile setting to three runs and summarize medians/ranges; no warm-up was used.
- [x] Run a four-epoch timing profile with epoch 1 as warm-up; the two-worker FP32 case was fastest in this single profile.
- [ ] Repeat multi-epoch profiles across seeds, assess validation stability, and optimize pipeline/precision settings before changing defaults.

## Current blockers and verification

- Current runtime: Python 3.14.8, Linux x86_64. Installed Git 2.56 and GitHub CLI 2.102.0. Outside-sandbox `gh auth status` confirmed D-Couch keyring authentication; GitHub API is reachable; the private repository and remote are verified.
- This runtime exposes an NVIDIA GeForce RTX 4090 with driver 615.78.08, CUDA 13.0, and 24,564 MiB total memory. A full 3,064-image one-epoch ResNet18 smoke run on official folds completed on CUDA in 10.16 seconds; it is labeled `experiment_kind=smoke` and is not a research result. An initial matched one-epoch CPU/GPU/input-pipeline profile is recorded in `docs/DATASETS.md`; repeated timing and validation-stability checks remain outstanding.
- Dependencies installed in `.venv` on Python 3.14: Streamlit, Pillow, NumPy, h5py, PyTorch, torchvision, scikit-learn, PennyLane, PennyLane-Lightning, pytest, and Ruff. Current local suite has 19 passing tests covering metrics, splits, persistence/migration, model forward, hybrid gradients, importer, training/inference, background job progress/cancellation/recovery, patient-cluster bootstrap, safe CSV export, dataset cohort identity, and dashboard views.
- ML frameworks are declared in the `ml` extra; GitHub Actions installs CPU-only PyTorch wheels to avoid pulling CUDA runtime packages onto CI runners.
- Latest implementation has manifest-driven dashboard training, metrics, checkpoint selection, report JSON, final test evaluation, and guarded checkpoint warm-start recovery. Synthetic training tests are explicitly labeled. A full-dataset one-epoch run is marked `experiment_kind=smoke` and excluded from best-model selection. Seeds 42–44 were evaluated on all five official folds for scratch ResNet18, ImageNet transfer ResNet18, hybrid, and scratch EfficientNet-B0 settings. Per-fold summaries, validation-only selection, and patient-cluster intervals are recorded in `docs/DATASETS.md`; these remain a fixed-protocol research pilot, not a clinical or directly comparable published benchmark.
- The evaluation engine reports macro/weighted F1, accuracy, balanced accuracy, per-class precision/recall/sensitivity/specificity, multiclass ROC-AUC and average precision, MCC, kappa, and confusion matrix. Training runs off the Streamlit request path and persists epoch progress. The current one-epoch CUDA smoke report and recovery lineage are stored in SQLite. Classical training uses CUDA when available; current PennyLane CPU simulators remain on CPU to avoid device mismatch.
- Figshare v5 archives are in ignored `data/raw/`; imported images/manifests are in ignored `data/figshare-v5/`. Source details, checksum findings, and class limitations are documented in `docs/DATASETS.md`.

## Decisions to investigate

- Evaluate the Figshare importer against future source versions and inspect duplicate/near-duplicate risks.
- Compare PennyLane Lightning devices against available classical controls empirically.
- Select transfer-learning checkpoints only after confirming license, source, and compatibility.
