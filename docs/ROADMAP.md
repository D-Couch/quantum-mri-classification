# Roadmap

## Status (2026-10-15)

**Priority:** reach the user's target of at least **0.99 validation macro-F1** for the supported image-level, three-class task, while preserving patient-disjoint evaluation. Current best fixed-pilot result is 0.943 ± 0.021 across the five official folds and three seeds; the target is not achieved. Throughput and latency profiles are secondary engineering checks and are not evidence of better classification.

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
- [x] Screen current Figshare v5 for perceptual near-duplicates across patient folds; retain the scan as a review-only heuristic.

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
- [x] Install and validate PennyLane `lightning.gpu` on the RTX 4090; hybrid training and checkpoint inference can run on CUDA.
- [x] Profile matched CPU/GPU quantum simulator throughput. In the current four-qubit smoke profile, `lightning.gpu` used CUDA but was slower than both CPU backends; see `docs/DATASETS.md`.

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

### Milestone 6 — Classification Quality and Generalization: leading pilot model identified; improvement studies outstanding

- [x] Compare scratch EfficientNet-B0 against matched scratch ResNet18 on all five folds and seeds 42–44; use validation macro-F1 for selection and report the patient bootstrap descriptively.
- [x] Confirm EfficientNet-B0 ImageNet weight source and terms caveat; run a three-seed, one-fold transfer pilot and do not extend it because its validation mean did not improve.
- [ ] Evaluate additional pretrained architectures and transfer strategies with confirmed provenance, selecting by validation macro-F1 across patient folds. The one-fold RadImageNet ResNet50 screen was materially below its matched ImageNet control; see `docs/DATASETS.md`.
- [x] Add torchvision ImageNet-pretrained ViT-B/16 as an optional architecture and screen it on one official validation fold with seeds 42–44; its validation mean was below mask-supervised ResNet18, so it remains exploratory and was not extended. See `docs/DATASETS.md`.
- [x] Screen ImageNet-pretrained Swin-T and ConvNeXt-Tiny, plus a fixed validation-only ResNet18/ConvNeXt probability blend, on the same one-fold seed set; none approached the 0.99 target or justified five-fold extension. See `docs/DATASETS.md`.
- [x] Pilot training-only horizontal flips and small rotations on one matched patient fold; results were mixed and inconclusive.
- [x] Enable spatially synchronized image/mask augmentation and screen the leading mask-supervised model on one matched fold with seeds 42–44; mean validation macro-F1 was 0.937 versus 0.951 unaugmented, so it was not advanced. See `docs/DATASETS.md`.
- [x] Assess BRISC as a potential external data source; its public release has no patient IDs and derives from collections including Figshare, so it cannot support verified patient-disjoint expansion. See `docs/DATASETS.md`.
- [ ] Evaluate additional training strategies on matched patient folds, using validation macro-F1; consider ensembles only when validation errors are complementary.
- [ ] Audit recurrent validation errors by patient and class, verify labels/masks/provenance for those cases, then prioritize data expansion and architecture changes that address the error pattern. Never alter or remove valid difficult cases solely to raise the score.
- [x] Screen discriminative encoder learning rate, auxiliary mask-loss weight, and 320-pixel input resolution on one official patient fold with seeds 42–44; none improved the established mask-supervised ResNet18 enough to justify a five-fold extension. Results and limitations are in `docs/DATASETS.md`.
- [x] Audit 35 recurrently misclassified validation images against the official Figshare source records; labels, patient IDs, imported pixels, and masks all matched exactly. This verifies the import path for this sample, not the clinical correctness of source annotations.
- [ ] Assess validation stability across longer runs and seeds; retain checkpoint selection by validation macro-F1.
- [ ] Profile and optimize the GPU input pipeline after classification-quality studies, if throughput limits those experiments.
- [x] Verify the exposed RTX 4090 and measure a full-dataset one-epoch CUDA smoke run (10.16 seconds; smoke only).
- [x] Add explicit device selection, configurable data-loader workers, opt-in CUDA mixed precision, and persist effective settings in reports.
- [x] Record per-epoch durations in experiment history.
- [x] Run a matched one-epoch CPU/GPU and CUDA input-pipeline/AMP smoke profile; results are in `docs/DATASETS.md` and SQLite.
- [x] Repeat each profile setting to three runs and summarize medians/ranges; no warm-up was used.
- [x] Repeat four-epoch profiles across seeds 42–44, treating epoch 1 as warm-up; the two-worker FP32 case was fastest across nine post-warm-up epochs per setting.
- [x] Profile a validation-only training-augmentation pilot on three matched seeds within one patient fold; the result was inconclusive and was not advanced as a model-quality claim.
- [ ] Evaluate host-specific worker defaults before changing pipeline/precision settings.

### Architecture opportunities to evaluate (proposals; validation evidence required)

- **Patient-level multi-instance classification:** The current 3,064 images belong to 233 patients (1–38 images per patient), and the current CNN predicts each slice independently. [x] Screened mean probability pooling on 46 fold-2 validation patients across seeds 42–44; mean patient macro-F1 was 0.944. This is a baseline only and did not establish a gain or justify attention MIL on a single fold. If patient diagnosis is a required output, validate pooling across all official folds before comparing attention-based MIL. Attention-based MIL provides a permutation-invariant way to aggregate labeled bags of instances ([Ilse et al., ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html)). Preserve image-level metrics as a secondary view.
- **Mask-supervised classification:** [x] Retain the source tumor masks and compare the auxiliary mask-supervised, predicted-ROI ResNet18 with the leading ImageNet-transfer ResNet18 across all five official patient folds and seeds 42–44. Validation macro-F1 improved from 0.931 ± 0.027 to 0.943 ± 0.021, with a higher fold mean on all five folds; see `docs/DATASETS.md`. The dashboard defaults to this configuration. This is an internal pilot; test metrics were not used for selection. Further independent dataset validation remains open.
- **Radiology-domain pretraining:** [x] Screened a RadImageNet ResNet50 against its matched ImageNet initialization on one official validation fold. RadImageNet was substantially lower; do not extend without a better justified checkpoint/strategy. See `docs/DATASETS.md`.
- **Ensembling:** Only add probability averaging or a learned ensemble if out-of-fold patient predictions show complementary errors and validation performance improves without test-set tuning.
- **Volumetric models:** Defer 3D CNN/transformer architectures until an ordered 3D volume dataset is available. The current Figshare records are independent 2D images without a reliable slice-order/volume interface.

## Current blockers and verification

- Current runtime: Python 3.14.8, Linux x86_64. Installed Git 2.56 and GitHub CLI 2.102.0. Outside-sandbox `gh auth status` confirmed D-Couch keyring authentication; GitHub API is reachable; the private repository and remote are verified.
- This runtime exposes an NVIDIA GeForce RTX 4090 with driver 615.78.08, CUDA 13.0, and 24,564 MiB total memory. A full 3,064-image one-epoch ResNet18 smoke run on official folds completed on CUDA in 10.16 seconds; it is labeled `experiment_kind=smoke` and is not a research result. Initial repeated CPU/GPU/input-pipeline/AMP and four-epoch warm-up profiles are recorded in `docs/DATASETS.md`; multi-seed steady-state timing and validation-stability checks remain outstanding.
- Dependencies installed in `.venv` on Python 3.14: Streamlit, Pillow, NumPy, h5py, PyTorch, torchvision, scikit-learn, PennyLane, PennyLane-Lightning, PennyLane-Lightning-GPU, pytest, and Ruff. The local suite covers metrics, splits, persistence/migration, model forward, CPU and GPU hybrid gradients, importer, training/inference, background job progress/cancellation/recovery, patient-cluster bootstrap, safe CSV export, dataset cohort identity, perceptual duplicate screening, per-epoch timing, and dashboard views.
- ML frameworks are declared in the `ml` extra; the NVIDIA simulator is an opt-in `quantum-gpu` extra. GitHub Actions installs CPU-only PyTorch wheels to avoid pulling CUDA runtime packages onto CI runners.
- Latest implementation has manifest-driven dashboard training, metrics, checkpoint selection, report JSON, final test evaluation, and guarded checkpoint warm-start recovery. Synthetic training tests are explicitly labeled. A full-dataset one-epoch run is marked `experiment_kind=smoke` and excluded from best-model selection. Seeds 42–44 were evaluated on all five official folds for scratch ResNet18, ImageNet transfer ResNet18, hybrid, and scratch EfficientNet-B0 settings. Per-fold summaries, validation-only selection, and patient-cluster intervals are recorded in `docs/DATASETS.md`; these remain a fixed-protocol research pilot, not a clinical or directly comparable published benchmark.
- The evaluation engine reports macro/weighted F1, accuracy, balanced accuracy, per-class precision/recall/sensitivity/specificity, multiclass ROC-AUC and average precision, MCC, kappa, and confusion matrix. Training runs off the Streamlit request path and persists epoch progress. The current one-epoch CUDA smoke report and recovery lineage are stored in SQLite. Classical training uses CUDA when available; hybrid `lightning.gpu` uses CUDA end to end, while `default.qubit` and `lightning.qubit` remain CPU backends.
- Figshare v5 archives are in ignored `data/raw/`; imported images/manifests are in ignored `data/figshare-v5/`. Source details, checksum findings, and class limitations are documented in `docs/DATASETS.md`.

## Decisions to investigate

- [x] Screen current Figshare v5 for cross-fold perceptual duplicates; the two strongest candidates were visually distinct. Repeat this screen for future dataset releases.
- [x] Compare the installed CPU `default.qubit` and `lightning.qubit` simulators and verify a CUDA forward/backward/update and synthetic training/inference pass with `lightning.gpu`.
- [x] Record the EfficientNet-B0 ImageNet checkpoint source, exact enum, software version, and dataset-license caveat; broader transfer comparisons remain limited by pretrained-data terms and validation results.
