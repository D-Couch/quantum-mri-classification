# Dataset registry

## Cheng et al. brain tumor dataset (Figshare)

- **Source:** [Figshare v5](https://doi.org/10.6084/m9.figshare.1512427.v5)
- **Download files:** [Part 1](https://ndownloader.figshare.com/files/3381290), [part 2](https://ndownloader.figshare.com/files/3381296), [part 3](https://ndownloader.figshare.com/files/3381293), [part 4](https://ndownloader.figshare.com/files/3381302), [published fold IDs](https://ndownloader.figshare.com/files/7005344).
- **License:** CC BY 4.0, as shown on the [Figshare record](https://figshare.com/articles/dataset/brain_tumor_data-set/1512427/5?report=reader).
- **Provenance:** Cheng et al. report 3,064 T1-weighted contrast-enhanced 2D slices from 233 patients acquired at two hospitals from 2005–2010. The dataset contains 708 meningioma, 1,426 glioma, and 930 pituitary tumor samples; patient information was anonymized. The paper describes partitioning patients into five folds to keep each patient's slices together.
- **Original fields:** MATLAB `.mat` records contain image, label, patient ID, tumor boundary and mask. Label mapping is `1=meningioma`, `2=glioma`, `3=pituitary`.
- **Limitation:** This dataset has no no-tumor class. It must be reported as a three-class problem. The study's original results used handcrafted ROI/feature methods and its fold protocol; they are not directly comparable to this platform's image-level models unless the same protocol is reproduced.
- **Import:** Extract the v5 `.mat` files and call `mri_classifier.figshare.import_figshare_directory(source_dir, output_dir, cvind_path)`. The importer creates grayscale images, `manifest.csv` with patient IDs and published fold IDs, and `dataset.json`. It does not download or commit the source data.
- **Integrity and protocol:** The importer checks patient IDs against the official `cvind.mat` five-fold assignment. The trainer can hold one fold for test and the next for validation, with remaining folds for training. For other datasets it offers deterministic patient-group holdout. Manifest paths must stay under the image root and byte-identical images are rejected.
- **Import verification:** All four archive MD5s matched official Figshare API metadata. Import yielded 3,064 readable images, 233 patient IDs, expected class counts, and zero patient/fold conflicts.
- **Near-duplicate screening:** The read-only command `python -m mri_classifier.near_duplicates --manifest data/figshare-v5/manifest.csv --image-root data/figshare-v5` compares 64-bit difference hashes (distance ≤3) and flags cross-patient/cross-fold candidates with global SSIM ≥0.90 on 128×128 grayscale thumbnails for manual review. On v5 it found 80 cross-patient/cross-fold hash candidates and two SSIM review candidates. Both were visually reviewed and had distinct lesion structures; no near-duplicate was confirmed. This heuristic is not proof that all near-duplicates are absent. Official patient-disjoint folds remain the primary leakage safeguard. The ignored local review output is `artifacts/near-duplicate-review.json` and contains filenames only, not patient identifiers.
- **Initial measured baseline (preliminary, not a final benchmark):** ResNet18 trained from scratch, seed 42, official fold 5 held out for test, fold 1 for validation, and the remaining folds for training. Checkpoint selection used validation macro-F1; early stopping ended after 8 epochs, with best epoch 3. On the 643-image test fold: accuracy 0.829, macro-F1 0.822, balanced accuracy 0.831, macro ROC-AUC 0.927, and macro average precision 0.850. Runtime was 339 seconds on CPU. This is one seed and one fold with no independent replication, so it is preliminary and not directly comparable with Cheng et al.'s reported results.
- **Initial transfer result (preliminary, separate fold):** ResNet18 initialized with torchvision's default ImageNet weights (`ResNet18_Weights.DEFAULT`), seed 42, official fold 4 held out for test, fold 5 for validation, and folds 1–3 for training. Validation macro-F1 selected the checkpoint; early stopping ended after 8 epochs, with best epoch 3. On the 628-image test fold: accuracy 0.946, macro-F1 0.929, balanced accuracy 0.921, macro ROC-AUC 0.994, and macro average precision 0.983. Runtime was 71.6 seconds on CPU. The held-out fold differs from the scratch run, so these scores are not a direct model comparison; both are single-seed preliminary results without uncertainty estimates.
- **Repeated matched comparison (fixed-protocol pilot):** Scratch ResNet18, ImageNet transfer ResNet18, and the hybrid model used seeds 42, 43, and 44 on all five official test folds. Each row reports test mean ± sample SD across seeds. For each run, test fold *F* was held out, fold *F+1* (wrapping 5→1) was validation, and the other three folds were training. Classical runs used CUDA on the RTX 4090; the hybrid model used CPU `lightning.qubit`.

  | Test fold (validation fold; test images) | Model | Macro-F1 | Accuracy |
  | --- | --- | ---: | ---: |
  | 1 (2; 542) | ResNet18 scratch | 0.816 ± 0.013 | 0.832 ± 0.013 |
  | 1 (2; 542) | ResNet18 ImageNet transfer | 0.933 ± 0.012 | 0.942 ± 0.011 |
  | 1 (2; 542) | Hybrid, 4 qubits, depth 2 | 0.490 ± 0.003 | 0.662 ± 0.003 |
  | 2 (3; 679) | ResNet18 scratch | 0.805 ± 0.019 | 0.819 ± 0.015 |
  | 2 (3; 679) | ResNet18 ImageNet transfer | 0.908 ± 0.019 | 0.917 ± 0.018 |
  | 2 (3; 679) | Hybrid, 4 qubits, depth 2 | 0.472 ± 0.001 | 0.629 ± 0.001 |
  | 3 (4; 572) | ResNet18 scratch | 0.858 ± 0.018 | 0.862 ± 0.019 |
  | 3 (4; 572) | ResNet18 ImageNet transfer | 0.933 ± 0.006 | 0.938 ± 0.007 |
  | 3 (4; 572) | Hybrid, 4 qubits, depth 2 | 0.497 ± 0.072 | 0.615 ± 0.028 |
  | 4 (5; 628) | ResNet18 scratch | 0.850 ± 0.014 | 0.867 ± 0.018 |
  | 4 (5; 628) | ResNet18 ImageNet transfer | 0.934 ± 0.016 | 0.949 ± 0.014 |
  | 4 (5; 628) | Hybrid, 4 qubits, depth 2 | 0.494 ± 0.009 | 0.670 ± 0.006 |
  | 5 (1; 643) | ResNet18 scratch | 0.841 ± 0.012 | 0.856 ± 0.011 |
  | 5 (1; 643) | ResNet18 ImageNet transfer | 0.892 ± 0.011 | 0.906 ± 0.009 |
  | 5 (1; 643) | Hybrid, 4 qubits, depth 2 | 0.495 ± 0.039 | 0.620 ± 0.010 |

  **Validation-based family comparison:** Across all five validation folds and three seeds per fold (15 runs per family), mean validation macro-F1 ± sample SD was 0.835 ± 0.043 for scratch ResNet18 (range 0.761–0.896), 0.931 ± 0.027 for ImageNet transfer ResNet18 (0.879–0.966), and 0.492 ± 0.036 for the hybrid (0.456–0.578). ImageNet transfer had the highest mean validation macro-F1 on each of the five fold configurations, so it is the selected leading family for this fixed pilot. This is not proof of general superiority; no family or hyperparameter was selected using test scores.

  **Scratch architecture comparison (validation only):** EfficientNet-B0 and ResNet18 used seeds 42–44 on all five official test folds, with the next fold used for validation. Each cell is validation macro-F1 mean ± sample SD across seeds.

  | Test fold (validation fold) | ResNet18 scratch | EfficientNet-B0 scratch |
  | --- | ---: | ---: |
  | 1 (2) | 0.807 ± 0.012 | 0.820 ± 0.036 |
  | 2 (3) | 0.847 ± 0.029 | 0.870 ± 0.011 |
  | 3 (4) | 0.885 ± 0.010 | 0.876 ± 0.031 |
  | 4 (5) | 0.776 ± 0.016 | 0.779 ± 0.057 |
  | 5 (1) | 0.862 ± 0.011 | 0.858 ± 0.033 |

  Across all 15 runs, validation macro-F1 was 0.835 ± 0.043 for ResNet18 and 0.841 ± 0.049 for EfficientNet-B0. EfficientNet's fold-mean advantage was small (+0.006) and it had larger seed variation; each architecture had the higher validation mean on some folds. The five-fold patient bootstrap below found a paired EfficientNet-minus-ResNet out-of-fold macro-F1 difference of −0.006 (95% interval −0.024 to 0.012), a descriptive test result that was not used to select the architecture. The transfer ResNet18 remained the leading family by validation macro-F1 (0.931 ± 0.027 across all folds and seeds). More architecture and transfer-learning comparisons remain appropriate before a broad claim.

  **Transfer architecture pilot (validation only):** On test fold 1 / validation fold 2, EfficientNet-B0 ImageNet transfer with seeds 42–44 reached validation macro-F1 0.918 ± 0.006, compared with 0.921 ± 0.004 for matched ResNet18 ImageNet transfer runs. This three-seed, single-fold result did not improve on ResNet18, so EfficientNet transfer was not extended across the other folds. The held-out test scores were not used for this decision.

  Across the five folds, the unweighted mean of each fold's three-seed test macro-F1 mean was 0.834 for scratch ResNet18, 0.920 for ImageNet transfer ResNet18, and 0.490 for the hybrid. The sample SD across the five fold means was 0.023, 0.019, and 0.010, respectively. These describe fold spread, not confidence intervals; folds and patient groups are not independent replicates. The test summaries are descriptive and were not used for model selection. This fixed three-seed protocol is a research pilot, not a clinical result or directly comparable Cheng et al. benchmark.

  **Patient-cluster uncertainty:** `python -m mri_classifier.patient_bootstrap` re-evaluates the saved fixed-cohort checkpoints, pools out-of-fold predictions across the five fold-specific models for each seed, then averages macro-F1 across seeds. A class-stratified bootstrap resamples the 233 patient groups (89 glioma, 82 meningioma, 62 pituitary) 2,000 times, keeping all slices from each sampled patient together. The pooled macro-F1 and 95% percentile interval were 0.835 (0.805–0.863) for scratch ResNet18, 0.920 (0.892–0.944) for ImageNet transfer, 0.829 (0.801–0.856) for scratch EfficientNet-B0, and 0.492 (0.472–0.512) for the hybrid. Paired bootstrap differences versus transfer were −0.085 (−0.104 to −0.068) for scratch ResNet18, −0.090 (−0.109 to −0.074) for EfficientNet-B0, and −0.428 (−0.455 to −0.400) for hybrid. The paired EfficientNet-minus-ResNet18 scratch difference was −0.006 (−0.024 to 0.012). The analysis used held-out predictions for descriptive uncertainty only; it did not tune models. Intervals condition on the saved checkpoints and three seeds, and do not measure training-seed, source-population, or clinical uncertainty. The reproducible summary is written to ignored `artifacts/patient-cluster-bootstrap.json`; patient identifiers and per-image predictions are not written to that summary.

- **Throughput profile (smoke only):** ResNet18 scratch, seed 42, batch size 32, one epoch, official test fold 1, validation fold 2, same dataset and code revision. Three runs per setting are labeled `experiment_kind=smoke`; their metrics are not model-selection evidence. Wall time recorded by the training reports:

  | Device and settings | Median (range) |
  | --- | ---: |
  | CPU, 0 workers, FP32 | 41.5 s (39.3–43.8) |
  | RTX 4090, 0 workers, FP32 | 10.9 s (10.8–11.0) |
  | RTX 4090, 2 workers, FP32 | 7.2 s (6.4–7.6) |
  | RTX 4090, 0 workers, CUDA FP16 autocast | 10.6 s (9.3–11.1) |

  CUDA FP32 was about 3.8× faster than CPU FP32 by median duration. Two workers reduced median duration by about 34% compared with CUDA FP32 at zero workers in this profile. The FP16 and FP32 ranges overlap, so this profile shows no clear AMP speed benefit. Runs were repeated without a separate warm-up phase and are only one-epoch pipeline smoke measurements; timing may include startup effects. Repeat with warm-up and compare multi-epoch validation behavior before choosing defaults. Effective device, workers, pinned-memory state, and AMP activation are recorded in each report; all 12 successful run records are in the local SQLite index. The dashboard defaults to automatic device selection, zero workers, and AMP off.

  **Multi-epoch GPU timing check (smoke only):** ResNet18 scratch, seed 42, batch size 32, four epochs, same official fold 1 / validation fold 2 split. Epoch 1 was treated as warm-up; the following three epochs had median durations of 9.0 s with zero workers / FP32, 4.0 s with two workers / FP32, and 8.8 s with zero workers / FP16 autocast. These are single training runs per setting. The speed result supports further loader profiling but is insufficient to change defaults. Validation macro-F1 varied across the short four-epoch curves, so this does not establish numerical or model-quality stability for mixed precision. Test metrics were not inspected for this profile. Run IDs: `c3e39e51-989e-4067-856f-6bd06cfa60d4`, `2195dd5c-1390-43fc-b499-6dd74ca33315`, and `c4980666-68e8-42bc-acbb-cf5506b07440`.

- **Quantum simulator profile (smoke only):** Hybrid classifier with four qubits, depth two, seed 42, batch size 32, one epoch, official test fold 1 and validation fold 2. All runs used the CPU; three runs per backend:

  | PennyLane backend | Median duration (range) |
  | --- | ---: |
  | `default.qubit` | 7.3 s (6.9–7.5) |
  | `lightning.qubit` | 12.4 s (11.9–12.7) |

  This small-circuit CPU profile did not show a `lightning.qubit` speed advantage. `lightning.gpu` is unavailable in this environment, so no accelerated quantum simulator was measured. These one-epoch smoke timings do not establish general backend performance.

## Pretrained model-weight provenance

- Torchvision 0.29.1 resolves `EfficientNet_B0_Weights.DEFAULT` to `IMAGENET1K_V1`. Torchvision identifies these as ImageNet-1K weights ported from the original paper, and documents their checkpoint URL and inference transforms in its [EfficientNet-B0 model entry](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0). The weight implementation points to [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) as its source.
- Torchvision warns that pretrained weights can carry terms derived from their training dataset and says users must determine permitted use; see its [pretrained model license note](https://github.com/pytorch/vision/blob/main/README.md#pre-trained-model-license). ImageNet-derived weights are used here for research evaluation only; this record does not establish commercial-use permission.
- The transfer pilot uses `torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1`, from `https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth`, with torchvision 0.29.1+cu130. New transfer reports record the exact weight enum, checkpoint URL, and installed torchvision version. Earlier reports used the `DEFAULT` alias and record ImageNet provenance but predate this more detailed metadata.

## References

1. Cheng J, Huang W, Cao S, et al. *Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition*. PLOS ONE 10(10): e0140381 (2015). [Article](https://doi.org/10.1371/journal.pone.0140381).
2. Cheng J. *brain tumor dataset*. Figshare, version 5 (2017). [Dataset](https://doi.org/10.6084/m9.figshare.1512427.v5).
