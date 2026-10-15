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
- **Initial measured baseline (preliminary, not a final benchmark):** ResNet18 trained from scratch, seed 42, official fold 5 held out for test, fold 1 for validation, and the remaining folds for training. Checkpoint selection used validation macro-F1; early stopping ended after 8 epochs, with best epoch 3. On the 643-image test fold: accuracy 0.829, macro-F1 0.822, balanced accuracy 0.831, macro ROC-AUC 0.927, and macro average precision 0.850. Runtime was 339 seconds on CPU. This is one seed and one fold with no independent replication, so it is preliminary and not directly comparable with Cheng et al.'s reported results.
- **Initial transfer result (preliminary, separate fold):** ResNet18 initialized with torchvision's default ImageNet weights (`ResNet18_Weights.DEFAULT`), seed 42, official fold 4 held out for test, fold 5 for validation, and folds 1–3 for training. Validation macro-F1 selected the checkpoint; early stopping ended after 8 epochs, with best epoch 3. On the 628-image test fold: accuracy 0.946, macro-F1 0.929, balanced accuracy 0.921, macro ROC-AUC 0.994, and macro average precision 0.983. Runtime was 71.6 seconds on CPU. The held-out fold differs from the scratch run, so these scores are not a direct model comparison; both are single-seed preliminary results without uncertainty estimates.
- **Repeated matched comparison (preliminary):** Scratch ResNet18, ImageNet transfer ResNet18, and the hybrid model used seeds 42, 43, and 44 on two official test folds. Each row reports test mean ± sample SD across those three seeds.

  | Test fold (validation fold; test images) | Model | Macro-F1 | Accuracy |
  | --- | --- | ---: | ---: |
  | 2 (3; 679) | ResNet18 scratch | 0.805 ± 0.019 | 0.819 ± 0.015 |
  | 2 (3; 679) | ResNet18 ImageNet transfer | 0.908 ± 0.019 | 0.917 ± 0.018 |
  | 2 (3; 679) | Hybrid, 4 qubits, depth 2 | 0.472 ± 0.001 | 0.629 ± 0.001 |
  | 3 (4; 572) | ResNet18 scratch | 0.858 ± 0.018 | 0.862 ± 0.019 |
  | 3 (4; 572) | ResNet18 ImageNet transfer | 0.933 ± 0.006 | 0.938 ± 0.007 |
  | 3 (4; 572) | Hybrid, 4 qubits, depth 2 | 0.497 ± 0.072 | 0.615 ± 0.028 |

  The remaining folds formed each training split. Classical runs used CUDA on the RTX 4090; the hybrid model used CPU `lightning.qubit`. These are seed-variation estimates within two fixed official folds, not a final five-fold estimate or a final benchmark. The hybrid scores were lower on both folds; broader fold coverage and validation-only diagnostics remain necessary.

## References

1. Cheng J, Huang W, Cao S, et al. *Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition*. PLOS ONE 10(10): e0140381 (2015). [Article](https://doi.org/10.1371/journal.pone.0140381).
2. Cheng J. *brain tumor dataset*. Figshare, version 5 (2017). [Dataset](https://doi.org/10.6084/m9.figshare.1512427.v5).
