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
- **Import verification:** All four archive MD5s matched official Figshare API metadata. Import yielded 3,064 readable images, 233 patient IDs, expected class counts, and zero patient/fold conflicts. This verifies import and fold metadata only; no model has been trained on this dataset yet.

## References

1. Cheng J, Huang W, Cao S, et al. *Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition*. PLOS ONE 10(10): e0140381 (2015). [Article](https://doi.org/10.1371/journal.pone.0140381).
2. Cheng J. *brain tumor dataset*. Figshare, version 5 (2017). [Dataset](https://doi.org/10.6084/m9.figshare.1512427.v5).
