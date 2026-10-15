# Requirements

This document is the authoritative specification for the Quantum Brain MRI Classification Platform. It carries forward the project brief and organizes acceptance criteria for implementation.

## Mission and scientific principles

Build working research software for 2D multiclass brain tumor MRI classification (glioma, meningioma, pituitary tumor, no tumor where supported). Pursue the highest validated generalizable performance regardless of classical, quantum, hybrid, or ensemble architecture. Quantum simulation is a research capability, not a preferred answer. This is not clinical diagnostic software.

## Dataset registry and integrity

- Discover, import, inspect, validate, version, and load accessible datasets; support common image formats initially and extensible medical formats later.
- Record provenance, license, label definitions, integrity, class counts, and available subject identity. Do not bypass access or licensing conditions or commit datasets.
- Initial candidate registry: Cheng et al. Figshare CC BY 4.0 dataset (three tumor classes, 233 patients); preserve its class definition and do not imply no-tumor support. See `docs/DATASETS.md`.
- Preserve original labels. Use official splits where appropriate; otherwise split at patient/subject group, prevent duplicate/correlated/augmented samples crossing partitions, and visibly disclose unverifiable provenance.
- Fit statistics on training data only and augment training data only. Never use final test data for optimization.
- Extend toward 3D and other modalities without prematurely implementing them.

## Models and training

- Modular interfaces for classical, trainable hybrid quantum-classical, and ensemble models; select the best validated model irrespective of family.
- Classical training from scratch and transfer learning; document pretrained checkpoint provenance and configuration. Consider ResNet, EfficientNet, ViT, and medically relevant models based on evidence.
- At least one differentiable hybrid classifier; verify circuit influence, gradient flow, and parameter updates. Configure qubits, encoding, depth, ansatz, entanglement, measurements, and simulator.
- Reproducible configuration, seeds, train/validation loops, early stopping, scheduling, checkpoints, recovery, repeated runs, and appropriate cross-validation.
- Optimize on validation macro-F1 by default. Report accuracy, balanced accuracy, precision/recall/sensitivity/specificity, macro/weighted/per-class F1, ROC-AUC, PR-AUC, confusion matrix, MCC, kappa, latency, throughput, parameters, memory, and quantum costs when defined.
- Compare classical/hybrid controls, pursue evidence-supported optimization and ensembles, quantify uncertainty, and report protocol-qualified research references without fabricating results.

## Simulation and resources

- CPU quantum simulation and compatible accelerated backends; evaluate PennyLane Lightning CPU/GPU and alternatives empirically.
- Profile circuit complexity, execution, memory, and numerical stability. Benchmark batching and safe mixed precision; do not assume GPU simulation wins or use impractical circuits.
- Support CPU-only development; detect actual hardware, never assume the user's workstation is present.

## Dashboard

- Locally hosted, localhost by default, responsive dashboard with Home, Datasets, Experiment Configuration, Live Training, Results, History, Comparison, and Best Model views.
- User workflow: choose dataset, configure and run experiment, review metrics, save and compare. Support compatible image inference with the selected best overall model.
- Keep interface separate from model/data/engine logic. Use background jobs or subprocesses, progress and resource status, duplicate-job prevention, cancellation, and checkpoint recovery where feasible.

## Persistence and security

- Unique ID and durable record for every run, including dataset/version, architecture/mode, hyperparameters, quantum settings, seeds, software/hardware, Git revision, histories, metrics, checkpoint paths, visualizations, status, and errors.
- SQLite index, structured artifacts (e.g. JSON), safe writes, local exports, filterable history. Keep artifacts outside Git.
- Clearly distinguish genuine, synthetic, incomplete, failed, and unverified results. Protect credentials and private data.

## Quality and acceptance

- Typed configuration, clear errors/logging, resource cleanup, modular package, reproducible dependency setup.
- Unit, integration, end-to-end smoke, quantum-gradient, leakage, metric, persistence, and recovery checks; suitable GitHub Actions.
- Version 1 must install and launch; provide functional dashboard and one real dataset; classical scratch and transfer training; trainable hybrid classifier; dashboard-driven training/evaluation; research metrics and persistent history; model comparison and best selection; inference; reproducibility/tests; explicit hardware/data/experiment verification status.
- Milestones: Foundation; Dataset Pipeline; Classical Training; Quantum Integration; Model Selection/Research; Performance Optimization.
