"""Single-job background training manager with persisted progress and cancellation."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from .config import ExperimentConfig
from .datasets import load_manifest
from .store import ExperimentStore
from .training import run_experiment


class TrainingManager:
    def __init__(self, store: ExperimentStore | None = None):
        self.store = store or ExperimentStore()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mri-training")
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._future: Future | None = None
        self._experiment_id: str | None = None

    def submit(self, config: ExperimentConfig) -> str:
        with self._lock:
            if self._future and not self._future.done():
                raise RuntimeError("a training job is already active")
            self._cancel_event.clear()
            experiment_id = self.store.create(config.to_dict(), status="queued")
            self._experiment_id = experiment_id
            self._future = self._executor.submit(self._run, experiment_id, config)
            return experiment_id

    def _run(self, experiment_id: str, config: ExperimentConfig):
        self.store.set_status(experiment_id, "running")
        try:
            rows = load_manifest(config.manifest_path, config.dataset_path)

            def progress(epoch, total, record):
                self.store.update_progress(
                    experiment_id,
                    {
                        "epoch": epoch,
                        "epochs": total,
                        "train_loss": record["train_loss"],
                        "validation_loss": record["validation_loss"],
                        "validation": record["validation"],
                    },
                )

            _, report = run_experiment(
                config,
                rows,
                f"artifacts/{experiment_id}",
                progress=progress,
                cancel=self._cancel_event.is_set,
            )
            self.store.finish(
                experiment_id,
                {
                    "macro_f1": report["validation_macro_f1"],
                    "test_metrics": report["test_metrics"],
                    "report": report,
                },
            )
        except InterruptedError as exc:
            self.store.set_status(experiment_id, "cancelled")
            self.store.finish(experiment_id, {}, error=str(exc))
            self.store.set_status(experiment_id, "cancelled")
        except Exception as exc:  # noqa: BLE001 - background failures must be persisted
            self.store.finish(experiment_id, {}, error=str(exc))

    def cancel(self) -> bool:
        with self._lock:
            if not self._future or self._future.done():
                return False
            self._cancel_event.set()
            return True

    def active_experiment_id(self) -> str | None:
        with self._lock:
            if self._future and not self._future.done():
                return self._experiment_id
            return None

    def wait(self, timeout: float | None = None):
        with self._lock:
            future = self._future
        if future is None:
            return None
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            return None

    def shutdown(self):
        self.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
