"""Single-job background training manager with persisted progress and cancellation."""

import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, datetime
from pathlib import Path

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
        self._recover_abandoned_jobs()

    def _recover_abandoned_jobs(self):
        """Reconcile completed reports and mark jobs left active by a process exit."""
        for row in self.store.list():
            if row["status"] not in {"queued", "running"}:
                continue
            if self._job_owner_may_be_alive(row):
                continue
            report_paths = sorted((Path("artifacts") / row["id"]).glob("*-report.json"))
            recovered = None
            for report_path in report_paths:
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                checkpoint = Path(report.get("checkpoint", ""))
                if report.get("status") == "completed" and checkpoint.is_file():
                    recovered = report
                    break
            if recovered:
                self.store.finish(
                    row["id"],
                    {
                        "macro_f1": recovered["validation_macro_f1"],
                        "test_metrics": recovered["test_metrics"],
                        "report": recovered,
                        "recovered_after_restart": True,
                    },
                )
            else:
                self.store.finish(
                    row["id"],
                    {},
                    error=(
                        "Training process stopped before writing a complete report. "
                        "A saved best checkpoint may be available for warm-start recovery."
                    ),
                )

    @staticmethod
    def _job_owner_may_be_alive(row: dict) -> bool:
        worker_pid = row.get("worker_pid")
        if worker_pid is not None:
            try:
                os.kill(int(worker_pid), 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
        created_at = datetime.fromisoformat(row["created_at"])
        return (datetime.now(UTC) - created_at).total_seconds() < 30

    def _resume_checkpoint(self, experiment_id: str) -> Path | None:
        root = (Path("artifacts") / experiment_id).resolve()
        if not root.is_dir():
            return None
        candidates = sorted(root.glob("*-best.pt"), key=lambda path: path.stat().st_mtime)
        return candidates[-1] if candidates else None

    def can_resume(self, experiment_id: str) -> bool:
        row = next((r for r in self.store.list() if r["id"] == experiment_id), None)
        if not row or row["status"] != "failed":
            return False
        checkpoint = self._resume_checkpoint(experiment_id)
        if checkpoint is None:
            return False
        try:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - invalid or incompatible checkpoint is not recoverable
            return False
        return all(
            key in state
            for key in ("model", "labels", "config", "best_epoch", "best_score", "history")
        )

    def submit(self, config: ExperimentConfig) -> str:
        with self._lock:
            if self._future and not self._future.done():
                raise RuntimeError("a training job is already active")
            self._cancel_event.clear()
            experiment_id = self.store.create(config.to_dict(), status="queued")
            self._experiment_id = experiment_id
            self._future = self._executor.submit(self._run, experiment_id, config)
            return experiment_id

    def _run(self, experiment_id: str, config: ExperimentConfig, resume_from: Path | None = None):
        self.store.set_status(experiment_id, "running", worker_pid=os.getpid())
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

            run_args = {
                "progress": progress,
                "cancel": self._cancel_event.is_set,
            }
            if resume_from is not None:
                run_args["resume_from"] = resume_from
            _, report = run_experiment(config, rows, f"artifacts/{experiment_id}", **run_args)
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

    def resume(self, experiment_id: str) -> str:
        with self._lock:
            if self._future and not self._future.done():
                raise RuntimeError("a training job is already active")
            row = next((r for r in self.store.list() if r["id"] == experiment_id), None)
            if not row or row["status"] != "failed":
                raise ValueError("only failed experiments can be resumed")
            if not self.can_resume(experiment_id):
                raise ValueError("no compatible best checkpoint is available to resume")
            checkpoint = self._resume_checkpoint(experiment_id)
            if checkpoint is None:
                raise ValueError("no saved best checkpoint is available to resume")
            if not checkpoint.resolve().is_relative_to(
                (Path("artifacts") / experiment_id).resolve()
            ):
                raise ValueError("resume checkpoint is outside the experiment artifact directory")
            config = ExperimentConfig(**row["config"])
            self._cancel_event.clear()
            resumed_id = self.store.create(
                config.to_dict(),
                status="queued",
                parent_id=experiment_id,
                resume_checkpoint=str(checkpoint),
            )
            self._experiment_id = resumed_id
            self._future = self._executor.submit(self._run, resumed_id, config, checkpoint)
            return resumed_id

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
