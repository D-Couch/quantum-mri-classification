"""SQLite-backed experiment index."""

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path


class ExperimentStore:
    def __init__(self, path: str | Path = "artifacts/experiments.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
                config TEXT NOT NULL, metrics TEXT NOT NULL, error TEXT,
                progress TEXT NOT NULL DEFAULT '{}'
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(experiments)")}
            if "progress" not in columns:
                db.execute("ALTER TABLE experiments ADD COLUMN progress TEXT NOT NULL DEFAULT '{}'")

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def create(self, config: dict, status: str = "queued") -> str:
        if status not in {"queued", "running", "completed", "failed", "cancelled"}:
            raise ValueError("invalid experiment status")
        experiment_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                "INSERT INTO experiments (id,created_at,status,config,metrics,error,progress) VALUES (?,?,?,?,?,NULL,'{}')",
                (
                    experiment_id,
                    datetime.now(UTC).isoformat(),
                    status,
                    json.dumps(config, sort_keys=True),
                    "{}",
                ),
            )
        return experiment_id

    def finish(self, experiment_id: str, metrics: dict, error: str | None = None):
        status = "failed" if error else "completed"
        with self._connect() as db:
            result = db.execute(
                "UPDATE experiments SET status=?, metrics=?, error=? WHERE id=?",
                (status, json.dumps(metrics, sort_keys=True), error, experiment_id),
            )
            if not result.rowcount:
                raise KeyError(experiment_id)

    def set_status(self, experiment_id: str, status: str):
        if status not in {"queued", "running", "cancelled"}:
            raise ValueError("status must be queued, running, or cancelled")
        with self._connect() as db:
            result = db.execute(
                "UPDATE experiments SET status=? WHERE id=?", (status, experiment_id)
            )
            if not result.rowcount:
                raise KeyError(experiment_id)

    def update_progress(self, experiment_id: str, progress: dict):
        with self._connect() as db:
            result = db.execute(
                "UPDATE experiments SET progress=? WHERE id=?",
                (json.dumps(progress, sort_keys=True), experiment_id),
            )
            if not result.rowcount:
                raise KeyError(experiment_id)

    def list(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        return [
            {
                **dict(row),
                "config": json.loads(row["config"]),
                "metrics": json.loads(row["metrics"]),
                "progress": json.loads(row["progress"]),
            }
            for row in rows
        ]
