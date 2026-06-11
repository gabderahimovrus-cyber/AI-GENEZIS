"""SQLite metrics database for training history."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_genesis.config import METRICS_DB_PATH


@dataclass(slots=True)
class TrainingMetric:
    run_id: str
    epoch: int
    step: int
    loss: float
    validation_loss: float | None = None
    perplexity: float | None = None
    learning_rate: float | None = None
    tokens_processed: int = 0
    elapsed_seconds: float = 0.0


class MetricsStore:
    """Stores epoch/step metrics for GUI charts and model comparison."""

    def __init__(self, db_path: Path = METRICS_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    validation_loss REAL,
                    perplexity REAL,
                    learning_rate REAL,
                    tokens_processed INTEGER,
                    elapsed_seconds REAL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record(self, metric: TrainingMetric) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO training_metrics(run_id, epoch, step, loss, validation_loss, perplexity, learning_rate, tokens_processed, elapsed_seconds, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (metric.run_id, metric.epoch, metric.step, metric.loss, metric.validation_loss, metric.perplexity, metric.learning_rate, metric.tokens_processed, metric.elapsed_seconds, datetime.now(timezone.utc).isoformat()),
            )

    def recent(self, limit: int = 100) -> list[TrainingMetric]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT run_id, epoch, step, loss, validation_loss, perplexity, learning_rate, tokens_processed, elapsed_seconds FROM training_metrics ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [TrainingMetric(*row) for row in reversed(rows)]
