"""SQLite Knowledge Graph storing Entity → Relation → Entity triples."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ai_genesis.config import MemoryConfig


class KnowledgeGraph:
    """Persists structured facts for later answer generation."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or MemoryConfig.load().knowledge_graph_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(subject, relation, object, source)
                )
                """
            )

    def add(self, subject: str, relation: str, object_: str, source: str | None = None, confidence: float = 1.0) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO triples(subject, relation, object, source, confidence) VALUES (?, ?, ?, ?, ?)",
                (subject, relation, object_, source, confidence),
            )

    def related(self, entity: str, limit: int = 20) -> list[tuple[str, str, str]]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT subject, relation, object FROM triples WHERE subject = ? OR object = ? ORDER BY confidence DESC LIMIT ?",
                (entity, entity, limit),
            ).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]
