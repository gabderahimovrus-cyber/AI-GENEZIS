"""SQLite-backed episodic, semantic, and learning memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from ai_genesis.config import DB_PATH, MemoryConfig


class MemoryStore:
    """Persists chats, facts, and training history locally in separated memory tables."""

    def __init__(self, db_path: Path | None = None) -> None:
        config = MemoryConfig.load()
        self.db_path = db_path or config.memory_db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL UNIQUE,
                    source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS learning_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT NOT NULL,
                    metrics TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def add_dialogue(self, role: str, content: str) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO episodic_memory(role, content) VALUES (?, ?)", (role, content))

    def add_fact(self, fact: str, source: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO semantic_memory(fact, source) VALUES (?, ?)", (fact, source))

    def add_learning_event(self, event: str, metrics: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO learning_memory(event, metrics) VALUES (?, ?)", (event, metrics))

    def recent_dialogue(self, limit: int = 20) -> list[tuple[str, str]]:
        with self.connect() as connection:
            rows: Iterable[tuple[str, str]] = connection.execute("SELECT role, content FROM episodic_memory ORDER BY id DESC LIMIT ?", (limit,))
            return list(reversed(list(rows)))

    def recent_facts(self, limit: int = 20) -> list[str]:
        with self.connect() as connection:
            return [row[0] for row in connection.execute("SELECT fact FROM semantic_memory ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def recent_learning_events(self, limit: int = 20) -> list[tuple[str, str | None]]:
        with self.connect() as connection:
            return connection.execute("SELECT event, metrics FROM learning_memory ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
