"""Corpus Manager for raw texts, sources, versions, statistics, and change history."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_genesis.config import DB_PATH


@dataclass(slots=True)
class CorpusDocument:
    source: str
    text: str
    language: str = "unknown"
    tokens: int = 0
    received_at: str = ""


class CorpusManager:
    """Stores source documents with metadata and immutable corpus versions."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.initialize()

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS corpus_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT,
                    tokens INTEGER,
                    received_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS corpus_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS corpus_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def add_document(self, document: CorpusDocument) -> int:
        received_at = document.received_at or datetime.now(timezone.utc).isoformat()
        tokens = document.tokens or len(document.text.split())
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO corpus_documents(source, text, language, tokens, received_at) VALUES (?, ?, ?, ?, ?)",
                (document.source, document.text, document.language, tokens, received_at),
            )
            doc_id = int(cursor.lastrowid)
            connection.execute("INSERT INTO corpus_history(event, details, created_at) VALUES (?, ?, ?)", ("document_added", document.source, received_at))
            return doc_id

    def create_version(self, name: str, notes: str = "") -> None:
        with sqlite3.connect(self.db_path) as connection:
            count, tokens = connection.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM corpus_documents").fetchone()
            connection.execute(
                "INSERT OR REPLACE INTO corpus_versions(name, created_at, document_count, token_count, notes) VALUES (?, ?, ?, ?, ?)",
                (name, datetime.now(timezone.utc).isoformat(), count, tokens, notes),
            )

    def stats(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as connection:
            docs, tokens, sources = connection.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0), COUNT(DISTINCT source) FROM corpus_documents").fetchone()
        return {"documents": docs, "tokens": tokens, "sources": sources}
