"""Dataset Registry storing metadata and statistics for every Genesis dataset."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_genesis.config import DATA_DIR, DB_PATH


@dataclass(slots=True)
class DatasetMetadata:
    name: str
    path: str
    created_at: str
    documents: int
    tokens: int
    size_bytes: int
    sources: int
    language: str = "unknown"
    quality_score: float | None = None
    extra_stats: dict[str, float | int | str] | None = None


class DatasetRegistry:
    """SQLite registry for datasets and metadata used by training and GUI."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    documents INTEGER NOT NULL,
                    tokens INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sources INTEGER NOT NULL,
                    language TEXT,
                    quality_score REAL,
                    extra_stats TEXT
                )
                """
            )

    def register(self, metadata: DatasetMetadata) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO datasets(name, path, created_at, documents, tokens, size_bytes, sources, language, quality_score, extra_stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.name,
                    metadata.path,
                    metadata.created_at,
                    metadata.documents,
                    metadata.tokens,
                    metadata.size_bytes,
                    metadata.sources,
                    metadata.language,
                    metadata.quality_score,
                    json.dumps(metadata.extra_stats or {}, ensure_ascii=False),
                ),
            )

    def register_from_documents(self, name: str, path: Path, documents: list[dict[str, object]], quality_score: float | None = None) -> DatasetMetadata:
        tokens = sum(int(doc.get("tokens", len(str(doc.get("text", "")).split()))) for doc in documents)
        sources = len({str(doc.get("source", doc.get("url", "unknown"))) for doc in documents})
        language = self._dominant_language(documents)
        metadata = DatasetMetadata(
            name=name,
            path=str(path),
            created_at=datetime.now(timezone.utc).isoformat(),
            documents=len(documents),
            tokens=tokens,
            size_bytes=path.stat().st_size if path.exists() else 0,
            sources=sources,
            language=language,
            quality_score=quality_score,
        )
        self.register(metadata)
        return metadata

    def list_datasets(self, limit: int = 50) -> list[DatasetMetadata]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT name, path, created_at, documents, tokens, size_bytes, sources, language, quality_score, extra_stats FROM datasets ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [DatasetMetadata(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], json.loads(row[9] or "{}")) for row in rows]

    def recent_documents(self, dataset_path: Path, limit: int = 10) -> list[str]:
        if not dataset_path.exists():
            return []
        lines = dataset_path.read_text(encoding="utf-8").splitlines()[-limit:]
        docs: list[str] = []
        for line in lines:
            if line.strip().startswith("{"):
                docs.append(str(json.loads(line).get("text", line))[:500])
            else:
                docs.append(line[:500])
        return docs

    def _dominant_language(self, documents: list[dict[str, object]]) -> str:
        counts: dict[str, int] = {}
        for doc in documents:
            lang = str(doc.get("language", "unknown"))
            counts[lang] = counts.get(lang, 0) + 1
        return max(counts, key=counts.get) if counts else "unknown"


class DatasetQualityAnalyzer:
    """Pre-training quality gate for duplicates, language, length, noise, and garbage text."""

    def analyze(self, documents: list[str], expected_language: str = "any") -> dict[str, float | int | str | bool]:
        total = max(len(documents), 1)
        normalized = [" ".join(doc.lower().split()) for doc in documents]
        duplicate_ratio = 1.0 - (len(set(normalized)) / total)
        lengths = [len(doc.split()) for doc in documents]
        too_short = sum(1 for length in lengths if length < 20) / total
        too_long = sum(1 for length in lengths if length > 5000) / total
        garbage_ratio = sum(self._garbage_score(doc) for doc in documents) / total
        language_ratio = sum(1 for doc in documents if expected_language == "any" or self.detect_language(doc) == expected_language) / total
        score = max(0.0, min(1.0, 1.0 - duplicate_ratio * 0.35 - too_short * 0.2 - too_long * 0.1 - garbage_ratio * 0.25 - (1 - language_ratio) * 0.1))
        return {
            "documents": len(documents),
            "duplicate_ratio": round(duplicate_ratio, 3),
            "too_short_ratio": round(too_short, 3),
            "too_long_ratio": round(too_long, 3),
            "garbage_ratio": round(garbage_ratio, 3),
            "language_match_ratio": round(language_ratio, 3),
            "quality_score": round(score, 3),
            "language": expected_language,
            "passed": score >= 0.65,
        }

    def detect_language(self, text: str) -> str:
        cyrillic = sum(1 for char in text if "а" <= char.lower() <= "я")
        latin = sum(1 for char in text if "a" <= char.lower() <= "z")
        if cyrillic > latin:
            return "ru"
        if latin:
            return "en"
        return "unknown"

    def _garbage_score(self, text: str) -> float:
        if not text:
            return 1.0
        printable = sum(1 for char in text if char.isprintable()) / len(text)
        alpha = sum(1 for char in text if char.isalpha()) / len(text)
        repeated = max((text.count(ch) / len(text) for ch in set(text[:1000])), default=0.0)
        return max(0.0, min(1.0, (1 - printable) + max(0.0, 0.35 - alpha) + max(0.0, repeated - 0.25)))
