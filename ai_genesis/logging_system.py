"""Thread-safe application logging for GUI and background engines."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterable

from .config import LOG_DIR


class GenesisLogger:
    """Stores recent log messages in memory and mirrors them to disk."""

    def __init__(self, log_file: Path | None = None, max_entries: int = 1_000) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file or LOG_DIR / "genesis.log"
        self.entries: deque[str] = deque(maxlen=max_entries)
        self._lock = Lock()
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    def log(self, message: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        with self._lock:
            self.entries.append(entry)
        logging.info(message)
        return entry

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
        self.log_file.write_text("", encoding="utf-8")

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.entries)

    def extend(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.log(message)


logger = GenesisLogger()
