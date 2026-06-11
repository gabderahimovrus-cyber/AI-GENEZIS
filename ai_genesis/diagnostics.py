"""System diagnostics for AI Genesis MVP readiness checks."""

from __future__ import annotations

import importlib.util
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ai_genesis.config import DATA_DIR, LOG_DIR, METRICS_DB_PATH, MODELS_DIR, ROOT_DIR, MemoryConfig, ModelConfig
from ai_genesis.knowledge.vector_store import VectorStore
from ai_genesis.logging_system import logger
from ai_genesis.system.monitor import SystemMonitor


@dataclass(slots=True)
class DiagnosticItem:
    name: str
    status: str
    message: str


class SystemDiagnostics:
    """Checks tokenizer, models, datasets, databases, dependencies, and subsystems."""

    def __init__(self, model_config: ModelConfig | None = None, memory_config: MemoryConfig | None = None) -> None:
        self.model_config = model_config or ModelConfig.load()
        self.memory_config = memory_config or MemoryConfig.load()

    def run(self) -> list[DiagnosticItem]:
        checks = [
            self._tokenizer(),
            self._dataset(),
            self._model("Production Model", self.model_config.production_dir),
            self._model("Candidate Model", self.model_config.candidate_dir),
            self._cuda(),
            self._dependency("PyTorch", "torch"),
            self._dependency("SentencePiece", "sentencepiece"),
            self._dependency("FAISS", "faiss", warning_when_missing=True),
            self._sqlite(),
            self._vector_store(),
            self._memory_system(),
            self._teacher_system(),
            self._internet_learning_engine(),
        ]
        return checks

    def missing_required_components(self) -> list[DiagnosticItem]:
        required = {"Tokenizer", "Dataset", "Production Model", "Candidate Model", "SQLite", "Memory System"}
        missing = [item for item in self.run() if item.name in required and item.status == "ERROR"]
        required_dirs = [DATA_DIR / "raw", DATA_DIR / "clean", DATA_DIR / "datasets", MODELS_DIR / "base", MODELS_DIR / "candidate", MODELS_DIR / "production", LOG_DIR]
        for directory in required_dirs:
            if not directory.exists():
                missing.append(DiagnosticItem("Directories", "ERROR", f"Отсутствует каталог {directory.relative_to(ROOT_DIR)}"))
        return missing

    def _tokenizer(self) -> DiagnosticItem:
        model = self.model_config.tokenizer_path
        vocab = model.with_suffix(".vocab")
        if model.exists() and vocab.exists() and model.stat().st_size > 0 and vocab.stat().st_size > 0:
            return DiagnosticItem("Tokenizer", "OK", f"Найден {model.name} и {vocab.name}")
        return DiagnosticItem("Tokenizer", "ERROR", "Отсутствуют tokenizer.model/tokenizer.vocab. Запустите инициализацию Genesis.")

    def _dataset(self) -> DiagnosticItem:
        datasets = sorted((DATA_DIR / "datasets").glob("*.jsonl"))
        datasets = [path for path in datasets if path.stat().st_size > 0]
        if datasets:
            return DiagnosticItem("Dataset", "OK", f"Найден датасет {datasets[-1].relative_to(ROOT_DIR)}")
        return DiagnosticItem("Dataset", "ERROR", "Нет подготовленного JSONL-датасета.")

    def _model(self, name: str, directory: Path) -> DiagnosticItem:
        checkpoints = sorted(directory.glob("*.pt")) if directory.exists() else []
        metadata = sorted(directory.glob("*.metadata.json")) if directory.exists() else []
        if checkpoints and metadata and checkpoints[-1].stat().st_size > 0:
            return DiagnosticItem(name, "OK", f"Найден checkpoint {checkpoints[-1].name}")
        return DiagnosticItem(name, "ERROR", f"В {directory.relative_to(ROOT_DIR)} нет зарегистрированной модели.")

    def _cuda(self) -> DiagnosticItem:
        if importlib.util.find_spec("torch") is None:
            return DiagnosticItem("CUDA", "WARNING", "PyTorch не установлен, CUDA не может быть проверена.")
        import torch

        if not torch.cuda.is_available():
            return DiagnosticItem(
                "CUDA",
                "WARNING",
                "CUDA недоступна. Для NVIDIA GTX 1650 установите PyTorch с CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121",
            )
        snap = SystemMonitor().snapshot()
        return DiagnosticItem("CUDA", "OK", f"{snap.gpu_name}, VRAM {snap.vram_total_mb or 0:.0f} MB")

    def _dependency(self, name: str, module: str, warning_when_missing: bool = False) -> DiagnosticItem:
        if importlib.util.find_spec(module) is None:
            status = "WARNING" if warning_when_missing else "ERROR"
            return DiagnosticItem(name, status, f"Модуль {module} не установлен.")
        return DiagnosticItem(name, "OK", f"Модуль {module} доступен.")

    def _sqlite(self) -> DiagnosticItem:
        try:
            for path in [self.memory_config.memory_db_path, self.memory_config.metrics_db_path, self.memory_config.knowledge_graph_db_path]:
                path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(path) as connection:
                    connection.execute("SELECT 1")
        except sqlite3.Error as exc:
            return DiagnosticItem("SQLite", "ERROR", f"Ошибка SQLite: {exc}")
        return DiagnosticItem("SQLite", "OK", "SQLite базы доступны.")

    def _vector_store(self) -> DiagnosticItem:
        try:
            store = VectorStore(dimension=8)
            store.add_texts(["genesis diagnostic vector store"])
            result = store.search("genesis", k=1)
            if result:
                backend = "FAISS" if store._faiss_available else "локальный fallback"
                return DiagnosticItem("Vector Store", "OK", f"Векторное хранилище работает ({backend}).")
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            return DiagnosticItem("Vector Store", "ERROR", f"Ошибка Vector Store: {exc}")
        return DiagnosticItem("Vector Store", "WARNING", "Vector Store пуст или не вернул результат.")

    def _memory_system(self) -> DiagnosticItem:
        try:
            from ai_genesis.memory.sqlite_memory import MemoryStore

            MemoryStore(self.memory_config.memory_db_path).initialize()
            return DiagnosticItem("Memory System", "OK", "Episodic/Semantic/Learning memory готовы.")
        except Exception as exc:  # noqa: BLE001
            return DiagnosticItem("Memory System", "ERROR", f"Ошибка Memory System: {exc}")

    def _teacher_system(self) -> DiagnosticItem:
        try:
            from ai_genesis.teacher.teacher import TeacherSystem

            tasks = TeacherSystem().create_questions("Genesis", count=1)
            return DiagnosticItem("Teacher System", "OK", f"Teacher сгенерировал {len(tasks)} задание.")
        except Exception as exc:  # noqa: BLE001
            return DiagnosticItem("Teacher System", "ERROR", f"Ошибка Teacher: {exc}")

    def _internet_learning_engine(self) -> DiagnosticItem:
        try:
            from ai_genesis.internet.learning import InternetLearningEngine

            engine = InternetLearningEngine()
            return DiagnosticItem("Internet Learning Engine", "OK", f"Whitelist доменов: {len(engine.config.allowed_domains)}")
        except Exception as exc:  # noqa: BLE001
            logger.log(f"Диагностика Internet Learning Engine: {exc}")
            return DiagnosticItem("Internet Learning Engine", "WARNING", f"Движок недоступен: {exc}")
