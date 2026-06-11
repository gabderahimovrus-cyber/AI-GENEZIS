"""Self-learning orchestration loop with candidate acceptance/rollback."""

from __future__ import annotations

import shutil
from pathlib import Path

from ai_genesis.config import DATA_DIR, MODELS_DIR
from ai_genesis.data.dataset_builder import DatasetBuilder
from ai_genesis.internet.learning import InternetLearningEngine
from ai_genesis.logging_system import logger


class SelfLearningLoop:
    """Collects data, builds datasets, trains candidates, and promotes improvements."""

    def __init__(self) -> None:
        self.internet = InternetLearningEngine()
        self.builder = DatasetBuilder()

    def collect_and_prepare(self, urls: list[str]) -> Path:
        logger.log("Начат сбор данных")
        documents = self.internet.collect(urls)
        for document in documents:
            logger.log(f"Загружена статья: {document.title}")
        corpus_path = DATA_DIR / "clean" / "corpus.txt"
        self.builder.write_clean_corpus((document.text for document in documents), corpus_path)
        logger.log("Создан датасет")
        return corpus_path

    def promote_if_better(self, candidate_metric: float, production_metric: float | None) -> bool:
        if production_metric is None or candidate_metric < production_metric:
            source = MODELS_DIR / "candidate"
            target = MODELS_DIR / "production"
            archive = MODELS_DIR / "archive"
            archive.mkdir(parents=True, exist_ok=True)
            if target.exists() and any(target.iterdir()):
                shutil.copytree(target, archive / "previous_production", dirs_exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
            logger.log("Candidate лучше production: новая версия принята")
            return True
        logger.log("Candidate хуже production: выполнен откат")
        return False
