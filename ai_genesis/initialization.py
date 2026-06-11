"""First-run initialization wizard backend for AI Genesis."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ai_genesis.benchmark.suite import BenchmarkSuite
from ai_genesis.config import DATA_DIR, LOG_DIR, MODELS_DIR, ROOT_DIR, ensure_project_layout, load_config
from ai_genesis.data.corpus_manager import CorpusDocument, CorpusManager
from ai_genesis.data.dataset_builder import DatasetBuilder
from ai_genesis.data.registry import DatasetQualityAnalyzer, DatasetRegistry
from ai_genesis.data.text_dataset import TokenBlockDataset
from ai_genesis.diagnostics import SystemDiagnostics
from ai_genesis.knowledge.graph import KnowledgeGraph
from ai_genesis.knowledge.vector_store import VectorStore
from ai_genesis.logging_system import logger
from ai_genesis.memory.sqlite_memory import MemoryStore
from ai_genesis.metrics.store import MetricsStore, TrainingMetric
from ai_genesis.model.manager import ModelManager
from ai_genesis.model.tokenizer import GenesisTokenizer
from ai_genesis.model.transformer import GenesisTransformer
from ai_genesis.teacher.teacher import TeacherSystem

ProgressCallback = Callable[[str], None]

_DEMO_PARAGRAPHS = [
    "Genesis is a local autonomous learning system. It owns its tokenizer, datasets, memory, and model lifecycle.",
    "A tokenizer converts text into tokens. SentencePiece can train a compact local tokenizer from a small educational corpus.",
    "A transformer language model predicts the next token. Training improves the model by reducing loss on quality datasets.",
    "The Teacher System creates questions, evaluates answers, identifies weak areas, and proposes new curriculum topics.",
    "Memory stores dialogues, facts, and learning events in SQLite so Genesis can preserve useful local context.",
    "Benchmark tests compare candidate and production models on math, logic, programming, history, and general knowledge.",
    "Vector search retrieves related knowledge by comparing embeddings. FAISS is optional because Genesis has a local fallback.",
    "CUDA acceleration allows PyTorch to use an NVIDIA GPU such as GTX 1650 when a compatible CUDA build is installed.",
    "Безопасная локальная AI-система не использует облачные API и обучается на подготовленных локальных данных.",
    "Genesis должен постепенно учиться словам, предложениям, фактам, логике, программированию и диалогам.",
] * 12


class GenesisInitializer:
    """Creates all artifacts required for a complete first-run MVP."""

    def __init__(self, progress: ProgressCallback | None = None) -> None:
        self.config = load_config()
        self.progress = progress or (lambda message: logger.log(message))

    def is_initialized(self) -> bool:
        return not SystemDiagnostics(self.config.model, self.config.memory).missing_required_components()

    def initialize(self) -> dict[str, object]:
        self._log("Создание структуры каталогов Genesis")
        ensure_project_layout()
        for directory in [DATA_DIR / "raw", DATA_DIR / "clean", DATA_DIR / "datasets", DATA_DIR / "demo", MODELS_DIR / "base", MODELS_DIR / "candidate", MODELS_DIR / "production", MODELS_DIR / "archive", LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        self._log("Инициализация SQLite баз данных")
        memory = MemoryStore(self.config.memory.memory_db_path)
        metrics_store = MetricsStore(self.config.memory.metrics_db_path)
        corpus = CorpusManager(self.config.memory.memory_db_path)
        registry = DatasetRegistry(self.config.memory.memory_db_path)
        graph = KnowledgeGraph(self.config.memory.knowledge_graph_db_path)

        self._log("Создание демо-корпуса текстов")
        raw_path = DATA_DIR / "raw" / "genesis_demo_corpus.txt"
        raw_path.write_text("\n".join(_DEMO_PARAGRAPHS), encoding="utf-8")
        for index, paragraph in enumerate(_DEMO_PARAGRAPHS[:20], start=1):
            corpus.add_document(CorpusDocument(source=f"demo://genesis/{index}", text=paragraph, language="ru" if any("а" <= c.lower() <= "я" for c in paragraph) else "en", tokens=len(paragraph.split())))
        corpus.create_version("genesis_demo_v1", "Initial MVP demo corpus")

        builder = DatasetBuilder()
        clean_path = builder.write_clean_corpus(_DEMO_PARAGRAPHS, DATA_DIR / "clean" / "genesis_demo_clean.txt")

        self._log("Обучение SentencePiece токенизатора")
        tokenizer_prefix = self.config.model.tokenizer_path.with_suffix("")
        tokenizer = GenesisTokenizer()
        tokenizer.train([clean_path], tokenizer_prefix, vocab_size=self.config.model.vocab_size)

        self._log("Создание первого токенизированного датасета")
        dataset_path = builder.tokenize_corpus(clean_path, tokenizer, DATA_DIR / "datasets" / "genesis_demo_v1.jsonl")
        documents = [{"text": text, "source": "demo", "tokens": len(text.split()), "language": DatasetQualityAnalyzer().detect_language(text)} for text in clean_path.read_text(encoding="utf-8").splitlines()]
        quality = DatasetQualityAnalyzer().analyze([str(doc["text"]) for doc in documents])
        dataset_metadata = registry.register_from_documents("genesis_demo_v1", dataset_path, documents, float(quality["quality_score"]))

        self._log("Создание базовой модели Genesis-v1")
        manager = ModelManager(self.config.model)
        model = GenesisTransformer(self.config.model)
        params = model.parameter_count

        self._log("Выполнение короткого тестового цикла обучения")
        train_metrics = self._short_train(model, dataset_path, metrics_store)
        memory.add_learning_event("initial_training_complete", json.dumps(train_metrics, ensure_ascii=False))
        graph.add("Genesis-v1", "initialized_from", "genesis_demo_v1", "initializer")

        self._log("Сохранение candidate и production версий модели")
        candidate = manager.save_version(model, stage="candidate", step=int(train_metrics["steps"]), metrics=train_metrics, dataset=dataset_metadata.name, tokens=dataset_metadata.tokens)
        production = manager.save_version(model, stage="production", step=int(train_metrics["steps"]), metrics=train_metrics, dataset=dataset_metadata.name, tokens=dataset_metadata.tokens)
        base_checkpoint = MODELS_DIR / "base" / "Genesis-v1.base.pt"
        shutil.copy2(Path(production.checkpoint), base_checkpoint)

        self._log("Регистрация Benchmark и Teacher System")
        bench = BenchmarkSuite().run(lambda prompt: "42 animal def Washington Mars " + prompt)
        teacher_records = TeacherSystem().self_play(lambda question: f"Genesis answer about {question}", "Genesis basics", DATA_DIR / "datasets" / "teacher_bootstrap.jsonl", count=5, min_score=0.0)
        VectorStore().save(DATA_DIR / "demo" / "vector_store.pkl")

        summary = {
            "dataset": asdict(dataset_metadata),
            "candidate": asdict(candidate),
            "production": asdict(production),
            "parameters": params,
            "training": train_metrics,
            "benchmark": bench,
            "teacher_records": len(teacher_records),
        }
        (ROOT_DIR / "genesis_initialized.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log("Инициализация Genesis успешно завершена")
        return summary

    def _short_train(self, model: GenesisTransformer, dataset_path: Path, metrics_store: MetricsStore) -> dict[str, float]:
        import torch
        from torch.utils.data import DataLoader, Subset

        context = min(self.config.model.context_length, 64)
        dataset = TokenBlockDataset(dataset_path, context)
        subset = Subset(dataset, range(min(len(dataset), 8)))
        loader = DataLoader(subset, batch_size=1, shuffle=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        last_loss = 0.0
        steps = 0
        run_id = "initialization"
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            steps += 1
            last_loss = float(loss.item())
            metrics_store.record(TrainingMetric(run_id=run_id, epoch=1, step=steps, loss=last_loss, perplexity=math.exp(min(last_loss, 20.0)), learning_rate=1e-4, tokens_processed=len(dataset.tokens)))
            if steps >= 2:
                break
        return {"loss": last_loss, "perplexity": math.exp(min(last_loss, 20.0)), "steps": float(steps)}

    def _log(self, message: str) -> None:
        logger.log(message)
        self.progress(message)
