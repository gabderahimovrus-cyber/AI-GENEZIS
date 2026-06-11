"""CLI entry point for AI Genesis."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_genesis.benchmark.suite import BenchmarkSuite
from ai_genesis.config import ensure_project_layout, load_config
from ai_genesis.data.corpus_manager import CorpusDocument, CorpusManager
from ai_genesis.data.registry import DatasetQualityAnalyzer, DatasetRegistry
from ai_genesis.gui.app import run_gui
from ai_genesis.internet.learning import InternetLearningEngine
from ai_genesis.knowledge.graph import KnowledgeGraph
from ai_genesis.logging_system import logger
from ai_genesis.memory.sqlite_memory import MemoryStore
from ai_genesis.metrics.store import MetricsStore
from ai_genesis.model.manager import ModelManager
from ai_genesis.training.trainer import TrainingSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Genesis local autonomous system")
    parser.add_argument("--init-db", action="store_true", help="Initialize SQLite memory, metrics, corpus, dataset, and graph databases")
    parser.add_argument("--gui", action="store_true", help="Start graphical interface")
    parser.add_argument("--train", type=Path, help="Run local training on a tokenized dataset JSONL path")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark scaffold against the current chat/model interface")
    parser.add_argument("--collect-data", nargs="*", default=None, help="Collect allowed educational URLs into the corpus")
    parser.add_argument("--export-model", nargs=2, metavar=("PTH", "ONNX"), help="Export production model to .pth and .onnx paths")
    args = parser.parse_args()

    ensure_project_layout()
    config = load_config()

    if args.init_db:
        MemoryStore(config.memory.memory_db_path).initialize()
        MetricsStore(config.memory.metrics_db_path).initialize()
        CorpusManager(config.memory.memory_db_path).initialize()
        DatasetRegistry(config.memory.memory_db_path).initialize()
        KnowledgeGraph(config.memory.knowledge_graph_db_path).initialize()
        print("AI Genesis databases initialized")
        return

    if args.collect_data is not None:
        engine = InternetLearningEngine(config.internet)
        corpus = CorpusManager(config.memory.memory_db_path)
        documents = engine.collect(args.collect_data)
        for document in documents:
            corpus.add_document(CorpusDocument(source=document.url, text=document.text, language="unknown", tokens=len(document.text.split())))
            logger.log(f"Документ добавлен в корпус: {document.title}")
        print(f"Collected {len(documents)} documents")
        return

    if args.train:
        texts = _dataset_texts(args.train)
        report = DatasetQualityAnalyzer().analyze(texts)
        if float(report["quality_score"]) < config.training.quality_threshold:
            print(f"Training blocked by quality gate: {report}")
            return
        from ai_genesis.data.text_dataset import TokenBlockDataset

        dataset = TokenBlockDataset(args.train, config.model.context_length)
        metrics = TrainingSystem(config.model, config.training).train(dataset, dataset_name=args.train.name, token_count=sum(len(text.split()) for text in texts))
        print(metrics)
        return

    if args.evaluate:
        suite = BenchmarkSuite()
        metrics = suite.run(lambda prompt: f"Local benchmark placeholder answer for: {prompt}")
        print(metrics)
        return

    if args.export_model:
        manager = ModelManager(config.model)
        model, _ = manager.load_production()
        if model is None:
            print("No production model found")
            return
        print(manager.export_model(model, Path(args.export_model[0]), Path(args.export_model[1])))
        return

    run_gui()


def _dataset_texts(path: Path) -> list[str]:
    import json

    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.strip().startswith("{"):
            texts.append(str(json.loads(line).get("text", line)))
        else:
            texts.append(line)
    return texts


if __name__ == "__main__":
    main()
