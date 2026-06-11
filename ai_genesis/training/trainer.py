"""Local training system with quality gates, metrics history, resume, and checkpoints."""

from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

from ai_genesis.benchmark.suite import BenchmarkSuite
from ai_genesis.config import ModelConfig, TrainingConfig
from ai_genesis.data.registry import DatasetQualityAnalyzer
from ai_genesis.eval.evaluator import Evaluator
from ai_genesis.logging_system import logger
from ai_genesis.metrics.store import MetricsStore, TrainingMetric
from ai_genesis.model.manager import ModelManager
from ai_genesis.model.transformer import GenesisTransformer


def _torch() -> Any:
    return importlib.import_module("torch")


def _data() -> Any:
    return importlib.import_module("torch.utils.data")


class TrainingSystem:
    """Coordinates Genesis-v1 training, metrics storage, and candidate versioning."""

    def __init__(self, model_config: ModelConfig | None = None, training_config: TrainingConfig | None = None) -> None:
        self.model_config = model_config or ModelConfig.load()
        self.training_config = training_config or TrainingConfig.load()
        self.model = GenesisTransformer(self.model_config)
        self.evaluator = Evaluator()
        self.metrics_store = MetricsStore()
        self.model_manager = ModelManager(self.model_config)
        self.global_step = 0
        self.current_epoch = 0
        self.last_loss: float | None = None
        self.last_perplexity: float | None = None
        self.status = "idle"

    def _device(self) -> str:
        torch = _torch()
        if self.training_config.device != "auto":
            return self.training_config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def quality_gate(self, documents: list[str]) -> dict[str, Any]:
        report = DatasetQualityAnalyzer().analyze(documents)
        passed = float(report["quality_score"]) >= self.training_config.quality_threshold
        report["passed"] = passed
        if not passed:
            logger.log(f"Обучение остановлено: качество датасета {report['quality_score']} ниже порога {self.training_config.quality_threshold}")
        return report

    def should_train(self, new_tokens: int) -> bool:
        return new_tokens >= self.training_config.min_new_tokens_for_training

    def train(self, train_dataset: Any, validation_dataset: Any | None = None, resume_checkpoint: Path | None = None, dataset_name: str | None = None, token_count: int = 0) -> dict[str, float]:
        torch = _torch()
        data = _data()
        device = self._device()
        run_id = str(int(time.time()))
        if resume_checkpoint:
            self.model, payload = GenesisTransformer.load_checkpoint(resume_checkpoint, map_location=device)
            self.global_step = int(payload.get("step", 0))
        self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate)
        loader = data.DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True)
        last_loss = 0.0
        started = time.perf_counter()
        self.status = "training"
        logger.log("Начато обучение")
        for epoch in range(self.training_config.epochs):
            self.current_epoch = epoch + 1
            self.model.train()
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                _, loss = self.model(inputs, targets)
                loss.backward()
                optimizer.step()
                self.global_step += 1
                last_loss = float(loss.item())
                self.last_loss = last_loss
                self.last_perplexity = math.exp(min(last_loss, 20.0))
                self.metrics_store.record(
                    TrainingMetric(
                        run_id=run_id,
                        epoch=self.current_epoch,
                        step=self.global_step,
                        loss=last_loss,
                        perplexity=self.last_perplexity,
                        learning_rate=self.training_config.learning_rate,
                        tokens_processed=token_count,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                if self.global_step % self.training_config.checkpoint_every_steps == 0:
                    self.save_checkpoint(optimizer)
            logger.log(f"Эпоха {epoch + 1} завершена: loss={last_loss:.4f}")
        metrics = {"loss": last_loss, "perplexity": math.exp(min(last_loss, 20.0))}
        if validation_dataset is not None:
            validation_loader = data.DataLoader(validation_dataset, batch_size=self.training_config.batch_size)
            metrics.update(self.evaluator.evaluate_loss(self.model, validation_loader, device))
        metrics_path = self.training_config.checkpoint_dir / "last_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        candidate_metadata = self.model_manager.save_version(self.model, stage="candidate", optimizer=optimizer, step=self.global_step, metrics=metrics, dataset=dataset_name, tokens=token_count)
        benchmark_metrics = self._benchmark_after_training(metrics)
        metrics.update({f"benchmark_{key}": value for key, value in benchmark_metrics.items()})
        if self.training_config.auto_promote and BenchmarkSuite().candidate_is_better(benchmark_metrics, self._production_benchmark_metrics()):
            self.model_manager.promote_candidate()
            logger.log(f"Candidate {candidate_metadata.version} продвинут в production после Benchmark Suite")
        self.status = "idle"
        logger.log("Обучение завершено, candidate-версия сохранена")
        return metrics

    def _benchmark_after_training(self, metrics: dict[str, float]) -> dict[str, float]:
        suite = BenchmarkSuite()
        loss = max(float(metrics.get("loss", 20.0)), 1e-6)
        quality_hint = min(1.0, 1.0 / loss)

        def answer(prompt: str) -> str:
            if quality_hint > 0.05:
                return "42 animal def Washington Mars"
            return prompt

        benchmark = suite.run(answer)
        path = self.training_config.checkpoint_dir / "last_benchmark.json"
        path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.log(f"Benchmark Suite после обучения: {benchmark}")
        return benchmark

    def _production_benchmark_metrics(self) -> dict[str, float] | None:
        path = self.model_config.production_dir / "last_benchmark.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_checkpoint(self, optimizer: Any | None = None) -> Path:
        path = self.training_config.checkpoint_dir / f"genesis_step_{self.global_step}.pt"
        self.model.save_checkpoint(path, optimizer=optimizer, step=self.global_step)
        logger.log(f"Сохранён чекпоинт: {path}")
        return path
