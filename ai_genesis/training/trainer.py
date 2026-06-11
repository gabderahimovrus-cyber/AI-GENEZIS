"""Local training system with scratch training, resume, and checkpoints."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from ai_genesis.config import ModelConfig, TrainingConfig
from ai_genesis.eval.evaluator import Evaluator
from ai_genesis.logging_system import logger
from ai_genesis.model.transformer import GenesisTransformer


def _torch() -> Any:
    return importlib.import_module("torch")


def _data() -> Any:
    return importlib.import_module("torch.utils.data")


class TrainingSystem:
    """Coordinates Genesis-v1 training and evaluation."""

    def __init__(self, model_config: ModelConfig | None = None, training_config: TrainingConfig | None = None) -> None:
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.model = GenesisTransformer(self.model_config)
        self.evaluator = Evaluator()
        self.global_step = 0

    def _device(self) -> str:
        torch = _torch()
        if self.training_config.device != "auto":
            return self.training_config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, train_dataset: Any, validation_dataset: Any | None = None, resume_checkpoint: Path | None = None) -> dict[str, float]:
        torch = _torch()
        data = _data()
        device = self._device()
        if resume_checkpoint:
            self.model, payload = GenesisTransformer.load_checkpoint(resume_checkpoint, map_location=device)
            self.global_step = int(payload.get("step", 0))
        self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate)
        loader = data.DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True)
        last_loss = 0.0
        logger.log("Начато обучение")
        for epoch in range(self.training_config.epochs):
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
                if self.global_step % self.training_config.checkpoint_every_steps == 0:
                    self.save_checkpoint(optimizer)
            logger.log(f"Эпоха {epoch + 1} завершена: loss={last_loss:.4f}")
        metrics = {"loss": last_loss}
        if validation_dataset is not None:
            validation_loader = data.DataLoader(validation_dataset, batch_size=self.training_config.batch_size)
            metrics.update(self.evaluator.evaluate_loss(self.model, validation_loader, device))
        metrics_path = self.training_config.checkpoint_dir / "last_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    def save_checkpoint(self, optimizer: Any | None = None) -> Path:
        path = self.training_config.checkpoint_dir / f"genesis_step_{self.global_step}.pt"
        self.model.save_checkpoint(path, optimizer=optimizer, step=self.global_step)
        logger.log(f"Сохранён чекпоинт: {path}")
        return path
