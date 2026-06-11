"""Model version lifecycle manager for production/candidate Genesis artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_genesis.config import MODELS_DIR, ModelConfig
from ai_genesis.logging_system import logger
from ai_genesis.model.transformer import GenesisTransformer


@dataclass(slots=True)
class ModelVersionMetadata:
    version: str
    created_at: str
    stage: str
    checkpoint: str
    loss: float | None = None
    perplexity: float | None = None
    dataset: str | None = None
    tokens: int = 0
    parameters: int | None = None
    size_bytes: int = 0


class ModelManager:
    """Loads, saves, promotes, rolls back, exports, and records model history."""

    def __init__(self, config: ModelConfig | None = None, root: Path = MODELS_DIR) -> None:
        self.config = config or ModelConfig.load()
        self.root = root
        self.base_dir = self.config.production_dir.parent if self.config.production_dir.parent.name == "models" else root
        self.production_dir = self.config.production_dir
        self.candidate_dir = self.config.candidate_dir
        self.archive_dir = self.config.archive_dir
        self.history_path = root / "history.jsonl"
        for path in [root / "base", self.candidate_dir, self.production_dir, self.archive_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def load_production(self, map_location: str = "cpu") -> tuple[GenesisTransformer | None, dict[str, Any]]:
        return self._load_latest(self.production_dir, map_location)

    def load_candidate(self, map_location: str = "cpu") -> tuple[GenesisTransformer | None, dict[str, Any]]:
        return self._load_latest(self.candidate_dir, map_location)

    def _load_latest(self, directory: Path, map_location: str) -> tuple[GenesisTransformer | None, dict[str, Any]]:
        checkpoints = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not checkpoints:
            return None, {}
        model, payload = GenesisTransformer.load_checkpoint(checkpoints[0], map_location=map_location)
        logger.log(f"Загружена модель: {checkpoints[0]}")
        return model, payload

    def save_version(
        self,
        model: GenesisTransformer,
        stage: str = "candidate",
        optimizer: Any | None = None,
        step: int = 0,
        metrics: dict[str, float] | None = None,
        dataset: str | None = None,
        tokens: int = 0,
    ) -> ModelVersionMetadata:
        directory = self.candidate_dir if stage == "candidate" else self.production_dir
        version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
        checkpoint = directory / f"{version}.pt"
        model.save_checkpoint(checkpoint, optimizer=optimizer, step=step)
        metadata = self._metadata(version, stage, checkpoint, model, metrics, dataset, tokens)
        self._write_metadata(directory / f"{version}.metadata.json", metadata)
        self._append_history("save", metadata)
        logger.log(f"Сохранена версия модели {version} в {stage}")
        return metadata

    def promote_candidate(self) -> bool:
        candidates = sorted(self.candidate_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            logger.log("Нет candidate-модели для продвижения")
            return False
        self._archive_production()
        for path in self.production_dir.glob("*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        for path in self.candidate_dir.glob("*"):
            target = self.production_dir / path.name
            shutil.copytree(path, target, dirs_exist_ok=True) if path.is_dir() else shutil.copy2(path, target)
        self._append_history("promote", {"candidate": candidates[0].name, "promoted_at": self._now()})
        logger.log("Candidate-модель продвинута в production")
        return True

    def rollback(self) -> bool:
        archives = sorted([p for p in self.archive_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if not archives:
            logger.log("Архив production пуст: откат невозможен")
            return False
        for path in self.production_dir.glob("*"):
            path.unlink() if path.is_file() else shutil.rmtree(path)
        for path in archives[0].glob("*"):
            target = self.production_dir / path.name
            shutil.copytree(path, target, dirs_exist_ok=True) if path.is_dir() else shutil.copy2(path, target)
        self._append_history("rollback", {"archive": archives[0].name, "rolled_back_at": self._now()})
        logger.log(f"Откат выполнен к {archives[0].name}")
        return True

    def export_model(self, model: GenesisTransformer, pth_path: Path, onnx_path: Path | None = None) -> dict[str, Path]:
        import importlib

        torch = importlib.import_module("torch")
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), pth_path)
        exported = {"pth": pth_path}
        if onnx_path is not None:
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            dummy = torch.zeros((1, min(8, model.config.context_length)), dtype=torch.long)
            torch.onnx.export(model, dummy, onnx_path, input_names=["input_ids"], output_names=["logits", "loss"], opset_version=17)
            exported["onnx"] = onnx_path
        logger.log(f"Экспорт модели завершён: {exported}")
        return exported

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        rows = [json.loads(line) for line in self.history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows[-limit:]

    def current_status(self) -> dict[str, Any]:
        metadata_files = sorted(self.production_dir.glob("*.metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8")) if metadata_files else {}
        metadata.setdefault("model_name", self.config.model_name)
        metadata.setdefault("status", "production не загружена" if not metadata_files else "production готова")
        return metadata

    def _archive_production(self) -> None:
        if not self.production_dir.exists() or not any(self.production_dir.iterdir()):
            return
        target = self.archive_dir / datetime.now(timezone.utc).strftime("production_%Y%m%d%H%M%S")
        shutil.copytree(self.production_dir, target, dirs_exist_ok=True)

    def _metadata(self, version: str, stage: str, checkpoint: Path, model: GenesisTransformer, metrics: dict[str, float] | None, dataset: str | None, tokens: int) -> ModelVersionMetadata:
        metrics = metrics or {}
        return ModelVersionMetadata(
            version=version,
            created_at=self._now(),
            stage=stage,
            checkpoint=str(checkpoint),
            loss=metrics.get("loss") or metrics.get("validation_loss"),
            perplexity=metrics.get("perplexity"),
            dataset=dataset,
            tokens=tokens,
            parameters=model.parameter_count,
            size_bytes=checkpoint.stat().st_size if checkpoint.exists() else 0,
        )

    def _write_metadata(self, path: Path, metadata: ModelVersionMetadata) -> None:
        path.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_history(self, event: str, payload: Any) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "payload": asdict(payload) if dataclasses_is_dataclass(payload) else payload, "created_at": self._now()}
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


def dataclasses_is_dataclass(value: Any) -> bool:
    import dataclasses

    return dataclasses.is_dataclass(value)
