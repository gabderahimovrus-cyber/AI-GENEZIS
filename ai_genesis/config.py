"""YAML-backed configuration system for AI Genesis."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
DB_PATH = ROOT_DIR / "genesis_memory.sqlite3"
METRICS_DB_PATH = ROOT_DIR / "metrics.db"

T = TypeVar("T")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a small YAML file with PyYAML when available and a safe fallback parser otherwise."""
    if not path.exists():
        return {}
    import importlib.util

    if importlib.util.find_spec("yaml") is not None:
        yaml = __import__("yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    return _minimal_yaml(path.read_text(encoding="utf-8"))


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the flat/list YAML subset used by bundled configs when PyYAML is absent."""
    result: dict[str, Any] = {}
    current_list: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_list:
            result[current_list].append(_coerce(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            result[key] = []
            current_list = key
        else:
            result[key] = _coerce(value)
            current_list = None
    return result


def _coerce(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_dataclass(cls: type[T], filename: str) -> T:
    payload = _read_yaml(CONFIG_DIR / filename)
    fields = {field.name: field for field in dataclasses.fields(cls)}
    values: dict[str, Any] = {}
    for name, field_info in fields.items():
        if name not in payload:
            continue
        value = payload[name]
        if field_info.type is Path or isinstance(field_info.default, Path):
            value = _resolve_path(value)
        values[name] = value
    return cls(**values)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(slots=True)
class ModelConfig:
    """Decoder-only Transformer configuration loaded from ``configs/model.yaml``."""

    vocab_size: int = 32_000
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    context_length: int = 512
    dropout: float = 0.1
    model_name: str = "Genesis-v1"
    tokenizer_path: Path = MODELS_DIR / "base" / "tokenizer.model"
    production_dir: Path = MODELS_DIR / "production"
    candidate_dir: Path = MODELS_DIR / "candidate"
    archive_dir: Path = MODELS_DIR / "archive"
    online_enabled: bool = True
    online_provider: str = "openai_compatible"
    online_model: str = "gpt-4o-mini"
    online_base_url: str = "https://api.openai.com/v1"
    online_api_key_env: str = "OPENAI_API_KEY"
    online_timeout_seconds: int = 60

    @classmethod
    def load(cls) -> "ModelConfig":
        return _load_dataclass(cls, "model.yaml")


@dataclass(slots=True)
class TrainingConfig:
    """Local training defaults loaded from ``configs/training.yaml``."""

    batch_size: int = 8
    learning_rate: float = 3e-4
    epochs: int = 1
    checkpoint_every_steps: int = 500
    device: str = "auto"
    checkpoint_dir: Path = MODELS_DIR / "candidate"
    min_new_tokens_for_training: int = 50_000
    quality_threshold: float = 0.65
    auto_promote: bool = True

    @classmethod
    def load(cls) -> "TrainingConfig":
        return _load_dataclass(cls, "training.yaml")


@dataclass(slots=True)
class MemoryConfig:
    """SQLite memory configuration loaded from ``configs/memory.yaml``."""

    memory_db_path: Path = DB_PATH
    metrics_db_path: Path = METRICS_DB_PATH
    knowledge_graph_db_path: Path = ROOT_DIR / "knowledge_graph.db"
    episodic_limit: int = 200
    semantic_limit: int = 1_000
    learning_limit: int = 1_000

    @classmethod
    def load(cls) -> "MemoryConfig":
        return _load_dataclass(cls, "memory.yaml")


@dataclass(slots=True)
class InternetConfig:
    """Crawler safety settings loaded from ``configs/internet.yaml``."""

    user_agent: str = "AI-Genesis-LearningBot/0.1 (+local educational project)"
    request_timeout: int = 15
    min_delay_seconds: float = 2.0
    max_pages_per_run: int = 20
    allowed_domains: list[str] = field(default_factory=lambda: [
        "wikipedia.org",
        "wikibooks.org",
        "python.org",
        "pytorch.org",
        "khanacademy.org",
        "mit.edu",
        "stanford.edu",
        "arxiv.org",
    ])

    @classmethod
    def load(cls) -> "InternetConfig":
        return _load_dataclass(cls, "internet.yaml")


@dataclass(slots=True)
class GUIConfig:
    """GUI configuration loaded from ``configs/gui.yaml``."""

    title: str = "AI Genesis"
    width: int = 1280
    height: int = 820
    refresh_ms: int = 1000
    metrics_points: int = 100

    @classmethod
    def load(cls) -> "GUIConfig":
        return _load_dataclass(cls, "gui.yaml")


@dataclass(slots=True)
class GenesisConfig:
    """Full project configuration bundle."""

    model: ModelConfig = field(default_factory=ModelConfig.load)
    training: TrainingConfig = field(default_factory=TrainingConfig.load)
    memory: MemoryConfig = field(default_factory=MemoryConfig.load)
    internet: InternetConfig = field(default_factory=InternetConfig.load)
    gui: GUIConfig = field(default_factory=GUIConfig.load)


def load_config() -> GenesisConfig:
    """Load all YAML configuration groups."""
    return GenesisConfig()


def ensure_project_layout() -> None:
    """Create local-only directories used by data, models, logs, and configs."""
    for path in [CONFIG_DIR, DATA_DIR, LOG_DIR, MODELS_DIR, DATA_DIR / "raw", DATA_DIR / "clean", DATA_DIR / "datasets", DATA_DIR / "corpora"]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [MODELS_DIR / "base", MODELS_DIR / "candidate", MODELS_DIR / "production", MODELS_DIR / "archive"]:
        path.mkdir(parents=True, exist_ok=True)
