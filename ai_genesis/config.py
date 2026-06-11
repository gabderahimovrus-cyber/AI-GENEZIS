"""Central configuration for AI Genesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
DB_PATH = ROOT_DIR / "genesis_memory.sqlite3"


@dataclass(slots=True)
class ModelConfig:
    """Decoder-only Transformer configuration for Genesis-v1 MVP."""

    vocab_size: int = 32_000
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    context_length: int = 512
    dropout: float = 0.1
    model_name: str = "Genesis-v1"


@dataclass(slots=True)
class TrainingConfig:
    """Local training defaults with checkpointing and resume support."""

    batch_size: int = 8
    learning_rate: float = 3e-4
    epochs: int = 1
    checkpoint_every_steps: int = 500
    device: str = "auto"
    checkpoint_dir: Path = MODELS_DIR / "candidate"


@dataclass(slots=True)
class InternetConfig:
    """Crawler safety settings."""

    user_agent: str = "AI-Genesis-LearningBot/0.1 (+local educational project)"
    request_timeout: int = 15
    min_delay_seconds: float = 2.0
    max_pages_per_run: int = 20
