"""SentencePiece BPE tokenizer wrapper owned by AI Genesis."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable


class GenesisTokenizer:
    """Train, save, load, encode, and decode a local SentencePiece tokenizer."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path
        self.processor = None
        if model_path is not None and model_path.exists():
            self.load(model_path)

    def train(
        self,
        input_files: Iterable[Path],
        model_prefix: Path,
        vocab_size: int = 32_000,
        character_coverage: float = 0.9995,
    ) -> Path:
        sentencepiece = importlib.import_module("sentencepiece")
        files = [str(path) for path in input_files]
        if not files:
            raise ValueError("Tokenizer training requires at least one input file")
        model_prefix.parent.mkdir(parents=True, exist_ok=True)
        sentencepiece.SentencePieceTrainer.train(
            input=",".join(files),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        model_path = model_prefix.with_suffix(".model")
        self.load(model_path)
        return model_path

    def load(self, model_path: Path) -> None:
        sentencepiece = importlib.import_module("sentencepiece")
        processor = sentencepiece.SentencePieceProcessor()
        processor.load(str(model_path))
        self.processor = processor
        self.model_path = model_path

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        self._ensure_loaded()
        ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        self._ensure_loaded()
        return str(self.processor.decode(list(ids)))

    @property
    def bos_id(self) -> int:
        self._ensure_loaded()
        return int(self.processor.bos_id())

    @property
    def eos_id(self) -> int:
        self._ensure_loaded()
        return int(self.processor.eos_id())

    @property
    def pad_id(self) -> int:
        self._ensure_loaded()
        return int(self.processor.pad_id())

    def _ensure_loaded(self) -> None:
        if self.processor is None:
            raise RuntimeError("Tokenizer is not loaded")
