"""PyTorch datasets for tokenized Genesis corpora."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


def _torch() -> Any:
    return importlib.import_module("torch")


def _dataset_base() -> Any:
    return importlib.import_module("torch.utils.data").Dataset


class TokenBlockDataset(_dataset_base()):
    """Turns JSONL token records into fixed-length next-token examples."""

    def __init__(self, jsonl_path: Path, context_length: int) -> None:
        self.context_length = context_length
        tokens: list[int] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tokens.extend(json.loads(line)["input_ids"])
        if len(tokens) < context_length + 1:
            raise ValueError("Dataset is too small for the configured context length")
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch = _torch()
        chunk = self.tokens[index : index + self.context_length + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
