"""Dataset construction pipeline: clean, deduplicate, normalize, tokenize."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from ai_genesis.model.tokenizer import GenesisTokenizer

_WHITESPACE = re.compile(r"\s+")


class DatasetBuilder:
    """Builds text and token datasets from collected educational material."""

    def clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = _WHITESPACE.sub(" ", text)
        return text.strip()

    def deduplicate(self, texts: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for text in texts:
            cleaned = self.clean_text(text)
            if len(cleaned) < 40:
                continue
            digest = hashlib.sha256(cleaned.lower().encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            result.append(cleaned)
        return result

    def write_clean_corpus(self, texts: Iterable[str], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = self.deduplicate(texts)
        output_path.write_text("\n".join(cleaned), encoding="utf-8")
        return output_path

    def tokenize_corpus(self, corpus_path: Path, tokenizer: GenesisTokenizer, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            ids = tokenizer.encode(line, add_bos=True, add_eos=True)
            records.append({"text": line, "input_ids": ids})
        output_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")
        return output_path
