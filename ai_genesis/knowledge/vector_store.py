"""FAISS-backed vector knowledge store with a deterministic local fallback."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import pickle
from pathlib import Path
from typing import Any


class VectorStore:
    """Stores knowledge vectors for RAG-style context lookup."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension
        self.texts: list[str] = []
        self._faiss_available = importlib.util.find_spec("faiss") is not None and importlib.util.find_spec("numpy") is not None
        if self._faiss_available:
            faiss = __import__("faiss")
            self.index: Any = faiss.IndexFlatIP(dimension)
            self._vectors: list[list[float]] = []
        else:
            self.index = None
            self._vectors = []

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def add_texts(self, texts: list[str]) -> None:
        if not texts:
            return
        vectors = [self.embed(text) for text in texts]
        self.texts.extend(texts)
        self._vectors.extend(vectors)
        if self._faiss_available:
            np = __import__("numpy")
            self.index.add(np.array(vectors, dtype="float32"))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not self.texts:
            return []
        query_vector = self.embed(query)
        if self._faiss_available:
            np = __import__("numpy")
            scores, indices = self.index.search(np.array([query_vector], dtype="float32"), min(k, len(self.texts)))
            return [(self.texts[int(index)], float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]
        scored = [
            (index, sum(left * right for left, right in zip(vector, query_vector)))
            for index, vector in enumerate(self._vectors)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(self.texts[index], float(score)) for index, score in scored[:k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump({"dimension": self.dimension, "texts": self.texts}, file)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        with path.open("rb") as file:
            payload = pickle.load(file)
        store = cls(dimension=payload["dimension"])
        store.add_texts(payload["texts"])
        return store
