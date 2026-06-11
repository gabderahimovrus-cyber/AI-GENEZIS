"""Chat engine combining model generation, memory, and RAG context."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ai_genesis.knowledge.vector_store import VectorStore
from ai_genesis.memory.sqlite_memory import MemoryStore
from ai_genesis.model.tokenizer import GenesisTokenizer
from ai_genesis.model.transformer import GenesisTransformer


class ChatEngine:
    """Provides local chat without cloud APIs."""

    def __init__(
        self,
        model: GenesisTransformer | None = None,
        tokenizer: GenesisTokenizer | None = None,
        memory: MemoryStore | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory or MemoryStore()
        self.vector_store = vector_store or VectorStore()

    def load_model(self, checkpoint: Path) -> None:
        self.model, _ = GenesisTransformer.load_checkpoint(checkpoint)

    def set_tokenizer(self, model_path: Path) -> None:
        self.tokenizer = GenesisTokenizer(model_path)

    def answer(self, user_message: str, max_new_tokens: int = 80) -> str:
        self.memory.add_dialogue("user", user_message)
        context = self._context(user_message)
        if self.tokenizer is None:
            response = "Токенизатор ещё не загружен. Обучите или загрузите tokenizer.model."
        elif self.model is None:
            response = "Модель ещё не загружена. Загрузите checkpoint Genesis-v1 или начните обучение."
        else:
            response = self._generate(f"Контекст:\n{context}\nПользователь: {user_message}\nGenesis:", max_new_tokens)
        self.memory.add_dialogue("genesis", response)
        return response

    def _context(self, message: str) -> str:
        matches = self.vector_store.search(message, k=3)
        return "\n".join(text for text, _ in matches)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        torch = importlib.import_module("torch")
        ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        output_ids = self.model.generate(input_ids.to(device), max_new_tokens=max_new_tokens)[0].tolist()
        decoded = self.tokenizer.decode(output_ids[len(ids) :])
        return decoded.strip() or "Я пока учусь и не смог сформировать ответ."
