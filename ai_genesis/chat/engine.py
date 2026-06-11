"""Chat engine combining model generation, memory, and RAG context."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ai_genesis.config import ModelConfig
from ai_genesis.knowledge.vector_store import VectorStore
from ai_genesis.memory.sqlite_memory import MemoryStore
from ai_genesis.model.tokenizer import GenesisTokenizer
from ai_genesis.model.online import OnlineModelClient, OnlineModelConfig
from ai_genesis.model.transformer import GenesisTransformer


class ChatEngine:
    """Provides chat through a local Genesis model or optional online fallback."""

    def __init__(
        self,
        model: GenesisTransformer | None = None,
        tokenizer: GenesisTokenizer | None = None,
        memory: MemoryStore | None = None,
        vector_store: VectorStore | None = None,
        online_client: OnlineModelClient | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory or MemoryStore()
        self.vector_store = vector_store or VectorStore()
        self.model_config = model_config or ModelConfig.load()
        self.online_client = online_client or self._online_client_from_config(self.model_config)

    def load_model(self, checkpoint: Path) -> None:
        self.model, _ = GenesisTransformer.load_checkpoint(checkpoint)

    def set_tokenizer(self, model_path: Path) -> None:
        self.tokenizer = GenesisTokenizer(model_path)

    def answer(self, user_message: str, max_new_tokens: int = 80) -> str:
        self.memory.add_dialogue("user", user_message)
        context = self._context(user_message)
        prompt = f"Контекст:\n{context}\nПользователь: {user_message}\nGenesis:"
        if self.model is not None and self.tokenizer is not None:
            response = self._generate(prompt, max_new_tokens)
        elif self.online_client and self.online_client.config.enabled:
            response = self._generate_online(prompt, max_new_tokens)
        elif self.tokenizer is None:
            response = "Токенизатор ещё не загружен. Обучите/загрузите tokenizer.model или включите internet-модель в configs/model.yaml."
        else:
            response = "Модель ещё не загружена. Загрузите checkpoint Genesis-v1 или включите internet-модель в configs/model.yaml."
        self.memory.add_dialogue("genesis", response)
        return response

    def _online_client_from_config(self, config: ModelConfig) -> OnlineModelClient:
        return OnlineModelClient(_online_config_from_model_config(config))

    def _generate_online(self, prompt: str, max_new_tokens: int) -> str:
        if self.online_client is None:
            return "Интернет-модель не настроена."
        try:
            response = self.online_client.generate(prompt, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            response = f"Интернет-модель не смогла ответить: {exc}"
        return response.strip() or "Интернет-модель вернула пустой ответ."

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


def _online_config_from_model_config(config: ModelConfig) -> OnlineModelConfig:
    return OnlineModelConfig(
        enabled=config.online_enabled,
        provider=config.online_provider,
        model=config.online_model,
        base_url=config.online_base_url,
        api_key_env=config.online_api_key_env,
        timeout_seconds=config.online_timeout_seconds,
    )
