"""Online chat model adapters for AI Genesis.

These clients are intentionally lightweight and optional: they let the GUI answer via
an internet-hosted model when local Genesis checkpoints/tokenizers are not ready yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class OnlineModelConfig:
    """Runtime configuration for an online model endpoint."""

    enabled: bool = False
    provider: str = "openai_compatible"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 60


class OnlineModelClient:
    """Small adapter for OpenAI-compatible, Hugging Face, and Ollama-style chat APIs."""

    def __init__(self, config: OnlineModelConfig) -> None:
        self.config = config

    def available(self) -> bool:
        """Return True when the selected backend has enough local configuration to run."""
        if not self.config.enabled:
            return False
        if self.config.provider == "ollama":
            return True
        if self.config.provider in {"openai_compatible", "huggingface"}:
            return bool(os.getenv(self.config.api_key_env))
        return False

    def status_message(self) -> str:
        """Human-readable readiness message for diagnostics and GUI errors."""
        if not self.config.enabled:
            return "Интернет-модель выключена в configs/model.yaml."
        if self.config.provider not in {"openai_compatible", "huggingface", "ollama"}:
            return f"Неподдерживаемый online provider: {self.config.provider}."
        if self.config.provider != "ollama" and not os.getenv(self.config.api_key_env):
            return f"Не задан API-ключ в переменной окружения {self.config.api_key_env}."
        return f"Готово: {self.config.provider}:{self.config.model}."

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a response from the configured online backend."""
        if self.config.provider == "openai_compatible":
            return self._generate_openai_compatible(prompt, max_new_tokens)
        if self.config.provider == "huggingface":
            return self._generate_huggingface(prompt, max_new_tokens)
        if self.config.provider == "ollama":
            return self._generate_ollama(prompt)
        raise ValueError(f"Unsupported online provider: {self.config.provider}")

    def _generate_openai_compatible(self, prompt: str, max_new_tokens: int) -> str:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(self.status_message())
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "Ты AI Genesis. Отвечай полезно, кратко и по-русски, если пользователь пишет по-русски."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.7,
        }
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"].get("content", "")).strip()

    def _generate_huggingface(self, prompt: str, max_new_tokens: int) -> str:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(self.status_message())
        endpoint = self.config.base_url.rstrip("/")
        if not endpoint or endpoint == "https://api.openai.com/v1":
            endpoint = "https://api-inference.huggingface.co/models"
        url = f"{endpoint}/{self.config.model}"
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.7}},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            generated = str(data[0].get("generated_text", ""))
            return generated.removeprefix(prompt).strip() or generated.strip()
        if isinstance(data, dict):
            return str(data.get("generated_text") or data.get("summary_text") or data).strip()
        return str(data).strip()

    def _generate_ollama(self, prompt: str) -> str:
        endpoint = self.config.base_url.rstrip("/") or "http://localhost:11434"
        response = requests.post(
            endpoint + "/api/generate",
            json={"model": self.config.model, "prompt": prompt, "stream": False},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()
