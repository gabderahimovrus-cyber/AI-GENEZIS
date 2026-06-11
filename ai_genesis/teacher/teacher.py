"""Rule-based Teacher AI for curriculum learning and self-play dataset curation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ai_genesis.config import ModelConfig
from ai_genesis.model.online import OnlineModelClient, OnlineModelConfig


@dataclass(slots=True)
class TeachingTask:
    topic: str
    question: str
    expected_traits: list[str]
    level: str = "facts"


class TeacherSystem:
    """Creates tasks and evaluates answers without editing code or model weights directly."""

    curriculum_levels = ["words", "sentences", "facts", "logic", "dialogues"]

    def suggest_topics(self) -> list[str]:
        return [
            "Python basics",
            "machine learning terminology",
            "Linux safety rules",
            "data cleaning",
            "Transformer architecture",
        ]

    def create_questions(self, topic: str, count: int = 5, level: str = "facts") -> list[TeachingTask]:
        templates = {
            "words": "Define the key term from {topic} #{index}.",
            "sentences": "Write one clear sentence about {topic} #{index}.",
            "facts": "Explain {topic} concept #{index} in simple language.",
            "logic": "Reason step-by-step about a practical {topic} problem #{index}.",
            "dialogues": "Answer a user dialogue question about {topic} #{index}.",
        }
        template = templates.get(level, templates["facts"])
        return [TeachingTask(topic=topic, question=template.format(topic=topic, index=index + 1), expected_traits=["clear", "accurate", "concise"], level=level) for index in range(count)]

    def curriculum(self, topic: str, per_level: int = 3) -> list[TeachingTask]:
        tasks: list[TeachingTask] = []
        for level in self.curriculum_levels:
            tasks.extend(self.create_questions(topic, per_level, level=level))
        return tasks

    def score_answer(self, question: str, answer: str) -> dict[str, float | str]:
        length_score = min(len(answer.split()) / 80.0, 1.0)
        relevance_score = 1.0 if any(word.lower().strip("?.!,") in answer.lower() for word in question.split()[:4]) else 0.4
        clarity_score = 0.8 if len(answer.strip()) > 20 and "\x00" not in answer else 0.2
        score = round((length_score + relevance_score + clarity_score) / 3.0, 3)
        return {"score": score, "feedback": "Use as training data if score is high; revise if low."}

    def self_play(self, genesis_answer: Callable[[str], str], topic: str, output_path: Path, count: int = 10, min_score: float = 0.65) -> list[dict[str, object]]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        accepted: list[dict[str, object]] = []
        for task in self.curriculum(topic, per_level=max(1, count // len(self.curriculum_levels)))[:count]:
            answer = genesis_answer(task.question)
            score = self.score_answer(task.question, answer)
            record = {"task": asdict(task), "answer": answer, "score": score["score"]}
            if float(score["score"]) >= min_score:
                accepted.append(record)
        with output_path.open("a", encoding="utf-8") as file:
            for record in accepted:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return accepted

@dataclass(slots=True)
class AssistantModelConfig:
    """Configuration for an optional local helper model used as Teacher/Knowledge Assistant."""

    backend: str
    model: str
    role: str = "teacher"


class LocalAssistantModel:
    """Adapter for optional local helper models via Transformers, GGUF llama-cpp, or Ollama.

    The adapter is intentionally separate from Genesis model training: helper output can be
    used to create or evaluate data, but this class never edits project code or weights.
    """

    def __init__(self, config: AssistantModelConfig) -> None:
        self.config = config
        self._pipeline = None
        self._llm = None

    def available(self) -> bool:
        import importlib.util

        if self.config.backend in {"openai_compatible", "huggingface"}:
            return self._online_client().available()
        if self.config.backend == "transformers":
            return importlib.util.find_spec("transformers") is not None
        if self.config.backend == "gguf":
            return importlib.util.find_spec("llama_cpp") is not None
        if self.config.backend == "ollama":
            return importlib.util.find_spec("requests") is not None
        return False

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.config.backend in {"openai_compatible", "huggingface"}:
            return self._online_client().generate(prompt, max_new_tokens=max_new_tokens)
        if self.config.backend == "transformers":
            return self._generate_transformers(prompt, max_new_tokens)
        if self.config.backend == "gguf":
            return self._generate_gguf(prompt, max_new_tokens)
        if self.config.backend == "ollama":
            return self._generate_ollama(prompt)
        raise ValueError(f"Unsupported assistant backend: {self.config.backend}")

    def _online_client(self) -> OnlineModelClient:
        model_config = ModelConfig.load()
        return OnlineModelClient(
            OnlineModelConfig(
                enabled=True,
                provider=self.config.backend,
                model=self.config.model,
                base_url="https://api-inference.huggingface.co/models" if self.config.backend == "huggingface" else model_config.online_base_url,
                api_key_env=model_config.online_api_key_env,
                timeout_seconds=model_config.online_timeout_seconds,
            )
        )

    def _generate_transformers(self, prompt: str, max_new_tokens: int) -> str:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline("text-generation", model=self.config.model, device_map="auto")
        result = self._pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        return str(result[0].get("generated_text", ""))

    def _generate_gguf(self, prompt: str, max_new_tokens: int) -> str:
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(model_path=self.config.model)
        result = self._llm(prompt, max_tokens=max_new_tokens)
        return str(result["choices"][0]["text"])

    def _generate_ollama(self, prompt: str) -> str:
        import requests

        response = requests.post("http://localhost:11434/api/generate", json={"model": self.config.model, "prompt": prompt, "stream": False}, timeout=120)
        response.raise_for_status()
        return str(response.json().get("response", ""))
