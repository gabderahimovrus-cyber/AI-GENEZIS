"""Benchmark Suite for math, logic, coding, history, and general knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ai_genesis.logging_system import logger


@dataclass(slots=True)
class BenchmarkItem:
    category: str
    prompt: str
    answer: str


class BenchmarkSuite:
    """Compares candidate and production models with deterministic local test sets."""

    def __init__(self) -> None:
        self.items = [
            BenchmarkItem("math", "What is 12 + 30?", "42"),
            BenchmarkItem("logic", "If all cats are animals and Tom is a cat, what is Tom?", "animal"),
            BenchmarkItem("programming", "Name the Python keyword used to define a function.", "def"),
            BenchmarkItem("history", "Who was the first president of the United States?", "Washington"),
            BenchmarkItem("general", "What planet is known as the Red Planet?", "Mars"),
        ]

    def run(self, answer_fn: Callable[[str], str]) -> dict[str, float]:
        correct = 0
        by_category: dict[str, list[int]] = {}
        for item in self.items:
            response = answer_fn(item.prompt)
            hit = int(item.answer.lower() in response.lower())
            correct += hit
            by_category.setdefault(item.category, []).append(hit)
        metrics = {f"{category}_accuracy": sum(values) / len(values) for category, values in by_category.items()}
        metrics["accuracy"] = correct / len(self.items)
        logger.log(f"Benchmark завершён: accuracy={metrics['accuracy']:.3f}")
        return metrics

    def candidate_is_better(self, candidate_metrics: dict[str, float], production_metrics: dict[str, float] | None) -> bool:
        if not production_metrics:
            return True
        return candidate_metrics.get("accuracy", 0.0) > production_metrics.get("accuracy", 0.0)
