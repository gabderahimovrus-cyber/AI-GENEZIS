"""Quality metrics for Genesis training runs."""

from __future__ import annotations

import importlib
import math
import time
from typing import Any


def _torch() -> Any:
    return importlib.import_module("torch")


class Evaluator:
    """Computes loss, validation loss, perplexity, and generation speed."""

    def evaluate_loss(self, model: Any, dataloader: Any, device: str) -> dict[str, float]:
        torch = _torch()
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                _, loss = model(inputs, targets)
                losses.append(float(loss.item()))
        validation_loss = sum(losses) / max(len(losses), 1)
        return {"validation_loss": validation_loss, "perplexity": math.exp(min(validation_loss, 20.0))}

    def generation_speed(self, model: Any, prompt_ids: Any, device: str, tokens: int = 32) -> float:
        start = time.perf_counter()
        model.generate(prompt_ids.to(device), max_new_tokens=tokens)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return tokens / elapsed
