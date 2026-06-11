"""Genesis-v1 decoder-only Transformer implemented in PyTorch."""

from __future__ import annotations

import importlib
import importlib.util
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_genesis.config import ModelConfig


def _torch() -> Any:
    return importlib.import_module("torch")


def _nn() -> Any:
    return importlib.import_module("torch.nn")


def _module_base() -> Any:
    if importlib.util.find_spec("torch") is None:
        return object
    return importlib.import_module("torch.nn").Module


class GenesisTransformer(_module_base()):
    """A compact GPT-style model for local training from scratch."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("PyTorch is required to instantiate GenesisTransformer. Install project dependencies first.")
        super().__init__()
        torch = _torch()
        nn = _nn()
        self.config = config or ModelConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.position_embedding = nn.Embedding(self.config.context_length, self.config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_size * 8,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(self.config.context_length, self.config.context_length), diagonal=1).bool(),
            persistent=False,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: Any) -> None:
        nn = _nn()
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Any, targets: Any | None = None) -> tuple[Any, Any | None]:
        torch = _torch()
        _, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context length {self.config.context_length}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = self.causal_mask[:seq_len, :seq_len]
        hidden = self.blocks(hidden, mask=mask)
        logits = self.output(self.norm(hidden))
        loss = None
        if targets is not None:
            loss_fn = _nn().CrossEntropyLoss()
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def approximate_parameter_label(self) -> str:
        return f"{round(self.parameter_count / 1_000_000):.0f}M"

    def save_checkpoint(self, path: Path, optimizer: Any | None = None, step: int = 0) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"config": asdict(self.config), "model": self.state_dict(), "step": step}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str = "cpu") -> tuple["GenesisTransformer", dict[str, Any]]:
        torch = _torch()
        payload = torch.load(path, map_location=map_location)
        model = cls(ModelConfig(**payload["config"]))
        model.load_state_dict(payload["model"])
        return model, payload

    def generate(self, input_ids: Any, max_new_tokens: int = 64, temperature: float = 0.8, top_k: int = 40) -> Any:
        torch = _torch()
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.context_length :]
                logits, _ = self(context)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -math.inf
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
