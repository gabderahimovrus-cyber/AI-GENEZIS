"""Local system monitoring for CPU, RAM, GPU, VRAM, and training status."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class SystemSnapshot:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    gpu_name: str = "unavailable"
    gpu_percent: float | None = None
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    gpu_temperature_c: float | None = None
    training_status: str = "idle"
    epoch: int = 0
    loss: float | None = None
    perplexity: float | None = None


class SystemMonitor:
    """Collects metrics without shelling out to arbitrary OS tools."""

    def __init__(self) -> None:
        self.training_status = "idle"
        self.epoch = 0
        self.loss: float | None = None
        self.perplexity: float | None = None

    def update_training(self, status: str, epoch: int = 0, loss: float | None = None, perplexity: float | None = None) -> None:
        self.training_status = status
        self.epoch = epoch
        self.loss = loss
        self.perplexity = perplexity

    def snapshot(self) -> SystemSnapshot:
        cpu, ram_percent, ram_gb = self._cpu_ram()
        gpu = self._gpu()
        return SystemSnapshot(
            cpu_percent=cpu,
            ram_percent=ram_percent,
            ram_used_gb=ram_gb,
            gpu_name=gpu["name"],
            gpu_percent=gpu.get("gpu_percent"),
            vram_used_mb=gpu.get("vram_used_mb"),
            vram_total_mb=gpu.get("vram_total_mb"),
            gpu_temperature_c=gpu.get("temperature"),
            training_status=self.training_status,
            epoch=self.epoch,
            loss=self.loss,
            perplexity=self.perplexity,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.snapshot())

    def _cpu_ram(self) -> tuple[float, float, float]:
        if importlib.util.find_spec("psutil") is not None:
            psutil = __import__("psutil")
            mem = psutil.virtual_memory()
            return float(psutil.cpu_percent(interval=None)), float(mem.percent), round(float(mem.used) / (1024**3), 2)
        return 0.0, 0.0, 0.0

    def _gpu(self) -> dict[str, Any]:
        if importlib.util.find_spec("torch") is None:
            return {"name": "unavailable"}
        torch = __import__("torch")
        if not torch.cuda.is_available():
            return {"name": "cpu-only"}
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        total = props.total_memory / (1024**2)
        telemetry = self._nvidia_smi()
        used = telemetry.get("vram_used_mb", max(allocated, reserved))
        return {
            "name": telemetry.get("name") or props.name,
            "vram_used_mb": round(float(used), 1),
            "vram_total_mb": round(float(telemetry.get("vram_total_mb", total)), 1),
            "gpu_percent": telemetry.get("gpu_percent"),
            "temperature": telemetry.get("temperature"),
        }

    def _nvidia_smi(self) -> dict[str, Any]:
        if shutil.which("nvidia-smi") is None:
            return {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            first = result.stdout.strip().splitlines()[0]
            name, used, total, temp, util = [part.strip() for part in first.split(",")[:5]]
            return {
                "name": name,
                "vram_used_mb": float(used),
                "vram_total_mb": float(total),
                "temperature": float(temp),
                "gpu_percent": float(util),
            }
        except Exception:
            return {}
