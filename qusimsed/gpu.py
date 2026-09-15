"""Strict GPU-only PennyLane execution helpers.

There is deliberately no CPU fallback in this module: a result produced here
is either executed by ``lightning.gpu`` or it fails before benchmarking.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPUEnvironment:
    device_name: str
    device_memory_mib: int | None
    pennylane_version: str
    backend: str
    python: str


def _nvidia_smi() -> tuple[str, int | None]:
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                   check=True, capture_output=True, text=True, timeout=10)
        name, memory = completed.stdout.strip().split(",", 1)
        return name.strip(), int(memory.strip().split()[0])
    except Exception:
        return "NVIDIA GPU (details unavailable)", None


def lightning_gpu_device(wires: int):
    try:
        import pennylane as qml
    except ImportError as exc:
        raise RuntimeError("GPU experiments require PennyLane and PennyLane-Lightning-GPU") from exc
    try:
        device = qml.device("lightning.gpu", wires=wires)
    except Exception as exc:
        raise RuntimeError("lightning.gpu could not be initialized; refusing CPU fallback") from exc
    name, memory = _nvidia_smi()
    return device, GPUEnvironment(name, memory, qml.__version__, "lightning.gpu", platform.python_version())
