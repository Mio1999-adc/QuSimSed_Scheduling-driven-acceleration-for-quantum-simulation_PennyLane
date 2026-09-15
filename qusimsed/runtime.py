"""Runtime capability detection and honest execution-mode selection."""
from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal


ExecutionMode = Literal["cuda-streams", "cpu-threads"]


@dataclass(frozen=True)
class RuntimeCapabilities:
    nvidia_driver_visible: bool
    gpu_name: str | None
    torch_cuda_usable: bool
    pennylane_lightning_gpu_usable: bool
    selected_mode: ExecutionMode
    reason: str

    def metadata(self) -> dict:
        return asdict(self)


def _nvidia_gpu_name() -> str | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip().splitlines()[0]
    except Exception:
        return None


def detect_runtime() -> RuntimeCapabilities:
    """Probe without importing a GPU backend unless it is installed.

    ``cuda-streams`` requires a usable CUDA Python runtime, not merely an
    NVIDIA driver. PennyLane GPU availability is recorded separately because
    an actual VQC experiment requires both PennyLane and lightning.gpu.
    """
    name = _nvidia_gpu_name(); torch_cuda = False; lightning_gpu = False
    if importlib.util.find_spec("torch"):
        try:
            import torch
            torch_cuda = bool(torch.cuda.is_available())
        except Exception:
            pass
    if importlib.util.find_spec("pennylane"):
        try:
            import pennylane as qml
            _ = qml.device("lightning.gpu", wires=1)
            lightning_gpu = True
        except Exception:
            pass
    if torch_cuda:
        return RuntimeCapabilities(bool(name), name, torch_cuda, lightning_gpu, "cuda-streams",
                                   "CUDA-enabled Torch is available; stream executor may be used.")
    if name:
        return RuntimeCapabilities(True, name, False, lightning_gpu, "cpu-threads",
                                   "NVIDIA driver found but no usable CUDA Python runtime; using CPU worker threads.")
    return RuntimeCapabilities(False, None, False, lightning_gpu, "cpu-threads",
                               "No NVIDIA GPU detected; using CPU worker threads.")
