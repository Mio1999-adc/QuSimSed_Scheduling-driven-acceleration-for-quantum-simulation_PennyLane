from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


Strategy = Literal[
    "sequential", "gradient-only", "quantum-only", "qusimsed",
    "batched-parameter-shift", "naive-multistream", "catalyst", "qusimsed-catalyst",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """All experiment inputs; results always serialize this complete object."""

    qubits: int = 4
    layers: int = 2
    differentiation: Literal["parameter-shift", "adjoint"] = "parameter-shift"
    strategy: Strategy = "qusimsed"
    workload: Literal["synthetic", "iris", "wine", "breast-cancer", "mnist-pca"] = "synthetic"
    seed: int = 7
    warmup: int = 1
    iterations: int = 10
    streams: int = 4
    memory_safety_factor: float = 0.8
    batch_size: int = 4
    samples: int = 16
    learning_rate: float = 0.05
    enable_trace: bool = True
    enable_nsight: bool = False
    output_dir: str = "results"

    def validate(self) -> None:
        if self.qubits < 1 or self.layers < 1 or self.streams < 1:
            raise ValueError("qubits, layers, and streams must be positive")
        if not 0 < self.memory_safety_factor <= 1:
            raise ValueError("memory_safety_factor must be in (0, 1]")
        if self.warmup < 0 or self.iterations < 1 or self.samples < 1:
            raise ValueError("warmup >= 0; iterations and samples >= 1")

    def metadata(self) -> dict:
        return asdict(self)

    @property
    def results_path(self) -> Path:
        return Path(self.output_dir)
