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
    collect_results: bool = True
    execution_backend: Literal["torch-cuda", "torch-cpu", "pennylane"] = "torch-cuda"
    gpu_device: int = 0
    memory_budget_bytes: int | None = None
    sm_demand: float | None = None
    partition_size: int = 16
    affinity_weight: float = 1.0
    sync_weight: float = 1.0
    parallelism_weight: float = 1.0
    resource_weight: float = 1.0
    trainable_parameters: int | None = None
    reference_backend: Literal["default.qubit", "lightning.gpu"] = "default.qubit"

    @property
    def parameter_count(self) -> int:
        return self.trainable_parameters if self.trainable_parameters is not None else 3 * self.qubits * self.layers

    def validate(self) -> None:
        if self.qubits < 1 or self.layers < 1 or self.streams < 1:
            raise ValueError("qubits, layers, and streams must be positive")
        if not 0 < self.memory_safety_factor <= 1:
            raise ValueError("memory_safety_factor must be in (0, 1]")
        if self.warmup < 0 or self.iterations < 1 or self.samples < 1:
            raise ValueError("warmup >= 0; iterations and samples >= 1")
        if self.differentiation not in ("parameter-shift", "adjoint"):
            raise ValueError("unsupported differentiation method")
        if self.execution_backend not in ("torch-cuda", "torch-cpu", "pennylane"):
            raise ValueError("unsupported execution backend")
        if (self.sm_demand is not None and not 0 < self.sm_demand <= 1) or self.partition_size < 1 or self.gpu_device < 0:
            raise ValueError("invalid SM demand, partition size, or GPU index")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes < 1:
            raise ValueError("memory budget must be positive")
        if not 1 <= self.parameter_count <= 3 * self.qubits * self.layers:
            raise ValueError("trainable parameter count must be in [1, 3 * qubits * layers]")
        if self.reference_backend not in ("default.qubit", "lightning.gpu"):
            raise ValueError("unsupported independent reference backend")

    def metadata(self) -> dict:
        return asdict(self)

    @property
    def results_path(self) -> Path:
        return Path(self.output_dir)
