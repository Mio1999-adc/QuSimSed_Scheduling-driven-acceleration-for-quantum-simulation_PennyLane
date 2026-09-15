"""Executor adapters.

The CUDA smoke executor is intentionally backend-neutral: it proves that the
scheduler can bind work to real CUDA streams, but it is not a PennyLane VQC
executor. A paper figure must identify the workload and use Nsight for kernel
overlap evidence.
"""
from __future__ import annotations

from typing import Callable


class TorchCUDAStreamExecutor:
    def __init__(self, streams: int) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for CUDA stream execution") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device is available to Torch")
        self.torch = torch
        self.streams = [torch.cuda.Stream() for _ in range(streams)]

    def wrap(self, work: Callable[[], None]) -> Callable[[int], dict]:
        def run(slot: int) -> dict:
            stream = self.streams[slot]
            with self.torch.cuda.stream(stream):
                work()
            stream.synchronize()  # scheduler completion means device completion
            return {"cuda_stream_id": int(stream.cuda_stream)}
        return run


def cuda_smoke_workload(executor: TorchCUDAStreamExecutor, matrix_size: int = 1024) -> Callable[[int], dict]:
    """A real CUDA matmul for plumbing validation only, not a VQC surrogate."""
    torch = executor.torch
    def work() -> None:
        a = torch.rand((matrix_size, matrix_size), device="cuda")
        b = torch.rand((matrix_size, matrix_size), device="cuda")
        _ = a @ b
    return executor.wrap(work)
