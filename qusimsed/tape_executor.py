"""Stream-owned execution of PennyLane unitary tapes using Torch state vectors.

This is an explicitly named backend, NOT lightning.gpu or cuStateVec. CPU is
an opt-in validation mode; requesting CUDA never falls back to CPU. Each tape
owns its state. Gate matrices use PennyLane's wire ordering (wire 0 is MSB).
"""
from __future__ import annotations

from typing import Callable


class TapeExecutor:
    def __init__(self, streams: int, *, device: str = "cuda:0") -> None:
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("only CUDA and explicit CPU validation are supported")
        if streams < 1:
            raise ValueError("streams must be positive")
        self.cuda = self.device.type == "cuda"
        if self.cuda and not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is unavailable; refusing CPU fallback")
        self.streams = ([torch.cuda.Stream(device=self.device) for _ in range(streams)]
                        if self.cuda else [None] * streams)
        self.events: dict = {}
        self.wait_counts: dict[str, int] = {}
        self.backend = "torch-cuda-statevector" if self.cuda else "torch-cpu-statevector-validation"

    @property
    def event_waits(self):
        return sum(self.wait_counts.values())

    def memory_info(self) -> tuple[int, int]:
        if not self.cuda:
            raise RuntimeError("CPU validation needs an explicit memory budget")
        return tuple(int(x) for x in self.torch.cuda.mem_get_info(self.device))

    def allocatable_bytes(self) -> int:
        # Torch's unused cache is reusable even when cudaMemGetInfo reports it
        # as occupied. Allocations from other processes remain unavailable.
        free, _ = self.memory_info()
        cached = (self.torch.cuda.memory_reserved(self.device)
                  - self.torch.cuda.memory_allocated(self.device))
        return free + int(cached)

    def wrap(self, node: str, parents: set[str], work: Callable[[], None]):
        def run(slot: int):
            if not self.cuda:
                with self.torch.no_grad():
                    work()
                return None
            stream = self.streams[slot]
            with self.torch.cuda.device(self.device), self.torch.cuda.stream(stream), self.torch.no_grad():
                waits = 0
                for parent in sorted(parents):
                    event, parent_slot = self.events[parent]
                    if parent_slot != slot:
                        stream.wait_event(event)
                        waits += 1
                self.wait_counts[node] = waits
                self.torch.cuda.nvtx.range_push(node)
                try:
                    work()
                finally:
                    self.torch.cuda.nvtx.range_pop()
                event = self.torch.cuda.Event()
                event.record(stream)
                self.events[node] = (event, slot)
            # Host readiness corresponds to device completion. No device-wide
            # barrier: other streams may continue while this worker waits.
            event.synchronize()
            return {"cuda_stream_id": int(stream.cuda_stream)}
        return run

    def close(self) -> None:
        for stream in self.streams:
            if stream is not None:
                stream.synchronize()
        self.events.clear()

    def tensor(self, value, *, complex_value: bool = False):
        return self.torch.as_tensor(value, device=self.device,
                                   dtype=self.torch.complex128 if complex_value else self.torch.float64)

    def protect(self, value) -> None:
        if not self.cuda:
            return
        if self.torch.is_tensor(value) and value.is_cuda:
            value.record_stream(self.torch.cuda.current_stream(self.device))
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.protect(item)
        elif isinstance(value, dict):
            for item in value.values():
                self.protect(item)

    def initial_state(self, qubits: int):
        state = self.torch.zeros(1 << qubits, dtype=self.torch.complex128, device=self.device)
        state[0] = 1
        return state

    def apply(self, state, matrix, wires: tuple[int, ...], qubits: int):
        self.protect(state)
        matrix = self.tensor(matrix, complex_value=True)
        if not wires:
            return state * matrix.reshape(())
        order = list(wires) + [w for w in range(qubits) if w not in wires]
        inverse = [order.index(w) for w in range(qubits)]
        folded = state.reshape([2] * qubits).permute(order).reshape(1 << len(wires), -1)
        result = matrix @ folded
        return result.reshape([2] * qubits).permute(inverse).reshape(-1)
