from __future__ import annotations

import gc
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any

from .core.cds import RecordPool


def live_memory_snapshot() -> dict[str, int | None]:
    """Observed process and CUDA allocations; unavailable values are null, not zero."""
    rss = None
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except Exception:
        pass
    allocated = reserved = peak_allocated = peak_reserved = None
    try:
        import torch
        if torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated()); reserved = int(torch.cuda.memory_reserved())
            peak_allocated = int(torch.cuda.max_memory_allocated()); peak_reserved = int(torch.cuda.max_memory_reserved())
    except Exception:
        pass
    return {"process_rss_bytes": rss, "cuda_allocated_bytes": allocated, "cuda_reserved_bytes": reserved,
            "cuda_peak_allocated_bytes": peak_allocated, "cuda_peak_reserved_bytes": peak_reserved}


def measure_memory(callable_) -> dict:
    """Measure one real workload invocation, including CUDA peaks when supported."""
    gc.collect(); tracemalloc.start()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    before = live_memory_snapshot(); callable_(); current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); after = live_memory_snapshot()
    return {"before": before, "after": after, "python_tracemalloc_current_bytes": current, "python_tracemalloc_peak_bytes": peak,
            "process_rss_delta_bytes": None if before["process_rss_bytes"] is None or after["process_rss_bytes"] is None else after["process_rss_bytes"] - before["process_rss_bytes"],
            "cuda_allocated_delta_bytes": None if before["cuda_allocated_bytes"] is None or after["cuda_allocated_bytes"] is None else after["cuda_allocated_bytes"] - before["cuda_allocated_bytes"]}


def estimate_statevector_bytes(qubits: int, *, precision: str = "complex128",
                               retained_states: int = 1) -> int:
    """Conservative state-vector allocation estimate used to reject unsafe sweeps."""
    if qubits < 1 or retained_states < 1:
        raise ValueError("qubits and retained_states must be positive")
    element_bytes = {"complex64": 8, "complex128": 16}.get(precision)
    if element_bytes is None:
        raise ValueError("precision must be complex64 or complex128")
    return (1 << qubits) * element_bytes * retained_states


def safe_qubit_counts(candidates: list[int], total_memory_bytes: int, *, streams: int,
                      safety_factor: float, precision: str = "complex128",
                      retained_states: int = 1) -> list[dict[str, int | bool]]:
    """Planning-only admission decision; real peak allocation is still recorded at runtime."""
    limit = int(total_memory_bytes * safety_factor)
    rows = []
    for qubits in candidates:
        per_task = estimate_statevector_bytes(qubits, precision=precision, retained_states=retained_states)
        concurrent = per_task * streams
        rows.append({"qubits": qubits, "estimated_per_task_bytes": per_task,
                     "estimated_concurrent_bytes": concurrent, "admitted": concurrent <= limit})
    return rows


def deep_size(value: Any, seen: set[int] | None = None) -> int:
    seen = seen or set(); identity = id(value)
    if identity in seen: return 0
    seen.add(identity); size = sys.getsizeof(value)
    if isinstance(value, dict): size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)): size += sum(deep_size(item, seen) for item in value)
    elif hasattr(value, "__dict__"): size += deep_size(vars(value), seen)
    return size


@dataclass(frozen=True)
class MemoryReport:
    cds_bytes: int
    scheduler_metadata_bytes: int
    total_metadata_bytes: int
    nodes: int
    bytes_per_node: float


def metadata_memory(pool: RecordPool, scheduler: Any | None = None) -> MemoryReport:
    cds = deep_size(pool)
    scheduler_size = deep_size(scheduler) if scheduler is not None else 0
    total = cds + scheduler_size
    return MemoryReport(cds, scheduler_size, total, len(pool.records), total / max(1, len(pool.records)))
