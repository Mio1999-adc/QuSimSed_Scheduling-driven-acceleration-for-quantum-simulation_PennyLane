from __future__ import annotations

import math
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .cds import CDSRecord, RecordPool, TaskState


@dataclass
class TaskTrace:
    task_id: str
    graph_type: str
    node_type: str
    stream_id: int
    start_ns: int
    end_ns: int
    dependencies: list[str]
    memory_bytes: int
    executor: str
    cuda_stream_id: int | None = None
    error: str | None = None
    subgraph_id: str | None = None
    sm_demand: float = 0.0
    cross_stream_dependencies: int = 0
    resource_admission: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TaskCallable = Callable[[int], Any]


class ResourceAwareScheduler:
    """Completion-driven scheduling with persistent state leases and SM admission.

    SM units are estimated fractions of whole-device capacity, NOT reserved
    physical SMs. VQC tasks supply automatic workload estimates or explicit
    overrides; this scheduler enforces their summed admission budget. A worker must not return before its device work has
    completed; CUDA adapters synchronize a completion event before returning.
    Host timestamps include submission/observation overhead, not kernel time.
    """

    def __init__(self, pool: RecordPool, streams: int, memory_budget_bytes: int,
                 safety_factor: float = 0.8, *, sm_capacity: float = 1.0,
                 affinity_weight: float = 1.0, sync_weight: float = 1.0,
                 aging_weight: float = 0.01, partition_size: int = 16,
                 baseline_memory_bytes: int = 0,
                 memory_available: Callable[[], int] | None = None,
                 memory_allocated: Callable[[], int] | None = None,
                 gpu_sm_count: int | None = None,
                 parallelism_weight: float = 1.0, resource_weight: float = 1.0) -> None:
        if streams < 1 or memory_budget_bytes < 1:
            raise ValueError("streams and memory_budget_bytes must be positive")
        if not 0 < safety_factor <= 1 or not math.isfinite(sm_capacity) or sm_capacity <= 0:
            raise ValueError("invalid safety factor or SM capacity")
        if baseline_memory_bytes < 0:
            raise ValueError("baseline memory must be nonnegative")
        if gpu_sm_count is not None and gpu_sm_count < 1:
            raise ValueError("GPU SM count must be positive")
        if memory_allocated is not None and memory_available is None:
            raise ValueError("allocation telemetry requires available-memory telemetry")
        if any(not math.isfinite(w) or w < 0 for w in (affinity_weight, sync_weight, aging_weight, parallelism_weight, resource_weight)):
            raise ValueError("heuristic weights must be finite and nonnegative")
        pool.validate()
        self.partitions = pool.partition(partition_size)
        self.pool, self.streams = pool, streams
        self.limit = int(memory_budget_bytes * safety_factor)
        self.baseline = baseline_memory_bytes
        self.sm_capacity = sm_capacity
        self.affinity_weight, self.sync_weight, self.aging_weight = affinity_weight, sync_weight, aging_weight
        self.memory_available = memory_available
        self.memory_allocated = memory_allocated
        self.safety_factor = safety_factor
        self.gpu_sm_count = gpu_sm_count
        self.parallelism_weight, self.resource_weight = parallelism_weight, resource_weight
        self.trace: list[TaskTrace] = []
        self.peak_reserved_bytes = 0
        self.peak_sm_demand = 0.0

    def run(self, tasks: dict[str, TaskCallable]) -> list[TaskTrace]:
        missing = set(self.pool.records) - set(tasks)
        if missing:
            raise KeyError(f"no callable for nodes: {sorted(missing)}")
        costs: dict[str, int] = {}
        remaining = Counter(r.resource_group for r in self.pool if r.resource_group is not None)
        for r in self.pool:
            if r.resource_group is not None:
                costs[r.resource_group] = max(costs.get(r.resource_group, 0), r.group_memory_bytes)
            if r.sm_demand > self.sm_capacity:
                raise ValueError(f"task {r.node_id} exceeds SM capacity")
        for r in self.pool:
            if self.baseline + r.memory_bytes + costs.get(r.resource_group, 0) > self.limit:
                raise MemoryError(f"task {r.node_id} cannot fit in memory limit")
        if self.baseline > self.limit:
            raise MemoryError("persistent outputs cannot fit in memory limit")
        self.trace, self.peak_reserved_bytes, self.peak_sm_demand = [], self.baseline, 0.0
        ready = {r.node_id: 0 for r in self.pool.initialize()}
        available = set(range(self.streams))
        running: dict[Future, tuple[CDSRecord, int, int, int, dict]] = {}
        leased: set[str] = set()
        affinity: dict[str, int] = {}
        reserved, used_sm, tick = self.baseline, 0.0, 0
        allocated_at_start = self.memory_allocated() if self.memory_allocated else 0

        with ThreadPoolExecutor(max_workers=self.streams, thread_name_prefix="qusimsed") as executor:
            while ready or running:
                while ready and available:
                    candidates = []
                    # Account for both materialized allocations and promised
                    # reservations: cudaMemGetInfo alone cannot see queued work.
                    # Add back this run's already materialized memory once, so
                    # it is not charged both as a lease and as reduced free VRAM.
                    allocated_before = self.memory_allocated() if self.memory_allocated else 0
                    free_now = self.memory_available() if self.memory_available else None
                    allocated_after = self.memory_allocated() if self.memory_allocated else 0
                    materialized = max(0, min(allocated_before, allocated_after) - allocated_at_start)
                    live_limit = self.limit if free_now is None else min(
                        self.limit, int(max(0, free_now + materialized) * self.safety_factor))
                    memory_headroom = max(0, live_limit - reserved)
                    sm_headroom = max(0.0, self.sm_capacity - used_sm)
                    for key, since in ready.items():
                        r = self.pool.records[key]
                        new_lease = costs.get(r.resource_group, 0) if r.resource_group not in leased else 0
                        required = r.memory_bytes + new_lease
                        if reserved + required > live_limit or r.sm_demand > sm_headroom + 1e-12:
                            continue
                        # Favor work that unlocks successors and leaves room
                        # for other ready tasks. These are heuristic terms in
                        # the paper's objective, separate from hard feasibility.
                        successors_ready = sum(self.pool.records[p].ready_counter == 1 for p in r.successors())
                        parallelism = min(self.streams, successors_ready) / self.streams
                        pressure = (required / max(1, memory_headroom)
                                    + r.sm_demand / max(1e-12, sm_headroom))
                        for slot in sorted(available):
                            cross = sum(self.pool.records[p].target_stream != slot for p in r.parents)
                            home = affinity.get(r.resource_group or r.subgraph_id)
                            score = (r.priority + self.aging_weight * (tick - since)
                                     + self.affinity_weight * (home == slot)
                                     - self.sync_weight * r.sync_cost * cross
                                     + self.parallelism_weight * parallelism
                                     - self.resource_weight * pressure)
                            candidates.append((score, -since, key, -slot, required, cross, parallelism, pressure))
                    if not candidates:
                        break
                    score, _, key, negative_slot, required, cross, parallelism, pressure = max(candidates)
                    r, slot = self.pool.records[key], -negative_slot
                    admission = {"memory_limit_bytes": live_limit, "memory_reserved_before_bytes": reserved,
                                 "task_incremental_memory_bytes": required, "live_allocatable_bytes": free_now,
                                 "materialized_run_bytes": materialized, "sm_capacity": self.sm_capacity,
                                 "sm_used_before": used_sm, "sm_available_before": sm_headroom,
                                 "task_sm_demand": r.sm_demand, "gpu_sm_count": self.gpu_sm_count,
                                 "available_sm_equivalents": None if self.gpu_sm_count is None else sm_headroom * self.gpu_sm_count,
                                 "task_sm_equivalents": None if self.gpu_sm_count is None else r.sm_demand * self.gpu_sm_count,
                                 "selection_score": score, "parallelism_benefit": parallelism, "resource_pressure": pressure}
                    del ready[key]
                    available.remove(slot)
                    r.state, r.target_stream = TaskState.RUNNING, slot
                    affinity[r.resource_group or r.subgraph_id] = slot
                    if r.resource_group is not None:
                        leased.add(r.resource_group)
                    reserved += required
                    used_sm += r.sm_demand
                    self.peak_reserved_bytes = max(self.peak_reserved_bytes, reserved)
                    self.peak_sm_demand = max(self.peak_sm_demand, used_sm)
                    started = time.perf_counter_ns()
                    running[executor.submit(tasks[key], slot)] = (r, slot, started, cross, admission)
                    tick += 1
                if not running:
                    if ready:
                        raise MemoryError("no feasible task: live memory pressure or persistent resource leases block progress")
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    r, slot, started, cross, admission = running.pop(future)
                    ended = time.perf_counter_ns()
                    result, error = None, None
                    try:
                        result = future.result()
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    cuda_id = result.get("cuda_stream_id") if isinstance(result, dict) else None
                    self.trace.append(TaskTrace(r.node_id, r.graph_type, r.node_type, slot,
                                                started, ended, sorted(r.parents), r.memory_bytes,
                                                "cuda" if cuda_id is not None else "host-thread",
                                                cuda_id, error, r.subgraph_id, r.sm_demand, cross, admission))
                    available.add(slot)
                    reserved -= r.memory_bytes
                    used_sm = max(0.0, used_sm - r.sm_demand)
                    if error:
                        r.state = TaskState.REJECTED
                        raise RuntimeError(f"task {r.node_id} failed: {error}")
                    r.state = TaskState.DONE
                    if r.resource_group is not None:
                        remaining[r.resource_group] -= 1
                        if remaining[r.resource_group] == 0:
                            reserved -= costs[r.resource_group]
                            leased.remove(r.resource_group)
                    for key in sorted(r.successors()):
                        child = self.pool.records[key]
                        child.ready_counter -= 1
                        if child.ready_counter == 0:
                            child.state = TaskState.READY
                            ready[key] = tick
        if any(r.state is not TaskState.DONE for r in self.pool):
            raise RuntimeError("scheduler stopped before all tasks completed")
        return self.trace
