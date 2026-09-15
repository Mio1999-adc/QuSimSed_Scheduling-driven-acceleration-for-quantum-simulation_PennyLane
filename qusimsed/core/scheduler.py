from __future__ import annotations

import heapq
import time
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TaskCallable = Callable[[int], Any]


class ResourceAwareScheduler:
    """Dependency-correct asynchronous scheduler.

    The worker receives the logical stream slot.  CUDA evidence is opt-in:
    a worker may return ``{\"cuda_stream_id\": int}``; otherwise traces are
    explicitly labelled as host-thread execution and cannot support a GPU
    overlap claim.
    """

    def __init__(self, pool: RecordPool, streams: int, memory_budget_bytes: int,
                 safety_factor: float = 0.8) -> None:
        if streams < 1 or memory_budget_bytes < 1:
            raise ValueError("streams and memory_budget_bytes must be positive")
        if not 0 < safety_factor <= 1:
            raise ValueError("safety_factor must be in (0, 1]")
        pool.validate()
        self.pool, self.streams = pool, streams
        self.limit = int(memory_budget_bytes * safety_factor)
        self.trace: list[TaskTrace] = []
        self.peak_reserved_bytes = 0

    def run(self, tasks: dict[str, TaskCallable]) -> list[TaskTrace]:
        missing = set(self.pool.records) - set(tasks)
        if missing:
            raise KeyError(f"no callable for nodes: {sorted(missing)}")
        if any(record.memory_bytes > self.limit for record in self.pool):
            too_large = [r.node_id for r in self.pool if r.memory_bytes > self.limit]
            raise MemoryError(f"tasks cannot fit in memory limit: {too_large}")
        self.trace, self.peak_reserved_bytes = [], 0
        queue: list[tuple[int, int, str]] = []
        sequence = 0
        for record in self.pool.initialize():
            heapq.heappush(queue, (-record.priority, sequence, record.node_id)); sequence += 1
        available = list(range(self.streams))
        running: dict[Future, tuple[CDSRecord, int, int]] = {}
        reserved = 0

        with ThreadPoolExecutor(max_workers=self.streams, thread_name_prefix="qusimsed") as executor:
            while queue or running:
                launched = False
                deferred: list[tuple[int, int, str]] = []
                while queue and available:
                    item = heapq.heappop(queue); record = self.pool.records[item[2]]
                    if reserved + record.memory_bytes > self.limit:
                        deferred.append(item)
                        continue
                    slot = available.pop(0)
                    record.state, record.target_stream = TaskState.RUNNING, slot
                    started = time.perf_counter_ns()
                    future = executor.submit(tasks[record.node_id], slot)
                    running[future] = (record, slot, started)
                    reserved += record.memory_bytes
                    self.peak_reserved_bytes = max(self.peak_reserved_bytes, reserved)
                    launched = True
                for item in deferred:
                    heapq.heappush(queue, item)
                if not running:
                    if queue:
                        raise MemoryError("ready queue is blocked by memory accounting")
                    break
                if launched and queue and available:
                    continue
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    record, slot, started = running.pop(future)
                    ended = time.perf_counter_ns(); result: Any = None; error = None
                    try:
                        result = future.result()
                    except Exception as exc:  # retain trace before surfacing failure
                        error = f"{type(exc).__name__}: {exc}"
                    cuda_id = result.get("cuda_stream_id") if isinstance(result, dict) else None
                    self.trace.append(TaskTrace(record.node_id, record.graph_type, record.node_type,
                                                 slot, started, ended, sorted(record.parents),
                                                 record.memory_bytes,
                                                 "cuda" if cuda_id is not None else "host-thread",
                                                 cuda_id, error))
                    available.append(slot); available.sort(); reserved -= record.memory_bytes
                    if error:
                        record.state = TaskState.REJECTED
                        raise RuntimeError(f"task {record.node_id} failed: {error}")
                    record.state = TaskState.DONE
                    for child_id in record.successors():
                        child = self.pool.records[child_id]
                        child.ready_counter -= 1
                        if child.ready_counter == 0:
                            child.state = TaskState.READY
                            heapq.heappush(queue, (-child.priority, sequence, child_id)); sequence += 1
        if any(record.state is not TaskState.DONE for record in self.pool):
            raise RuntimeError("scheduler stopped before all tasks completed")
        return self.trace
