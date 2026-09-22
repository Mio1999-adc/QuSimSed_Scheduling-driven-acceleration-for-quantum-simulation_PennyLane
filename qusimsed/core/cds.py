from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Iterable


class TaskState(str, Enum):
    NOT_READY = "not-ready"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    REJECTED = "rejected"


@dataclass
class CDSRecord:
    node_id: str
    graph_type: str  # "circuit" or "autograd"
    node_type: str
    memory_bytes: int = 0
    priority: int = 0
    parents: set[str] = field(default_factory=set)
    children: set[str] = field(default_factory=set)
    cross_graph_links: set[str] = field(default_factory=set)
    ready_counter: int = 0
    state: TaskState = TaskState.NOT_READY
    target_stream: int | None = None
    subgraph_id: str | None = None
    sm_demand: float = 0.0
    sync_cost: float = 1.0
    # A state/workspace lease survives individual gate completions. It is
    # released only after EVERY node in this resource group has completed.
    resource_group: str | None = None
    group_memory_bytes: int = 0
    metadata: dict = field(default_factory=dict)

    def successors(self) -> set[str]:
        return self.children | self.cross_graph_links


class RecordPool:
    """Flat CDS storage with explicit intra- and cross-graph edges."""

    def __init__(self) -> None:
        self.records: dict[str, CDSRecord] = {}

    def add(self, record: CDSRecord) -> None:
        if record.node_id in self.records:
            raise ValueError(f"duplicate node id: {record.node_id}")
        self.records[record.node_id] = record

    def add_edge(self, parent: str, child: str, *, cross_graph: bool = False) -> None:
        if parent not in self.records or child not in self.records:
            raise KeyError(f"edge references unknown node: {parent}->{child}")
        if cross_graph:
            self.records[parent].cross_graph_links.add(child)
        else:
            self.records[parent].children.add(child)
        self.records[child].parents.add(parent)

    def initialize(self) -> list[CDSRecord]:
        """Reset a run and return the initially-ready nodes."""
        ready: list[CDSRecord] = []
        for record in self.records.values():
            record.ready_counter = len(record.parents)
            record.target_stream = None
            record.state = TaskState.READY if record.ready_counter == 0 else TaskState.NOT_READY
            if record.state is TaskState.READY:
                ready.append(record)
        return ready

    def validate(self) -> None:
        for source in self.records.values():
            if source.memory_bytes < 0 or source.group_memory_bytes < 0:
                raise ValueError("memory estimates must be nonnegative")
            if not 0 <= source.sm_demand <= 1 or not math.isfinite(source.sync_cost) or source.sync_cost < 0:
                raise ValueError("SM demand must be in [0, 1]; sync cost must be nonnegative")
            if source.group_memory_bytes and source.resource_group is None:
                raise ValueError("persistent memory requires a resource group")
            for parent in source.parents:
                if parent not in self.records or source.node_id not in self.records[parent].successors():
                    raise ValueError(f"inconsistent parent {parent}->{source.node_id}")
            for target in source.successors():
                if source.node_id not in self.records[target].parents:
                    raise ValueError(f"inconsistent edge {source.node_id}->{target}")
        # Kahn traversal is also a useful early cycle check.
        counts = {key: len(value.parents) for key, value in self.records.items()}
        todo = [key for key, count in counts.items() if count == 0]
        visited = 0
        while todo:
            key = todo.pop(); visited += 1
            for child in self.records[key].successors():
                counts[child] -= 1
                if counts[child] == 0:
                    todo.append(child)
        if visited != len(self.records):
            raise ValueError("CDS contains a dependency cycle")

    def partition(self, max_nodes: int = 16) -> dict[str, list[str]]:
        """Group contiguous topological runs with the same execution affinity.

        Contracting contiguous topological intervals cannot introduce cycles.
        Partition labels NEVER change the original edges or node readiness.
        Resource groups identify shared state; otherwise graph membership is
        the affinity key. This is a deterministic heuristic, not an optimizer.
        """
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.validate()
        import heapq
        counts = {r.node_id: len(r.parents) for r in self}
        ready = [key for key, count in counts.items() if count == 0]
        heapq.heapify(ready)
        groups: dict[str, list[str]] = {}
        last_key = None
        current = None
        while ready:
            node = heapq.heappop(ready)
            record = self.records[node]
            key = (record.resource_group, record.graph_type)
            if key != last_key or current is None or len(groups[current]) >= max_nodes:
                current = f"subgraph_{len(groups)}"
                groups[current] = []
            groups[current].append(node)
            record.subgraph_id = current
            last_key = key
            for child in sorted(record.successors()):
                counts[child] -= 1
                if counts[child] == 0:
                    heapq.heappush(ready, child)
        return groups

    def __iter__(self) -> Iterable[CDSRecord]:
        return iter(self.records.values())
