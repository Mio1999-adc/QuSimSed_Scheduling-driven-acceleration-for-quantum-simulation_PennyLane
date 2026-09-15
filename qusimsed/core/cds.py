from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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

    def __iter__(self) -> Iterable[CDSRecord]:
        return iter(self.records.values())
