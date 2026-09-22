"""Executable Circuit/gradient CDS built from PennyLane tapes and Torch FX.

Parameter-shift recipes come from PennyLane. The adjoint path uncomputes the
forward state during a strictly ordered reverse sweep, retaining O(2**n)
state memory rather than all forward states. Both paths use the same kernels.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import numpy as np

from .core.cds import CDSRecord, RecordPool
from .core.scheduler import ResourceAwareScheduler
from .graph_adapter import extract_fx, extract_tape, trace_shift_reduction, trace_training_step
from .tape_executor import TapeExecutor
from .memory import estimate_statevector_bytes
from .resource_estimation import SMDemandEstimator


class ScheduledVQC:
    def __init__(self, tape, *, differentiation: str = "parameter-shift", streams: int = 4,
                 device: str = "cuda:0", memory_budget_bytes: int | None = None,
                 safety_factor: float = 0.8, sm_demand: float | None = None,
                 partition_size: int = 16, affinity_weight: float = 1.0,
                 sync_weight: float = 1.0, parallelism_weight: float = 1.0,
                 resource_weight: float = 1.0) -> None:
        if differentiation not in ("parameter-shift", "adjoint", "none"):
            raise ValueError("unsupported differentiation method")
        if sm_demand is not None and (not np.isfinite(sm_demand) or not 0 < sm_demand <= 1):
            raise ValueError("sm_demand must be a finite fraction in (0, 1]")
        self.tape, self.differentiation = tape, differentiation
        self.native = extract_tape(tape)
        if not self.native["parameters"] and differentiation != "none":
            raise ValueError("tape has no trainable gate parameters")
        self.executor = TapeExecutor(streams, device=device)
        self.streams, self.sm_demand = streams, sm_demand
        properties = self.executor.torch.cuda.get_device_properties(self.executor.device) if self.executor.cuda else None
        self.demand_estimator = SMDemandEstimator(sm_count=properties.multi_processor_count if properties else None,
                                                override=sm_demand)
        self.demand_estimates = {}
        available = self.executor.allocatable_bytes() if self.executor.cuda else memory_budget_bytes
        if available is None:
            raise ValueError("CPU validation requires memory_budget_bytes")
        self.budget = min(available, memory_budget_bytes) if memory_budget_bytes is not None else available
        self.safety_factor, self.partition_size = safety_factor, partition_size
        self.affinity_weight, self.sync_weight = affinity_weight, sync_weight
        self.parallelism_weight, self.resource_weight = parallelism_weight, resource_weight
        self.gpu_sm_count = properties.multi_processor_count if properties else None
        self.gpu_total_memory_bytes = properties.total_memory if properties else None
        self.pool = RecordPool()
        self.work: dict[str, Callable] = {}
        self.values: dict = {}
        self.states: dict = {}
        self.fx_metadata: dict = {}
        self.circuit_metadata: dict = {}
        self._ran = False
        try:
            self._build()
        except Exception:
            self.executor.close()
            raise

    def _add(self, key, graph, kind, work, parents=(), *, group=None, priority=10,
             memory=0, group_memory=0, metadata=None):
        state_task = graph == "circuit" or bool(group)
        width = len((metadata or {}).get("wires", ())) or max(
            [len(op["wires"]) for op in self.native["operations"]] + [len(self.native["observable"]["wires"])])
        estimate = self.demand_estimator.estimate(self.native["qubits"], state_task=state_task,
                                                  gate_wires=width, reverse=kind == "adjoint-reverse")
        self.demand_estimates[key] = estimate
        r = CDSRecord(key, graph, kind, memory_bytes=memory, priority=priority,
                      sm_demand=estimate["fraction"],
                      resource_group=group, group_memory_bytes=group_memory,
                      metadata={**(metadata or {}), "resource_estimate": estimate})
        self.pool.add(r)
        for parent in parents:
            self.pool.add_edge(parent, key, cross_graph=self.pool.records[parent].graph_type != graph)
        self.work[key] = work
        return key

    def _circuit(self, tape, prefix: str, *, adjoint: bool = False):
        import pennylane as qml
        native = extract_tape(tape)
        self.circuit_metadata[prefix] = {
            "operations": [{k: v for k, v in node.items() if k not in ("matrix", "derivative")}
                           for node in native["operations"]],
            "measurement_parents": native["observable"]["parents"],
            "storage_order": "one state vector per evaluation; sequential gate edges preserve storage hazards",
        }
        n = native["qubits"]
        # Bound state temporaries (permute/reshape, GEMM, adjoint states) plus
        # matrix/allocator slack. This estimate is conservative, not telemetry.
        lease = estimate_statevector_bytes(n, retained_states=12 if adjoint else 8) + (1 << 20)
        initial = f"{prefix}/init"
        self._add(initial, "circuit", "state-init",
                  lambda: self.states.__setitem__(prefix, self.executor.initial_state(n)),
                  group=prefix, group_memory=lease, priority=0)
        previous = initial
        for op in native["operations"]:
            key = f"{prefix}/gate_{op['index']:05d}"
            def gate(op=op):
                self.states[prefix] = self.executor.apply(self.states[prefix], op["matrix"], op["wires"], n)
            # Native wire edges plus the shared-state storage dependency.
            parents = {previous} | {f"{prefix}/gate_{p:05d}" for p in op["parents"]}
            previous = self._add(key, "circuit", op["name"], gate, parents, group=prefix,
                                 group_memory=lease, metadata={"operation_index": op["index"], "wires": op["wires"]})
        measurement = f"{prefix}/measurement"
        obs = native["observable"]
        def measure():
            psi = self.states[prefix]
            observed = self.executor.apply(psi, obs["matrix"], obs["wires"], n)
            self.values[measurement] = self.executor.torch.vdot(psi, observed).real
            if adjoint:
                self.states[prefix + "/lambda"] = observed
            else:
                del self.states[prefix]
        self._add(measurement, "circuit", "expectation", measure, [previous], group=prefix, group_memory=lease)
        if not adjoint:
            return measurement, []
        derivatives = []
        previous = measurement
        for op in reversed(native["operations"]):
            key = f"{prefix}/reverse_{op['index']:05d}"
            derivative = None
            if op["trainable_index"] is not None:
                derivative = np.asarray(qml.operation.operation_derivative(tape.operations[op["index"]]), dtype=np.complex128)
                derivatives.append((op["trainable_index"], key))
            def reverse(op=op, derivative=derivative, key=key):
                psi = self.states[prefix]
                lam = self.states[prefix + "/lambda"]
                self.executor.protect(lam)
                dagger = op["matrix"].conj().T.copy()
                before = self.executor.apply(psi, dagger, op["wires"], n)
                if derivative is not None:
                    dpsi = self.executor.apply(before, derivative, op["wires"], n)
                    self.values[key] = 2 * self.executor.torch.vdot(lam, dpsi).real
                self.states[prefix] = before
                self.states[prefix + "/lambda"] = self.executor.apply(lam, dagger, op["wires"], n)
            previous = self._add(key, "autograd", "adjoint-reverse", reverse, [previous],
                                 group=prefix, group_memory=lease, metadata={"operation_index": op["index"]})
        end = f"{prefix}/release"
        def release():
            del self.states[prefix]
            del self.states[prefix + "/lambda"]
        self._add(end, "autograd", "release-adjoint-state", release, [previous], group=prefix, group_memory=lease)
        return measurement, [key for _, key in sorted(derivatives)]

    def _fx(self, module, prefix: str, bindings: dict[str, str]):
        """Lower traced classical arithmetic using framework-provided edges."""
        from torch.fx.node import map_arg
        self.fx_metadata[prefix] = extract_fx(module)
        mapping = {}
        for node in module.graph.nodes:
            if node.op == "placeholder":
                mapping[node] = bindings[node.name]
                continue
            key = f"{prefix}/{node.name}"
            mapping[node] = key
            parents = [mapping[parent] for parent in node.all_input_nodes]
            def operation(node=node, key=key):
                args = map_arg(node.args, lambda arg: self.values[mapping[arg]])
                kwargs = map_arg(node.kwargs, lambda arg: self.values[mapping[arg]])
                self.executor.protect(args)
                self.executor.protect(kwargs)
                if node.op == "call_function":
                    result = node.target(*args, **kwargs)
                elif node.op == "call_method":
                    result = getattr(args[0], node.target)(*args[1:], **kwargs)
                elif node.op == "output":
                    result = args[0]
                else:
                    raise ValueError(f"unsupported FX operation: {node.op}")
                self.values[key] = result
            self._add(key, "autograd", f"fx:{node.op}", operation, parents, priority=20,
                      metadata={"fx_target": str(node.target)})
        return mapping[next(node for node in module.graph.nodes if node.op == "output")]

    def _build(self):
        import pennylane as qml
        if self.differentiation == "none":
            self.expectation_node, _ = self._circuit(self.tape, "forward")
            return
        if self.differentiation == "adjoint":
            self.expectation_node, gradients = self._circuit(self.tape, "forward", adjoint=True)
        else:
            self.expectation_node, _ = self._circuit(self.tape, "forward")
            gradients = []
            original = self.tape.get_parameters(trainable_only=False)
            for position, owner in sorted(self.native["parameters"].items()):
                op = self.tape.operations[owner["operation"]]
                recipe = op.grad_recipe[0] if op.grad_recipe and op.grad_recipe[0] is not None else None
                if recipe is None:
                    rule = qml.gradients.generate_shift_rule(op.parameter_frequencies[0])
                    recipe = [(float(c), 1.0, float(s)) for c, s in rule]
                measurements, coefficients = [], []
                for j, (coefficient, multiplier, shift) in enumerate(recipe):
                    parameter = owner["tape_parameter"]
                    shifted = self.tape.bind_new_parameters([multiplier * original[parameter] + shift], [parameter])
                    measured, _ = self._circuit(shifted, f"shift_{position:05d}_{j:03d}")
                    measurements.append(measured)
                    coefficients.append(float(coefficient))
                packed = f"derivative_{position:05d}/inputs"
                def pack(packed=packed, measurements=measurements):
                    self.values[packed] = tuple(self.values[k] for k in measurements)
                self._add(packed, "autograd", "shift-results", pack, measurements, priority=20)
                gradients.append(self._fx(trace_shift_reduction(coefficients), f"derivative_{position:05d}", {"values": packed}))
        self.gradient_node = "gradient/stack"
        def stack():
            values = [self.values[k] for k in gradients]
            self.executor.protect(values)
            self.values[self.gradient_node] = self.executor.torch.stack(values)
        self._add(self.gradient_node, "autograd", "gradient-stack", stack, gradients, priority=20)

    def add_training_step(self, *, target: float = 1.0, learning_rate: float = 0.05):
        """MSE loss, its true gradient, and SGD update from an extracted FX graph.

        Parameters refer to independent trainable tape entries, not arbitrary
        QNode arguments with an unprovided classical parameter transform.
        """
        if self._ran or hasattr(self, "training_node"):
            raise ValueError("add one training step before execution")
        if self.differentiation == "none":
            raise ValueError("training requires a differentiation method")
        values = np.asarray(self.tape.get_parameters(), dtype=float)
        key = "parameters/input"
        self._add(key, "autograd", "parameters", lambda: self.values.__setitem__(key, self.executor.tensor(values)), priority=20)
        self.training_node = self._fx(trace_training_step(target, learning_rate), "training",
                                      {"expectation": self.expectation_node, "gradient": self.gradient_node, "parameters": key})
        return self

    def run(self) -> dict:
        if self._ran:
            raise ValueError("execution plans are single-use; build a new plan for new parameters")
        self._ran = True
        # Small outputs/FX intermediates survive tasks. Reserve them separately
        # from state leases, including allocator granularity and vector outputs.
        baseline = (16 << 20) + len(self.pool.records) * 1024 + len(self.native["parameters"]) * 128
        if self.executor.cuda:
            self.executor.torch.cuda.reset_peak_memory_stats(self.executor.device)
        try:
            scheduler = ResourceAwareScheduler(
                self.pool, self.streams, self.budget, self.safety_factor,
                baseline_memory_bytes=baseline, partition_size=self.partition_size,
                affinity_weight=self.affinity_weight, sync_weight=self.sync_weight,
                memory_available=self.executor.allocatable_bytes if self.executor.cuda else None,
                memory_allocated=(lambda: self.executor.torch.cuda.memory_allocated(self.executor.device)) if self.executor.cuda else None,
                gpu_sm_count=self.gpu_sm_count, parallelism_weight=self.parallelism_weight,
                resource_weight=self.resource_weight)
            tasks = {r.node_id: self.executor.wrap(r.node_id, r.parents, self.work[r.node_id]) for r in self.pool}
            trace = scheduler.run(tasks)
            def host(value):
                if self.executor.torch.is_tensor(value):
                    return value.detach().cpu().numpy().copy()
                return tuple(host(item) for item in value)
            result = {"backend": self.executor.backend, "differentiation": self.differentiation,
                      "expectation": float(host(self.values[self.expectation_node])),
                      "trace": [asdict(row) for row in trace],
                      "partitions": scheduler.partitions,
                      "peak_reserved_bytes": scheduler.peak_reserved_bytes,
                      "peak_estimated_sm_demand": scheduler.peak_sm_demand,
                      "memory_budget_bytes": self.budget, "memory_safety_factor": self.safety_factor,
                      "gpu_total_memory_bytes": self.gpu_total_memory_bytes, "gpu_sm_count": self.gpu_sm_count,
                      "parallelism_weight": self.parallelism_weight, "resource_weight": self.resource_weight,
                      "sm_demand_per_quantum_task": self.sm_demand,
                      "sm_demand_mode": "automatic" if self.sm_demand is None else "manual",
                      "resource_estimates": self.demand_estimates,
                      "cross_stream_event_waits": self.executor.event_waits,
                      "observed_torch_peak_allocated_bytes": (
                          int(self.executor.torch.cuda.max_memory_allocated(self.executor.device))
                          if self.executor.cuda else None),
                      "circuit_metadata": self.circuit_metadata, "classical_graph_metadata": self.fx_metadata}
            if self.differentiation != "none":
                result["gradient"] = host(self.values[self.gradient_node])
            if hasattr(self, "training_node"):
                loss, gradient, updated = host(self.values[self.training_node])
                result.update(loss=float(loss), loss_gradient=gradient, parameters=updated)
            return result
        finally:
            self.executor.close()
            self.states.clear()
            self.values.clear()
