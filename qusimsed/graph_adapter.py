"""Extract framework metadata without mutating PennyLane tapes or FX graphs."""
from __future__ import annotations

import numpy as np


def extract_tape(tape, *, max_gate_wires: int = 4) -> dict:
    """Extract operation/wire dependencies and trainable parameter ownership.

    The supported contract is an analytic unitary tape with one expectation
    measurement and scalar, single-parameter trainable gates. Shared/derived
    classical parameters require an external classical Jacobian and are not
    silently treated as independent parameters.
    """
    import pennylane as qml
    if tape.shots.total_shots is not None:
        raise ValueError("finite-shot tapes are not supported")
    if len(tape.measurements) != 1 or not isinstance(tape.measurements[0], qml.measurements.ExpectationMP):
        raise ValueError("one expectation measurement is required")
    wire_order = list(tape.wires)
    wire_index = {wire: i for i, wire in enumerate(wire_order)}
    last, nodes, parameter_owner = {}, [], {}
    parameter_index = 0
    trainable = set(tape.trainable_params)
    for index, op in enumerate(tape.operations):
        if len(op.wires) > max_gate_wires:
            raise ValueError(f"{op.name} exceeds the small-matrix gate limit ({max_gate_wires} wires)")
        if isinstance(op, qml.operation.StatePrepBase) or not op.has_matrix:
            raise ValueError(f"unsupported non-unitary/state-preparation operation: {op.name}")
        if op.batch_size is not None:
            raise ValueError("broadcasted gates are not supported by the stream executor")
        matrix = np.asarray(qml.matrix(op), dtype=np.complex128)
        if not np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-10):
            raise ValueError(f"{op.name} is not unitary")
        wires = tuple(wire_index[w] for w in op.wires)
        parents = sorted({last[w] for w in wires if w in last})
        node = {"index": index, "name": op.name, "wires": wires, "parents": parents,
                "matrix": matrix, "trainable_index": None, "derivative": None}
        for local, _ in enumerate(op.data):
            if parameter_index in trainable:
                if len(op.data) != 1 or np.ndim(op.data[local]) != 0:
                    raise ValueError("trainable operations must have one scalar parameter")
                position = sorted(trainable).index(parameter_index)
                parameter_owner[position] = {"operation": index, "tape_parameter": parameter_index}
                node["trainable_index"] = position
            parameter_index += 1
        for w in wires:
            last[w] = index
        nodes.append(node)
    if len(parameter_owner) != len(trainable):
        raise ValueError("trainable observable parameters are unsupported")
    obs = tape.measurements[0].obs
    if len(obs.wires) > max_gate_wires:
        raise ValueError("observable exceeds small-matrix limit; decompose Hamiltonians before execution")
    observable_matrix = np.asarray(qml.matrix(obs), dtype=np.complex128)
    if not np.allclose(observable_matrix, observable_matrix.conj().T, atol=1e-10):
        raise ValueError("expectation observable must be Hermitian")
    observable = {"matrix": observable_matrix,
                  "wires": tuple(wire_index[w] for w in obs.wires),
                  "parents": sorted(set(last.values()))}
    return {"qubits": len(wire_order), "wires": [str(w) for w in wire_order],
            "operations": nodes, "observable": observable, "parameters": parameter_owner}


def extract_fx(module) -> dict:
    """Read actual FX edges, including repeated inputs only once per edge."""
    return {node.name: {"op": node.op, "target": str(node.target),
                        "parents": [parent.name for parent in node.all_input_nodes]}
            for node in module.graph.nodes}


def trace_shift_reduction(coefficients):
    import torch
    class Reduction(torch.nn.Module):
        def forward(self, values):
            result = values[0] * coefficients[0]
            for i in range(1, len(coefficients)):
                result = result + values[i] * coefficients[i]
            return result
    return torch.fx.symbolic_trace(Reduction())


def trace_training_step(target: float, learning_rate: float):
    import torch
    class Step(torch.nn.Module):
        def forward(self, expectation, gradient, parameters):
            error = expectation - target
            loss = error * error
            loss_gradient = (2 * error) * gradient
            updated = parameters - learning_rate * loss_gradient
            return loss, loss_gradient, updated
    return torch.fx.symbolic_trace(Step())
