"""Fair, deterministic PennyLane experiments for the two experiment reviews.

Every strategy receives the same circuit, feature vector, initial parameters,
parameter-shift rule, and learning-rate update.  Only evaluation dispatch
changes.  Results are numerical data, not a synthetic cost model.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ExperimentConfig
from .core.cds import CDSRecord, RecordPool
from .core.scheduler import ResourceAwareScheduler
from .runtime import detect_runtime
from .validation import validate_training

METHODS = ("Sequential", "Batched parameter-shift", "Naive multi-stream", "QuSim-Sed")


def _imports():
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as exc:
        raise RuntimeError("Install PennyLane to run correctness or baseline experiments") from exc
    return qml, pnp


def _device(qml, qubits: int):
    runtime = detect_runtime()
    if runtime.pennylane_lightning_gpu_usable:
        return qml.device("lightning.gpu", wires=qubits), "lightning.gpu"
    return qml.device("default.qubit", wires=qubits), "default.qubit"


def _circuit(qml, device, features, qubits: int, layers: int):
    @qml.qnode(device, interface="autograd", diff_method=None)
    def circuit(theta):
        for wire, feature in enumerate(features): qml.RY(feature, wires=wire)
        index = 0
        for _ in range(layers):
            for wire in range(qubits):
                qml.RX(theta[index], wires=wire); qml.RY(theta[index + 1], wires=wire); qml.RZ(theta[index + 2], wires=wire); index += 3
            for wire in range(qubits): qml.CNOT(wires=[wire, (wire + 1) % qubits])
        return qml.expval(qml.PauliZ(0))
    return circuit


def _shifted(parameters, index: int, direction: float):
    shifted = parameters.copy(); shifted[index] += direction * np.pi / 2
    return shifted


def _evaluate_method(name: str, circuit, parameters, *, streams: int) -> np.ndarray:
    """Return manual parameter-shift gradients with a strategy-specific dispatch."""
    count = len(parameters)
    if name == "Sequential":
        return np.array([(circuit(_shifted(parameters, i, 1)) - circuit(_shifted(parameters, i, -1))) / 2 for i in range(count)])
    if name == "Batched parameter-shift":
        # PennyLane execute batches all independent QuantumScripts in one device call.
        tapes = []
        for i in range(count):
            for direction in (1, -1):
                circuit.construct([_shifted(parameters, i, direction)], {})
                tapes.append(circuit._tape.copy())
        values = circuit.device.execute(tapes)
        return (np.asarray(values[0::2]) - np.asarray(values[1::2])) / 2
    if name == "Naive multi-stream":
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=streams) as workers:
            futures = [workers.submit(lambda i=i: (circuit(_shifted(parameters, i, 1)) - circuit(_shifted(parameters, i, -1))) / 2) for i in range(count)]
            return np.asarray([future.result() for future in futures])
    if name == "QuSim-Sed":
        pool = RecordPool(); results: dict[tuple[int, int], float] = {}
        for i in range(count):
            pool.add(CDSRecord(f"plus_{i}", "circuit", "shift-plus", memory_bytes=1, priority=2))
            pool.add(CDSRecord(f"minus_{i}", "circuit", "shift-minus", memory_bytes=1, priority=2))
            pool.add(CDSRecord(f"grad_{i}", "autograd", "gradient", memory_bytes=1, priority=1))
            pool.add_edge(f"plus_{i}", f"grad_{i}", cross_graph=True); pool.add_edge(f"minus_{i}", f"grad_{i}", cross_graph=True)
        def shifted_task(i: int, direction: int):
            def run(_: int): results[(i, direction)] = float(circuit(_shifted(parameters, i, direction)))
            return run
        tasks: dict[str, Callable] = {}
        for i in range(count):
            tasks[f"plus_{i}"] = shifted_task(i, 1); tasks[f"minus_{i}"] = shifted_task(i, -1)
            tasks[f"grad_{i}"] = lambda _, i=i: results.__setitem__((i, 0), (results[(i, 1)] - results[(i, -1)]) / 2)
        ResourceAwareScheduler(pool, streams, memory_budget_bytes=max(4, streams * 2), safety_factor=1.0).run(tasks)
        return np.asarray([results[(i, 0)] for i in range(count)])
    raise ValueError(f"unknown method: {name}")


def correctness_experiment(config: ExperimentConfig) -> dict:
    qml, pnp = _imports(); rng = np.random.default_rng(config.seed); parameters = config.layers * config.qubits * 3
    initial = pnp.array(rng.normal(0, .1, parameters), requires_grad=False); features = pnp.array(rng.normal(size=config.qubits), requires_grad=False)
    device, backend = _device(qml, config.qubits); circuit = _circuit(qml, device, features, config.qubits, config.layers)
    reference_gradient = _evaluate_method("Sequential", circuit, initial, streams=1); reference_expectation = float(circuit(initial)); reference_loss = (reference_expectation - 1) ** 2
    reference = {"expectation": [reference_expectation], "gradient": reference_gradient, "loss": [reference_loss], "parameters": initial - config.learning_rate * reference_gradient}
    rows = []
    for method in METHODS:
        gradient = _evaluate_method(method, circuit, initial, streams=config.streams); expectation = float(circuit(initial)); loss = (expectation - 1) ** 2
        candidate = {"expectation": [expectation], "gradient": gradient, "loss": [loss], "parameters": initial - config.learning_rate * gradient}
        metrics = validate_training(reference, candidate)
        rows.append({"method": method, **{f"{field}_max_abs": values["max_absolute_error"] for field, values in metrics.items()},
                     **{f"{field}_max_rel": values["max_relative_error"] for field, values in metrics.items()},
                     "all_within_tolerance": all(value["within_tolerance"] for value in metrics.values())})
    return {"backend": backend, "runtime": detect_runtime().metadata(), "configuration": config.metadata(), "rows": rows}


def baseline_experiment(config: ExperimentConfig) -> dict:
    qml, pnp = _imports(); rng = np.random.default_rng(config.seed); parameters = config.layers * config.qubits * 3
    initial = pnp.array(rng.normal(0, .1, parameters), requires_grad=False); features = pnp.array(rng.normal(size=config.qubits), requires_grad=False)
    device, backend = _device(qml, config.qubits); circuit = _circuit(qml, device, features, config.qubits, config.layers)
    rows = []
    for method in METHODS:
        _evaluate_method(method, circuit, initial, streams=config.streams)  # warm-up
        started = time.perf_counter()
        for _ in range(config.iterations): _evaluate_method(method, circuit, initial, streams=config.streams)
        elapsed_ms = (time.perf_counter() - started) * 1000 / config.iterations
        rows.append({"method": method, "mean_gradient_time_ms": elapsed_ms, "backend": backend})
    sequential = next(row["mean_gradient_time_ms"] for row in rows if row["method"] == "Sequential")
    for row in rows: row["speedup_vs_sequential"] = sequential / row["mean_gradient_time_ms"]
    return {"backend": backend, "runtime": detect_runtime().metadata(), "configuration": config.metadata(), "rows": rows}


def save_result(result: dict, path: str | Path) -> Path:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result["rows"][0].keys()); writer.writeheader(); writer.writerows(result["rows"])
    return output
