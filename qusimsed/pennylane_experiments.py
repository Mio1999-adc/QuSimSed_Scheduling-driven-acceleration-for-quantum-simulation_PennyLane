"""Reproducible VQC validation and timing with explicit backend identities.

The stream-controlled backend supports Sequential and QuSim-Sed on identical
Torch kernels. Legacy PennyLane batching is kept as a separate backend; it is
not relabelled CUDA-stream scheduling and is not mixed into same-backend speedups.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .runtime import detect_runtime
from .validation import validate_training
from .metrics import performance_metrics, timing_summary
from .result_collection import collect_results

# Retained for discovery/older clients; methods_for() is the executable matrix.
METHODS = ("Sequential", "Batched parameter-shift", "Naive multi-stream", "QuSim-Sed")


def methods_for(config):
    if config.execution_backend.startswith("torch-"):
        return ("Sequential", "QuSim-Sed")
    return (("Sequential", "Batched parameter-shift") if config.differentiation == "parameter-shift"
            else ("Sequential",))


def _imports():
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ImportError as exc:
        raise RuntimeError("Install PennyLane to run VQC experiments") from exc
    return qml, pnp


def _device(qml, qubits: int, config=None):
    if config is not None and config.execution_backend.startswith("torch-"):
        if config.execution_backend == "torch-cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Torch CUDA is unavailable; refusing CPU fallback")
        # This device constructs tapes only. All benchmark kernels run through
        # ScheduledVQC; it is never used as a silent execution fallback.
        return qml.device("default.qubit", wires=qubits), (
            "torch-cuda-statevector" if config.execution_backend == "torch-cuda"
            else "torch-cpu-statevector-validation")
    return qml.device("lightning.gpu", wires=qubits), "lightning.gpu"


def _circuit(qml, device, features, qubits: int, layers: int, differentiation="parameter-shift", *, fixed_parameters=None):
    @qml.qnode(device, interface="autograd", diff_method=differentiation)
    def circuit(theta):
        for wire, feature in enumerate(features):
            qml.RY(feature, wires=wire)
        index = 0
        for _ in range(layers):
            for wire in range(qubits):
                angles = [theta[i] if i < len(theta) else float(fixed_parameters[i])
                          for i in range(index, index + 3)]
                qml.RX(angles[0], wires=wire)
                qml.RY(angles[1], wires=wire)
                qml.RZ(angles[2], wires=wire)
                index += 3
            if qubits > 1:
                for wire in range(qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % qubits])
        return qml.expval(qml.PauliZ(0))
    return circuit


def circuit_for_config(qml, device, features, config):
    # Freeze a suffix of the SAME rotation slots for independent parameter
    # scaling: gate count, depth, features, and initial angles remain fixed.
    fixed = np.random.default_rng(config.seed).normal(0, .1, 3 * config.qubits * config.layers)
    return _circuit(qml, device, features, config.qubits, config.layers,
                    config.differentiation, fixed_parameters=fixed)


def _tape(circuit, parameters):
    _, pnp = _imports()
    theta = pnp.array(parameters, requires_grad=True)
    tape = circuit.construct([theta], {})
    # Compatibility with PennyLane versions whose construct() returns None.
    return tape if tape is not None else circuit._tape


def _run_scheduled(name, circuit, parameters, config, *, training=False, forward_only=False):
    from .scheduled_vqc import ScheduledVQC
    if name not in ("Sequential", "QuSim-Sed"):
        raise ValueError(f"{name} has no implementation for the stream-owned backend")
    config.validate()
    device = f"cuda:{config.gpu_device}" if config.execution_backend == "torch-cuda" else "cpu"
    budget = config.memory_budget_bytes
    if device == "cpu" and budget is None:
        budget = 1 << 30  # explicit validation ceiling, never GPU telemetry
    plan = ScheduledVQC(_tape(circuit, parameters),
                        differentiation="none" if forward_only else config.differentiation,
                        streams=1 if name == "Sequential" else config.streams,
                        device=device, memory_budget_bytes=budget,
                        safety_factor=config.memory_safety_factor, sm_demand=config.sm_demand,
                        partition_size=config.partition_size, affinity_weight=config.affinity_weight,
                        sync_weight=config.sync_weight, parallelism_weight=config.parallelism_weight,
                        resource_weight=config.resource_weight)
    if training:
        plan.add_training_step(learning_rate=config.learning_rate)
    return plan.run()


def _evaluate_method(name, circuit, parameters, *, streams: int, config=None):
    config = config or ExperimentConfig(streams=streams)
    if config.execution_backend.startswith("torch-"):
        return _run_scheduled(name, circuit, parameters, config)["gradient"]
    qml, pnp = _imports()
    theta = pnp.array(parameters, requires_grad=True)
    if name == "Sequential":
        return np.asarray(qml.grad(circuit)(theta))
    if name == "Batched parameter-shift" and config.differentiation == "parameter-shift":
        tapes, processing = qml.gradients.param_shift(_tape(circuit, theta))
        values = qml.execute(tapes, circuit.device, diff_method=None, interface=None)
        return np.asarray(processing(values))
    raise ValueError("Legacy PennyLane execution has no verified multi-stream/CDS adapter; select torch-cuda")


def evaluate_expectation(circuit, parameters, config):
    if config.execution_backend.startswith("torch-"):
        return _run_scheduled("Sequential", circuit, parameters, config, forward_only=True)["expectation"]
    return float(circuit(parameters))


def _workload(config):
    config.validate()
    qml, pnp = _imports()
    rng = np.random.default_rng(config.seed)
    full_initial = rng.normal(0, .1, config.layers * config.qubits * 3)
    initial = pnp.array(full_initial[:config.parameter_count], requires_grad=True)
    features = pnp.array(rng.normal(size=config.qubits), requires_grad=False)
    device, backend = _device(qml, config.qubits, config)
    circuit = circuit_for_config(qml, device, features, config)
    return qml, initial, circuit, backend


@collect_results('correctness')
def correctness_experiment(config: ExperimentConfig) -> dict:
    qml, initial, circuit, backend = _workload(config)
    # Independent framework reference, outside measured timings. The suite
    # defaults to Lightning-GPU on a GPU server; explicit CPU validation uses
    # default.qubit. Never silently fall back when the requested reference fails.
    reference_circuit = qml.QNode(circuit.func,
                                  qml.device(config.reference_backend, wires=config.qubits),
                                  interface="autograd", diff_method=config.differentiation)
    expectation = float(reference_circuit(initial))
    derivative = np.asarray(qml.grad(reference_circuit)(initial))
    loss_gradient = 2 * (expectation - 1) * derivative
    reference = {"expectation": [expectation], "gradient": loss_gradient,
                 "loss": [(expectation - 1) ** 2],
                 "parameters": initial - config.learning_rate * loss_gradient}
    rows, traces, resource_estimates = [], {}, {}
    for method in methods_for(config):
        if config.execution_backend.startswith("torch-"):
            result = _run_scheduled(method, circuit, initial, config, training=True)
            resource_estimates[method] = result["resource_estimates"]
            candidate = {"expectation": [result["expectation"]], "gradient": result["loss_gradient"],
                         "loss": [result["loss"]], "parameters": result["parameters"]}
            if config.enable_trace:
                traces[method] = result["trace"]
        else:
            candidate_expectation = float(circuit(initial))
            gradient = 2 * (candidate_expectation - 1) * _evaluate_method(method, circuit, initial, streams=config.streams, config=config)
            candidate = {"expectation": [candidate_expectation], "gradient": gradient,
                         "loss": [(candidate_expectation - 1) ** 2], "parameters": initial - config.learning_rate * gradient}
        metrics = validate_training(reference, candidate)
        rows.append({"method": method,
                     **{f"{field}_max_abs": value["max_absolute_error"] for field, value in metrics.items()},
                     **{f"{field}_max_rel": value["max_relative_error"] for field, value in metrics.items()},
                     **{f"{field}_mean_abs": value["mean_absolute_error"] for field, value in metrics.items()},
                     **{f"{field}_rmse": value["rmse"] for field, value in metrics.items()},
                     "all_within_tolerance": all(value["within_tolerance"] for value in metrics.values())})
    return {"backend": backend, "reference_backend": reference_circuit.device.name,
            "runtime": detect_runtime().metadata(), "configuration": config.metadata(), "rows": rows, "traces": traces,
            "resource_estimates": resource_estimates}


@collect_results('benchmark')
def baseline_experiment(config: ExperimentConfig) -> dict:
    _, initial, circuit, backend = _workload(config)
    rows, traces, resource_estimates = [], {}, {}
    for method in methods_for(config):
        def iteration():
            if config.execution_backend.startswith("torch-"):
                return _run_scheduled(method, circuit, initial, config, training=True)
            expectation = float(circuit(initial))
            gradient = _evaluate_method(method, circuit, initial, streams=config.streams, config=config)
            _ = initial - config.learning_rate * 2 * (expectation - 1) * gradient
            return None
        for _ in range(config.warmup):
            iteration()
        timings = []
        for _ in range(config.iterations):
            started = time.perf_counter()
            result = iteration()
            timings.append((time.perf_counter() - started) * 1000)
        if result is not None and config.enable_trace:
            traces[method] = result["trace"]
        if result is not None:
            resource_estimates[method] = result["resource_estimates"]
        rows.append({"method": method, **timing_summary(timings), "backend": backend})
    sequential = rows[0]["mean_iteration_time_ms"]
    for row in rows:
        row.update(performance_metrics(sequential, row["mean_iteration_time_ms"]))
    return {"backend": backend, "runtime": detect_runtime().metadata(), "configuration": config.metadata(),
            "resource_estimates": resource_estimates,
            "timing_scope": "end-to-end: tape/graph construction, scheduler, forward, differentiation, MSE/SGD, device completion",
            "rows": rows, "traces": traces}


def save_result(result: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result["rows"][0].keys())
        writer.writeheader()
        writer.writerows(result["rows"])
    return output
