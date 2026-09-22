"""Reproducible application workloads for QuSim-Sed evaluation.

The tabular datasets are fast QML smoke/application benchmarks. ``mnist-pca``
uses a binary 3-vs-5 MNIST task and fits PCA only on the training split; this
keeps the image benchmark practical without data leakage.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .pennylane_experiments import methods_for, evaluate_expectation, circuit_for_config, _device, _evaluate_method, _imports
from .runtime import detect_runtime
from .validation import compare
from .metrics import performance_metrics
from .result_collection import collect_results


DATASET_DESCRIPTIONS = {
    "iris": "Iris binary classification: class 0 versus class 1.",
    "wine": "Wine binary classification: class 0 versus class 1.",
    "breast-cancer": "Wisconsin diagnostic breast-cancer classification.",
    "mnist-pca": "MNIST handwritten-digit classification: digit 3 versus digit 5, PCA-reduced after split.",
}


def load_workload(name: str, *, qubits: int, samples: int, seed: int):
    try:
        from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
        from sklearn.decomposition import PCA
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Real benchmarks require scikit-learn") from exc
    if name == "iris":
        data = load_iris(); mask = data.target < 2; features, labels = data.data[mask], data.target[mask]
    elif name == "wine":
        data = load_wine(); mask = data.target < 2; features, labels = data.data[mask], data.target[mask]
    elif name == "breast-cancer":
        data = load_breast_cancer(); features, labels = data.data, data.target.astype(int)
    elif name == "mnist-pca":
        # OpenML caches the download. Exact version avoids a moving dataset target.
        data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        raw_labels = np.asarray(data.target, dtype=str); mask = np.isin(raw_labels, ["3", "5"])
        features, labels = np.asarray(data.data[mask], dtype=float), (raw_labels[mask] == "5").astype(int)
    else:
        raise ValueError(f"unknown workload: {name}")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2, random_state=seed, stratify=labels)
    rng = np.random.default_rng(seed)
    if samples < len(x_train):
        selected = rng.permutation(len(x_train))[:samples]
        x_train, y_train = x_train[selected], y_train[selected]
    test_cap = min(len(x_test), max(20, samples // 4)); x_test, y_test = x_test[:test_cap], y_test[:test_cap]
    scaler = StandardScaler().fit(x_train); x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    dimension = min(qubits, x_train.shape[1], len(x_train))
    projection = PCA(n_components=dimension, random_state=seed).fit(x_train); x_train, x_test = projection.transform(x_train), projection.transform(x_test)
    # Angle encoding requires one number per qubit. Zero-pad only when PCA has fewer dimensions.
    x_train = np.pad(x_train, ((0, 0), (0, qubits - dimension))); x_test = np.pad(x_test, ((0, 0), (0, qubits - dimension)))
    scale = max(np.max(np.abs(x_train)), 1e-12); return x_train / scale * np.pi, x_test / scale * np.pi, y_train, y_test


def _prediction(circuit, theta, config) -> float:
    return (evaluate_expectation(circuit, theta, config) + 1) / 2


def _run_method(method: str, qml, device, config: ExperimentConfig, x_train, y_train, x_test, y_test, initial):
    theta = initial.copy(); history = []; epoch_times = []; started = time.perf_counter()
    for _ in range(config.iterations):
        epoch_started = time.perf_counter()
        total = np.zeros_like(theta)
        for features, label in zip(x_train, y_train):
            circuit = circuit_for_config(qml, device, features, config)
            prediction = _prediction(circuit, theta, config)
            # d((f+1)/2-y)^2/dtheta = (prediction-y) * df/dtheta.
            total += (prediction - label) * _evaluate_method(method, circuit, theta, streams=config.streams, config=config)
        theta = theta - config.learning_rate * total / len(x_train)
        train_predictions = np.array([_prediction(circuit_for_config(qml, device, x, config), theta, config) for x in x_train])
        history.append(float(np.mean((train_predictions - y_train) ** 2)))
        epoch_times.append(time.perf_counter() - epoch_started)
    elapsed = time.perf_counter() - started
    test_predictions = np.array([_prediction(circuit_for_config(qml, device, x, config), theta, config) for x in x_test])
    return {"parameters": theta, "trajectory": history, "epoch_times_seconds": epoch_times, "final_loss": float(np.mean((test_predictions - y_test) ** 2)),
            "accuracy": float(np.mean((test_predictions >= .5) == y_test)), "training_seconds": elapsed}


@collect_results('training')
def run_real_benchmark(config: ExperimentConfig) -> dict:
    config.validate()
    if config.workload not in DATASET_DESCRIPTIONS: raise ValueError("choose iris, wine, breast-cancer, or mnist-pca")
    qml, pnp = _imports(); x_train, x_test, y_train, y_test = load_workload(config.workload, qubits=config.qubits, samples=config.samples, seed=config.seed)
    device, backend = _device(qml, config.qubits, config); rng = np.random.default_rng(config.seed); initial = pnp.array(rng.normal(0, .1, config.layers * config.qubits * 3)[:config.parameter_count], requires_grad=False)
    outcomes = {method: _run_method(method, qml, device, config, x_train, y_train, x_test, y_test, initial) for method in methods_for(config)}
    reference = outcomes["Sequential"]; rows = []
    for method, outcome in outcomes.items():
        trajectory = compare(reference["trajectory"], outcome["trajectory"])
        parameters = compare(reference["parameters"], outcome["parameters"])
        rows.append({"method": method, "training_seconds": outcome["training_seconds"], "test_loss": outcome["final_loss"], "test_accuracy": outcome["accuracy"],
                     "trajectory_max_abs_error": trajectory.max_absolute_error, "parameters_max_abs_error": parameters.max_absolute_error,
                     "correctness_within_tolerance": trajectory.within_tolerance and parameters.within_tolerance})
    sequential = rows[0]["training_seconds"]
    for row in rows: row.update(performance_metrics(sequential, row["training_seconds"]))
    return {"dataset": config.workload, "dataset_description": DATASET_DESCRIPTIONS[config.workload], "backend": backend, "runtime": detect_runtime().metadata(),
            "samples": {"train": len(x_train), "test": len(x_test), "features_after_pca": config.qubits}, "configuration": config.metadata(), "rows": rows,
            "timing_scope": "training loop including per-epoch training-loss evaluation; excludes preprocessing and final test prediction",
            "trajectories": {method: outcome["trajectory"] for method, outcome in outcomes.items()},
            "epoch_times_seconds": {method: outcome["epoch_times_seconds"] for method, outcome in outcomes.items()},
            "final_parameters": {method: np.asarray(outcome["parameters"]).tolist() for method, outcome in outcomes.items()}}


def save_real_result(result: dict, output: str | Path) -> Path:
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result["rows"][0].keys()); writer.writeheader(); writer.writerows(result["rows"])
    return path
