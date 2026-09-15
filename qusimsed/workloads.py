from __future__ import annotations

import numpy as np


def synthetic(qubits: int, parameters: int, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.1, parameters), rng.normal(size=(samples, qubits)), rng.integers(0, 2, samples))


def iris(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Iris subset. sklearn remains optional, never silently replaced."""
    try:
        from sklearn.datasets import load_iris
    except ImportError as exc:
        raise RuntimeError("Iris workload requires scikit-learn; install it explicitly") from exc
    data = load_iris(); rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data.data))[:min(samples, len(data.data))]
    features = data.data[indices].astype(float); features = (features - features.mean(0)) / (features.std(0) + 1e-12)
    return features, data.target[indices]
