from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ErrorMetrics:
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    rmse: float
    within_tolerance: bool


def compare(reference: Any, candidate: Any, *, atol: float = 1e-7,
            rtol: float = 1e-6, epsilon: float = 1e-12) -> ErrorMetrics:
    """Numerical comparison suitable for changed floating-point order."""
    ref, test = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if ref.shape != test.shape:
        raise ValueError(f"shape mismatch: {ref.shape} != {test.shape}")
    absolute = np.abs(ref - test)
    relative = absolute / np.maximum(np.abs(ref), epsilon)
    return ErrorMetrics(float(absolute.max(initial=0)), float(absolute.mean()),
                        float(relative.max(initial=0)), float(np.sqrt(np.mean((ref - test) ** 2))),
                        bool(np.allclose(ref, test, atol=atol, rtol=rtol)))


def validate_training(reference: Mapping[str, Any], candidate: Mapping[str, Any], **kwargs: Any) -> dict[str, dict]:
    required = {"expectation", "gradient", "loss", "parameters"}
    missing = required - set(reference) | required - set(candidate)
    if missing:
        raise KeyError(f"correctness payload missing: {sorted(missing)}")
    fields = list(required) + (["trajectory"] if "trajectory" in reference and "trajectory" in candidate else [])
    return {field: asdict(compare(reference[field], candidate[field], **kwargs)) for field in fields}
