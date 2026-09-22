"""Timing summaries and ratios; never mix workloads/backends when comparing."""
from __future__ import annotations

import math
import numpy as np


def performance_metrics(baseline: float, candidate: float) -> dict:
    if not all(math.isfinite(value) and value > 0 for value in (baseline, candidate)):
        raise ValueError('timings must be positive and finite')
    return {'speedup_vs_sequential': baseline / candidate,
            'time_saving_percent': 100.0 * (1.0 - candidate / baseline)}


def timing_summary(samples_ms) -> dict:
    values = np.asarray(samples_ms, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('timing samples must be a nonempty vector of positive finite times')
    return {'mean_iteration_time_ms': float(values.mean()),
            'std_iteration_time_ms': float(values.std()),
            'median_iteration_time_ms': float(np.median(values)),
            'min_iteration_time_ms': float(values.min()),
            'max_iteration_time_ms': float(values.max()),
            'p95_iteration_time_ms': float(np.percentile(values, 95)),
            'timed_iterations': len(values), 'iteration_times_ms': values.tolist()}
