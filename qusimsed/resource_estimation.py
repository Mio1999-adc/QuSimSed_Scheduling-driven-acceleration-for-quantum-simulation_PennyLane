"""Static scheduling admission estimates, not CUDA occupancy measurements.

Torch/cuBLAS choose their own launches. Work units below model state-vector
work; they are deliberately not presented as actual CUDA block counts.
"""
import math


class SMDemandEstimator:
    def __init__(self, *, sm_count=None, override=None):
        if override is not None and (not math.isfinite(override) or not 0 < override <= 1):
            raise ValueError('sm_demand must be None (automatic) or a finite fraction in (0, 1]')
        if sm_count is not None and sm_count < 1:
            raise ValueError('SM count must be positive')
        self.sm_count, self.override = sm_count, override

    def estimate(self, qubits, *, state_task=True, gate_wires=1, reverse=False):
        if self.override is not None:
            demand = self.override if state_task else min(.05, self.override)
            source = 'manual-override'
        elif self.sm_count is None:
            demand, source = 1.0, 'conservative-no-gpu-properties'
        else:
            # One model unit per 256 complex amplitudes; wide matrices and
            # reverse passes increase work. One unit per SM saturates this
            # conservative admission model. Large state sweeps remain serial
            # to avoid assuming that bandwidth-intensive kernels overlap well.
            work = (1 << qubits) * (1 << max(0, gate_wires - 1)) if state_task else 1
            if reverse:
                work *= 2
            units = max(1, math.ceil(work / 256))
            demand = min(1.0, units / self.sm_count)
            source = 'static-statevector-work-model-v1'
        return {'fraction': demand, 'source': source, 'sm_count': self.sm_count,
                'measured_occupancy': False, 'qubits': qubits,
                'gate_wires': gate_wires, 'state_task': state_task, 'reverse': reverse}
