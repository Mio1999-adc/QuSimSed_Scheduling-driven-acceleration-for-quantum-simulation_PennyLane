"""GPU-only validation of the actual CDS execution path against PennyLane."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .pennylane_experiments import correctness_experiment


def run(qubits=4, layers=2, seed=7, learning_rate=.05, differentiation='parameter-shift', streams=4):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU correctness requires Torch CUDA; refusing CPU fallback')
    return correctness_experiment(ExperimentConfig(
        qubits=qubits, layers=layers, seed=seed, learning_rate=learning_rate,
        differentiation=differentiation, streams=streams, execution_backend='torch-cuda'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubits', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--streams', type=int, default=4)
    parser.add_argument('--differentiation', choices=['parameter-shift', 'adjoint'], default='parameter-shift')
    parser.add_argument('--output', default='results/gpu-correctness.json')
    args = parser.parse_args()
    result = run(args.qubits, args.layers, args.seed, differentiation=args.differentiation, streams=args.streams)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['rows'], indent=2))
    if not all(row['all_within_tolerance'] for row in result['rows']):
        raise SystemExit('GPU correctness failed')


if __name__ == '__main__':
    main()
