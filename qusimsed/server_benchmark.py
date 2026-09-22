"""GPU-server entry point for the stream-controlled scheduler.

Example: python -m qusimsed.server_benchmark --mode correctness --qubits 4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import ExperimentConfig
from .profiling import profile_qusimsed
from .result_collection import save_archive
from .pennylane_experiments import (
    _run_scheduled, _workload, baseline_experiment, correctness_experiment, save_result,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['correctness', 'benchmark', 'trace'], default='correctness')
    parser.add_argument('--qubits', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--parameters', type=int, default=None,
                        help='Trainable rotation count; remaining rotation slots keep fixed seeded values')
    parser.add_argument('--reference-backend', choices=['default.qubit', 'lightning.gpu'], default='default.qubit')
    parser.add_argument('--differentiation', choices=['parameter-shift', 'adjoint'], default='parameter-shift')
    parser.add_argument('--streams', type=int, default=4)
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--sm-demand', type=float, default=None,
                        help='Optional manual SM-demand override; omitted means automatic workload/hardware estimation')
    parser.add_argument('--memory-gib', type=float, default=None)
    parser.add_argument('--memory-safety-factor', type=float, default=.8)
    parser.add_argument('--partition-size', type=int, default=16)
    parser.add_argument('--affinity-weight', type=float, default=1.)
    parser.add_argument('--sync-weight', type=float, default=1.)
    parser.add_argument('--parallelism-weight', type=float, default=1.)
    parser.add_argument('--resource-weight', type=float, default=1.)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--strategy', choices=['sequential', 'qusimsed'], default='qusimsed')
    parser.add_argument('--cpu-validation', action='store_true', help='Explicit CPU test mode; never GPU evidence')
    parser.add_argument('--output', default='results/server/result.json')
    parser.add_argument('--no-profile', action='store_true',
                        help='Disable automatic companion Nsight collection after QuSim-Sed')
    parser.add_argument('--profile-output', default=None,
                        help='Nsight output prefix; default: <output-dir>/profiling/<output-stem>/qusimsed')
    args = parser.parse_args()
    config = ExperimentConfig(qubits=args.qubits, layers=args.layers,
                              output_dir=str(Path(args.output).parent),
                              trainable_parameters=args.parameters, reference_backend=args.reference_backend,
                              differentiation=args.differentiation, streams=args.streams,
                              gpu_device=args.gpu_device, sm_demand=args.sm_demand,
                              memory_budget_bytes=None if args.memory_gib is None else int(args.memory_gib * (1 << 30)),
                              memory_safety_factor=args.memory_safety_factor,
                              partition_size=args.partition_size, affinity_weight=args.affinity_weight,
                              sync_weight=args.sync_weight, parallelism_weight=args.parallelism_weight, resource_weight=args.resource_weight, warmup=args.warmup, iterations=args.iterations,
                              seed=args.seed, strategy=args.strategy,
                              execution_backend='torch-cpu' if args.cpu_validation else 'torch-cuda',
                              enable_nsight=not args.no_profile)
    config.validate()
    import torch
    if not args.cpu_validation and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; refusing CPU fallback. Use --cpu-validation only for explicit local checks.')
    environment = {'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda}
    if not args.cpu_validation:
        properties = torch.cuda.get_device_properties(args.gpu_device)
        environment.update(gpu=properties.name, total_memory_bytes=properties.total_memory,
                           sm_count=properties.multi_processor_count, gpu_index=args.gpu_device)
    if args.mode == 'correctness':
        result = correctness_experiment(config)
    elif args.mode == 'benchmark':
        result = baseline_experiment(config)
    else:
        _, parameters, circuit, _ = _workload(config)
        method = 'Sequential' if args.strategy == 'sequential' else 'QuSim-Sed'
        result = _run_scheduled(method, circuit, parameters, config, training=True)
        result['configuration'] = config.metadata()
    result['environment'] = environment
    output = Path(args.output)
    if 'rows' in result:
        save_result(result, output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, default=lambda value: value.tolist()), encoding='utf-8')
    # Keep benchmark timings free of profiler overhead. The companion executes
    # one additional QuSim-Sed step with the same workload/configuration.
    valid = args.mode != 'correctness' or all(row['all_within_tolerance'] for row in result['rows'])
    includes_qusimsed = args.mode != 'trace' or args.strategy == 'qusimsed'
    if config.enable_nsight and includes_qusimsed and valid and os.environ.get('QUSIMSED_UNDER_NSYS') != '1':
        prefix = args.profile_output or output.parent / 'profiling' / output.stem / 'qusimsed'
        if not args.cpu_validation:
            # The child has a separate allocator. Release unused parent cache
            # so the profiling replay has access to the same physical VRAM.
            with torch.cuda.device(args.gpu_device):
                torch.cuda.synchronize(args.gpu_device)
                torch.cuda.empty_cache()
        print(f'Collecting companion QuSim-Sed profile: {prefix}', flush=True)
        result['profiling'] = profile_qusimsed(config, prefix)
        output.write_text(json.dumps(result, indent=2, default=lambda value: value.tolist()), encoding='utf-8')
    if 'collection' in result:
        save_archive(result, result['collection']['directory'])
    print(json.dumps({'output': str(output), 'collection': result.get('collection'), 'backend': result['backend'], 'mode': args.mode,
                      'rows': result.get('rows', []), 'tasks': len(result.get('trace', [])),
                      'profiling': result.get('profiling', {'collected': False, 'reason': 'Profiling disabled, externally managed, or not applicable'})}, indent=2))
    if args.mode == 'correctness' and not all(row['all_within_tolerance'] for row in result['rows']):
        raise SystemExit('Numerical validation failed; do not benchmark this configuration')


if __name__ == '__main__':
    main()
