"""Collect reproducible correctness, scaling, timing and training experiments.

GPU example: python -m qusimsed.experiment_suite --output results/experiments
Use --plan-only to inspect the matrix without executing it. Unsupported method
rows are explicit and never receive invented timings, speedups or errors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .metrics import performance_metrics
from .pennylane_experiments import baseline_experiment, correctness_experiment, methods_for
from .profiling import profile_qusimsed
from .real_benchmarks import run_real_benchmark

METHOD_NAMES = {
    'sequential': 'Sequential', 'qusimsed': 'QuSim-Sed',
    'gradient-only': 'Gradient-only', 'quantum-only': 'Quantum-only',
    'batched-parameter-shift': 'Batched parameter-shift',
    'naive-multistream': 'Naive multi-stream', 'catalyst': 'Catalyst',
    'qusimsed-catalyst': 'QuSim-Sed + Catalyst',
}
SCHEMA_VERSION = 1


def unsupported_reason(key, config):
    if key in ('catalyst', 'qusimsed-catalyst'):
        return 'Catalyst is not integrated with the maintained CDS executor; historical proxy results are not imported'
    if key == 'batched-parameter-shift':
        return 'Batched parameter-shift is available only in the separate pennylane backend with parameter-shift differentiation'
    return f'{METHOD_NAMES[key]} is not implemented for {config.execution_backend}/{config.differentiation}'


def capabilities(config, requested):
    actual = set(methods_for(config))
    return [{'method_key': key, 'method': METHOD_NAMES[key],
             'supported': METHOD_NAMES[key] in actual,
             'reason': None if METHOD_NAMES[key] in actual else unsupported_reason(key, config)}
            for key in requested]


def build_cases(base, *, axes, qubits, depths, fixed_qubits, fixed_layers,
                parameters, parameter_qubits, parameter_layers,
                differentiations, seeds, workloads, epochs):
    """Deduplicate overlapping axes while preserving their labels."""
    points = []
    if 'qubits' in axes:
        points += [('qubits', n, fixed_layers, None) for n in qubits]
    if 'depth' in axes:
        points += [('depth', fixed_qubits, depth, None) for depth in depths]
    if 'parameters' in axes:
        points += [('parameters', parameter_qubits, parameter_layers, count) for count in parameters]
    cases = {}
    for axis, n, depth, count in points:
        for diff in differentiations:
            for seed in seeds:
                for workload in workloads:
                    config = replace(base, qubits=n, layers=depth, trainable_parameters=count,
                                     differentiation=diff, seed=seed, workload=workload,
                                     iterations=base.iterations if workload == 'synthetic' else epochs)
                    config.validate()
                    # None and explicitly full trainability represent the same workload.
                    config = replace(config, trainable_parameters=config.parameter_count)
                    canonical = json.dumps(config.metadata(), sort_keys=True, separators=(',', ':'))
                    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
                    identity = f'{workload}-{n}q-{depth}l-{config.parameter_count}p-{diff}-s{seed}-{digest}'
                    if identity not in cases:
                        cases[identity] = {'case_id': identity, 'axes': [], 'configuration': config.metadata()}
                    if axis not in cases[identity]['axes']:
                        cases[identity]['axes'].append(axis)
    if not cases:
        raise ValueError('experiment matrix must not be empty')
    return list(cases.values())


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'tolist'):
        return _json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns or ['status'])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(_json_safe(value)) if isinstance(value, (dict, list, tuple)) else _json_safe(value)
                         for key, value in row.items()})
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(buffer.getvalue(), encoding='utf-8')
    temporary.replace(path)


def environment_snapshot(*, probe_gpu=True):
    versions = {}
    for name in ['numpy', 'pennylane', 'pennylane-lightning-gpu', 'torch', 'jax', 'pennylane-catalyst', 'scikit-learn']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {'python': sys.version, 'platform': platform.platform(), 'packages': versions,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    if probe_gpu:
        try:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                    '--format=csv,noheader'], capture_output=True, text=True, timeout=10, check=True)
            result['nvidia_smi'] = probe.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            result['nvidia_smi_error'] = str(exc)
    return result


def require_execution_backend(config):
    if config.execution_backend == 'torch-cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --cpu-validation explicitly for local checks')
        torch.cuda.get_device_properties(config.gpu_device)


def execute_case(case, directory, requested, environment):
    directory = Path(directory)
    # This runner already owns per-case persistence; avoid a second archive.
    config = replace(ExperimentConfig(**case['configuration']), collect_results=False)
    catalog = capabilities(config, requested)
    result = {**case, 'capabilities': catalog, 'status': 'running', 'environment': environment,
              'started_at_unix': time.time(), 'validation_workload': 'seeded synthetic VQC with the same ansatz and trainable mask'}
    write_json(directory / 'case.json', result)
    if not any(row['supported'] for row in catalog):
        result['status'] = 'unsupported'
        result['finished_at_unix'] = time.time()
        write_json(directory / 'case.json', result)
        return result
    try:
        # Dataset loading and GPU reference work stay outside timed candidate runs.
        validation = correctness_experiment(config)
        write_json(directory / 'correctness.json', validation)
        result['correctness'] = validation
        if not all(row['all_within_tolerance'] for row in validation['rows']):
            result['status'] = 'validation_failed'
            result['reason'] = 'Independent numerical validation failed; performance collection was not run'
        else:
            if config.workload == 'synthetic':
                performance = baseline_experiment(config)
                artifact = 'benchmark.json'
            else:
                performance = run_real_benchmark(config)
                artifact = 'training.json'
            write_json(directory / artifact, performance)
            result['performance'] = performance
            training_valid = all(row.get('correctness_within_tolerance', True) for row in performance['rows'])
            result['status'] = 'completed' if training_valid else 'validation_failed'
            if not training_valid:
                result['reason'] = 'Training trajectory or final parameters diverged; aggregate speedup comparisons are suppressed'
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    result['finished_at_unix'] = time.time()
    # Persist costly measurements before starting the additional profiling process.
    write_json(directory / 'case.json', result)
    if result['status'] == 'completed' and config.enable_nsight and any(row['method'] == 'QuSim-Sed' and row['supported'] for row in catalog):
        try:
            if config.execution_backend == 'torch-cuda':
                import torch
                with torch.cuda.device(config.gpu_device):
                    torch.cuda.synchronize(config.gpu_device)
                    torch.cuda.empty_cache()
            result['profiling'] = profile_qusimsed(config, directory / 'profiling' / 'qusimsed')
        except Exception as exc:
            result['profiling'] = {'collected': False, 'reason': f'{type(exc).__name__}: {exc}'}
            write_json(directory / 'profiling' / 'qusimsed.nsight.json', result['profiling'])
    else:
        result['profiling'] = {'collected': False, 'reason': 'Disabled, validation unsuccessful, or QuSim-Sed not requested'}
    result['collection_finished_at_unix'] = time.time()
    write_json(directory / 'case.json', result)
    return result


def export_tables(root, plan):
    """Compare only matching case IDs, timing scopes and backend identities."""
    root = Path(root)
    rows, errors, samples, convergence, comparisons = [], [], [], [], []
    for case in plan:
        path = root / 'cases' / case['case_id'] / 'case.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        config = ExperimentConfig(**case['configuration'])
        base = {'case_id': case['case_id'], 'axes': '|'.join(case['axes']), 'workload': config.workload,
                'qubits': config.qubits, 'layers': config.layers, 'parameters': config.parameter_count,
                'rotation_slots': 3 * config.qubits * config.layers, 'differentiation': config.differentiation,
                'seed': config.seed, 'execution_backend': config.execution_backend,
                'streams': config.streams, 'sm_demand': config.sm_demand,
                'reference_backend': config.reference_backend,
                'timing_kind': 'repeated_training_step' if config.workload == 'synthetic' else 'multi_epoch_training'}
        correct = {row['method']: row for row in result.get('correctness', {}).get('rows', [])}
        perf = result.get('performance', {})
        measured = {row['method']: row for row in perf.get('rows', [])}
        accepted = []
        for method in result['capabilities']:
            name = method['method']
            row = {**base, 'method': name, 'method_key': method['method_key']}
            if not method['supported']:
                row.update(status='unsupported', reason=method['reason'])
            else:
                row.update(status=result['status'], reason=result.get('reason'),
                           backend=perf.get('backend', result.get('correctness', {}).get('backend')),
                           timing_scope=perf.get('timing_scope'))
                row.update({k: v for k, v in correct.get(name, {}).items() if k != 'method'})
                if result['status'] == 'completed' and name in measured:
                    row.update({k: v for k, v in measured[name].items() if k not in ('method', 'iteration_times_ms')})
                    accepted.append(row)
            rows.append(row)
            if name in correct:
                errors.append({**base, 'case_status': result['status'], **correct[name]})
            if name in measured:
                for index, elapsed in enumerate(measured[name].get('iteration_times_ms', [])):
                    samples.append({**base, 'method': name, 'case_status': result['status'],
                                    'sample_index': index, 'elapsed_ms': elapsed})
            for index, loss in enumerate(perf.get('trajectories', {}).get(name, [])):
                epoch_times = perf.get('epoch_times_seconds', {}).get(name, [])
                convergence.append({**base, 'method': name, 'case_status': result['status'], 'epoch': index + 1,
                                    'training_loss': loss, 'epoch_seconds': epoch_times[index] if index < len(epoch_times) else None})
        timing_key = 'mean_iteration_time_ms' if config.workload == 'synthetic' else 'training_seconds'
        for baseline in accepted:
            for candidate in accepted:
                if baseline['backend'] != candidate['backend'] or baseline['timing_scope'] != candidate['timing_scope']:
                    continue
                ratios = performance_metrics(baseline[timing_key], candidate[timing_key])
                comparisons.append({**base, 'backend': baseline['backend'], 'baseline_method': baseline['method'],
                                    'method': candidate['method'], 'timing_metric': timing_key,
                                    'baseline_time': baseline[timing_key], 'method_time': candidate[timing_key],
                                    'speedup': ratios['speedup_vs_sequential'],
                                    'time_saving_percent': ratios['time_saving_percent']})
    write_json(root / 'summary.json', {'schema_version': SCHEMA_VERSION, 'rows': rows})
    for name, data in [('summary', rows), ('errors', errors), ('timings', samples), ('convergence', convergence), ('comparisons', comparisons)]:
        write_csv(root / f'{name}.csv', data)
    return rows


def run_suite(plan, output, requested, *, resume=False, plan_only=False):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    spec = {'schema_version': SCHEMA_VERSION, 'cases': plan, 'requested_methods': list(requested)}
    if manifest_path.exists():
        if not resume:
            raise ValueError('output already contains a suite; use --resume or a different output directory')
        manifest = json.loads(manifest_path.read_text())
        if manifest['specification'] != spec:
            raise ValueError('resume configuration differs from the saved experiment matrix')
    else:
        if any(root.iterdir()):
            raise ValueError('output directory must be empty for a new suite')
        manifest = {'specification': spec, 'created_at_unix': time.time(), 'sessions': []}
    environment = environment_snapshot(probe_gpu=not plan_only)
    manifest['sessions'].append({'started_at_unix': time.time(), 'environment': environment, 'plan_only': plan_only})
    manifest['status'] = 'planned' if plan_only else 'running'
    write_json(manifest_path, manifest)
    write_json(root / 'capabilities.json', [
        {'case_id': case['case_id'], 'methods': capabilities(ExperimentConfig(**case['configuration']), requested)} for case in plan])
    if plan_only:
        return manifest
    try:
        require_execution_backend(ExperimentConfig(**plan[0]['configuration']))
    except Exception as exc:
        manifest.update(status='blocked', reason=f'{type(exc).__name__}: {exc}')
        write_json(manifest_path, manifest)
        raise
    completed, failed, unsupported = 0, 0, 0
    try:
        for index, case in enumerate(plan, 1):
            directory = root / 'cases' / case['case_id']
            result = None
            if resume and (directory / 'case.json').exists():
                result = json.loads((directory / 'case.json').read_text())
                # Interrupted profiling does not mark collection complete.
                if result['status'] not in ('completed', 'unsupported') or (
                    result['status'] == 'completed' and 'collection_finished_at_unix' not in result):
                    result = None
            if result is None:
                print(f'[{index}/{len(plan)}] Collecting {case["case_id"]}', flush=True)
                result = execute_case(case, directory, requested, environment)
            else:
                print(f'[{index}/{len(plan)}] Reusing {case["case_id"]}', flush=True)
            completed += result['status'] == 'completed'
            unsupported += result['status'] == 'unsupported'
            failed += result['status'] in ('failed', 'validation_failed')
            export_tables(root, plan)
            manifest['counts'] = {'completed_cases': completed, 'failed_cases': failed, 'unsupported_cases': unsupported}
            write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(status='interrupted', reason=f'{type(exc).__name__}: {exc}')
        write_json(manifest_path, manifest)
        raise
    manifest.update(status='completed_with_failures' if failed else 'completed', finished_at_unix=time.time())
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/experiments')
    parser.add_argument('--axes', nargs='+', choices=['qubits', 'depth', 'parameters'], default=['qubits', 'depth', 'parameters'])
    parser.add_argument('--qubits', nargs='+', type=int, default=[10, 15, 20, 25])
    parser.add_argument('--depths', nargs='+', type=int, default=[3, 5, 7])
    parser.add_argument('--fixed-qubits', type=int, default=20)
    parser.add_argument('--fixed-layers', type=int, default=3)
    parser.add_argument('--parameters', nargs='+', type=int, default=[36, 72, 108, 144, 180, 216, 270])
    parser.add_argument('--parameter-qubits', type=int, default=20)
    parser.add_argument('--parameter-layers', type=int, default=5)
    parser.add_argument('--differentiations', nargs='+', choices=['parameter-shift', 'adjoint'], default=['parameter-shift', 'adjoint'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7])
    parser.add_argument('--workloads', nargs='+', choices=['synthetic', 'iris', 'wine', 'breast-cancer', 'mnist-pca'], default=['synthetic'])
    parser.add_argument('--methods', nargs='+', choices=list(METHOD_NAMES), default=list(METHOD_NAMES))
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--streams', type=int, default=4)
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--sm-demand', type=float, default=None)
    parser.add_argument('--partition-size', type=int, default=16)
    parser.add_argument('--affinity-weight', type=float, default=1.0)
    parser.add_argument('--sync-weight', type=float, default=1.0)
    parser.add_argument('--parallelism-weight', type=float, default=1.)
    parser.add_argument('--resource-weight', type=float, default=1.)
    parser.add_argument('--memory-gib', type=float)
    parser.add_argument('--memory-safety-factor', type=float, default=.8)
    parser.add_argument('--reference-backend', choices=['default.qubit', 'lightning.gpu'])
    parser.add_argument('--backend', choices=['torch-cuda', 'pennylane'], default='torch-cuda')
    parser.add_argument('--cpu-validation', action='store_true')
    parser.add_argument('--save-traces', action='store_true', help='Retain correctness/timing traces in addition to profile artifacts')
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.cpu_validation and args.backend != 'torch-cuda':
        parser.error('--cpu-validation is only available for the Torch backend')
    base = ExperimentConfig(execution_backend='torch-cpu' if args.cpu_validation else args.backend,
                            reference_backend=args.reference_backend or ('default.qubit' if args.cpu_validation else 'lightning.gpu'),
                            warmup=args.warmup, iterations=args.iterations, samples=args.samples, streams=args.streams,
                            gpu_device=args.gpu_device, sm_demand=args.sm_demand, partition_size=args.partition_size,
                            affinity_weight=args.affinity_weight, sync_weight=args.sync_weight, parallelism_weight=args.parallelism_weight, resource_weight=args.resource_weight,
                            memory_budget_bytes=None if args.memory_gib is None else int(args.memory_gib * (1 << 30)),
                            memory_safety_factor=args.memory_safety_factor,
                            enable_trace=args.save_traces, enable_nsight=not args.no_profile)
    plan = build_cases(base, axes=args.axes, qubits=args.qubits, depths=args.depths,
                       fixed_qubits=args.fixed_qubits, fixed_layers=args.fixed_layers,
                       parameters=args.parameters, parameter_qubits=args.parameter_qubits,
                       parameter_layers=args.parameter_layers, differentiations=args.differentiations,
                       seeds=args.seeds, workloads=args.workloads, epochs=args.epochs)
    result = run_suite(plan, args.output, args.methods, resume=args.resume, plan_only=args.plan_only)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'status': result['status'],
                      'case_count': len(plan), 'counts': result.get('counts')}, indent=2))
    if result.get('counts', {}).get('failed_cases'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
