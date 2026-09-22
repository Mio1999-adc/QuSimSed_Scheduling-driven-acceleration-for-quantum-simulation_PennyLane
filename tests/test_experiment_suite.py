import csv
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qusimsed.config import ExperimentConfig
from qusimsed.experiment_suite import build_cases, capabilities, execute_case, export_tables, run_suite, write_json
from qusimsed.metrics import performance_metrics, timing_summary


def small_plan(**changes):
    base = ExperimentConfig(execution_backend='torch-cpu', iterations=1, warmup=0,
                            enable_trace=False, enable_nsight=False)
    options = dict(axes=['qubits', 'depth', 'parameters'], qubits=[2], depths=[1],
                   fixed_qubits=2, fixed_layers=1, parameters=[1, 3], parameter_qubits=2,
                   parameter_layers=1, differentiations=['parameter-shift', 'adjoint'],
                   seeds=[7], workloads=['synthetic'], epochs=1)
    options.update(changes)
    return build_cases(base, **options)


class CollectionTests(unittest.TestCase):
    def test_ratios_and_slowdowns(self):
        self.assertEqual(performance_metrics(10, 2)['speedup_vs_sequential'], 5)
        self.assertEqual(performance_metrics(10, 2)['time_saving_percent'], 80)
        self.assertEqual(performance_metrics(10, 20)['time_saving_percent'], -100)
        for value in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                performance_metrics(1, value)
        self.assertEqual(timing_summary([1, 2, 3])['median_iteration_time_ms'], 2)

    def test_axes_are_deduplicated_and_both_differentiations_retained(self):
        plan = small_plan()
        self.assertEqual(len(plan), 6)
        full = [p for p in plan if p['configuration']['trainable_parameters'] == 6]
        self.assertTrue(all(p['axes'] == ['qubits', 'depth'] for p in full))
        self.assertEqual({p['configuration']['differentiation'] for p in plan}, {'parameter-shift', 'adjoint'})

    def test_invalid_parameter_capacity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'trainable parameter'):
            small_plan(parameters=[7])

    def test_unsupported_catalyst_is_explicit(self):
        rows = capabilities(ExperimentConfig(), ['sequential', 'qusimsed', 'catalyst', 'qusimsed-catalyst'])
        self.assertEqual([r['supported'] for r in rows], [True, True, False, False])
        self.assertIn('historical proxy', rows[-1]['reason'])

    def test_numerical_failure_blocks_performance(self):
        case = small_plan(axes=['qubits'], differentiations=['adjoint'])[0]
        with tempfile.TemporaryDirectory() as directory:
            validation = {'rows': [{'method': 'QuSim-Sed', 'all_within_tolerance': False}]}
            with patch('qusimsed.experiment_suite.correctness_experiment', return_value=validation):
                with patch('qusimsed.experiment_suite.baseline_experiment') as benchmark:
                    result = execute_case(case, directory, ['sequential', 'qusimsed'], {})
                    benchmark.assert_not_called()
            self.assertEqual(result['status'], 'validation_failed')
            self.assertTrue((Path(directory) / 'correctness.json').exists())

    def test_failed_case_is_saved(self):
        case = small_plan()[0]
        with tempfile.TemporaryDirectory() as directory:
            with patch('qusimsed.experiment_suite.correctness_experiment', side_effect=MemoryError('too large')):
                result = execute_case(case, directory, ['qusimsed'], {})
            self.assertEqual(result['status'], 'failed')
            self.assertIn('too large', json.loads((Path(directory) / 'case.json').read_text())['reason'])

    def test_exports_do_not_fabricate_unsupported_results(self):
        plan = small_plan(axes=['qubits'], differentiations=['adjoint'])
        case = plan[0]
        config = ExperimentConfig(**case['configuration'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'method': 'Sequential', 'mean_iteration_time_ms': 10, 'iteration_times_ms': [9, 11], 'speedup_vs_sequential': 1},
                    {'method': 'QuSim-Sed', 'mean_iteration_time_ms': 2, 'iteration_times_ms': [1, 3], 'speedup_vs_sequential': 5}]
            result = {**case, 'status': 'completed',
                      'capabilities': capabilities(config, ['sequential', 'qusimsed', 'catalyst']),
                      'correctness': {'rows': [{'method': r['method'], 'all_within_tolerance': True, 'gradient_max_abs': 0} for r in rows]},
                      'performance': {'backend': 'test', 'timing_scope': 'test scope', 'rows': rows}}
            write_json(root / 'cases' / case['case_id'] / 'case.json', result)
            export_tables(root, plan)
            summary = list(csv.DictReader((root / 'summary.csv').read_text().splitlines()))
            catalyst = next(r for r in summary if r['method'] == 'Catalyst')
            self.assertEqual(catalyst['status'], 'unsupported')
            self.assertEqual(catalyst['mean_iteration_time_ms'], '')
            pairs = list(csv.DictReader((root / 'comparisons.csv').read_text().splitlines()))
            comparison = next(r for r in pairs if r['baseline_method'] == 'Sequential' and r['method'] == 'QuSim-Sed')
            self.assertEqual(float(comparison['speedup']), 5)
            self.assertEqual(float(comparison['time_saving_percent']), 80)
            self.assertEqual(len(list(csv.DictReader((root / 'timings.csv').read_text().splitlines()))), 4)
            self.assertEqual(len(list(csv.DictReader((root / 'errors.csv').read_text().splitlines()))), 2)
            # A failed training trajectory must suppress published comparisons.
            result['status'] = 'validation_failed'
            write_json(root / 'cases' / case['case_id'] / 'case.json', result)
            export_tables(root, plan)
            self.assertEqual(list(csv.DictReader((root / 'comparisons.csv').read_text().splitlines())), [])

    def test_plan_resume_rejects_changed_spec_and_reuses_complete_cases(self):
        plan = small_plan(axes=['qubits'], differentiations=['adjoint'])
        with tempfile.TemporaryDirectory() as directory:
            with patch('qusimsed.experiment_suite.environment_snapshot', return_value={}):
                run_suite(plan, directory, ['catalyst'], plan_only=True)
                first = run_suite(plan, directory, ['catalyst'], resume=True)
                self.assertEqual(first['counts']['unsupported_cases'], 1)
                with patch('qusimsed.experiment_suite.execute_case') as execute:
                    run_suite(plan, directory, ['catalyst'], resume=True)
                    execute.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'resume configuration'):
                    run_suite(plan, directory, ['qusimsed'], resume=True)
                with self.assertRaisesRegex(ValueError, 'already contains'):
                    run_suite(plan, directory, ['catalyst'])


HAS_STACK = importlib.util.find_spec('pennylane') and importlib.util.find_spec('torch')


@unittest.skipUnless(HAS_STACK, 'PennyLane and Torch required')
class ParameterScalingTests(unittest.TestCase):
    def test_freezing_parameters_preserves_circuit_and_initial_function(self):
        from qusimsed.pennylane_experiments import _workload, _tape, correctness_experiment
        full = ExperimentConfig(qubits=2, layers=1, execution_backend='torch-cpu', enable_trace=False)
        for method in ['parameter-shift', 'adjoint']:
            qml, initial, circuit, _ = _workload(replace(full, differentiation=method))
            value = float(circuit(initial))
            full_tape = _tape(circuit, initial)
            for count in [1, 3]:
                config = replace(full, differentiation=method, trainable_parameters=count)
                _, partial, partial_circuit, _ = _workload(config)
                tape = _tape(partial_circuit, partial)
                self.assertEqual(len(tape.trainable_params), count)
                self.assertEqual(len(tape.operations), len(full_tape.operations))
                self.assertAlmostEqual(float(partial_circuit(partial)), value, places=12)
                for a, b in zip(tape.operations, full_tape.operations):
                    np.testing.assert_allclose(qml.matrix(a), qml.matrix(b), atol=1e-12)
                result = correctness_experiment(config)
                self.assertTrue(all(row['all_within_tolerance'] for row in result['rows']))

    def test_application_curves_are_saved(self):
        from qusimsed.real_benchmarks import run_real_benchmark
        config = ExperimentConfig(qubits=2, layers=1, trainable_parameters=2, workload='iris',
                                  execution_backend='torch-cpu', differentiation='adjoint', iterations=2)
        x = np.array([[.1, .2], [.3, -.1]])
        y = np.array([0, 1])
        with patch('qusimsed.real_benchmarks.load_workload', return_value=(x, x, y, y)):
            result = run_real_benchmark(config)
        self.assertEqual(len(result['trajectories']['QuSim-Sed']), 2)
        self.assertEqual(len(result['epoch_times_seconds']['QuSim-Sed']), 2)
        self.assertTrue(all(row['correctness_within_tolerance'] for row in result['rows']))
        self.assertIn('time_saving_percent', result['rows'][1])


if __name__ == '__main__':
    unittest.main()
