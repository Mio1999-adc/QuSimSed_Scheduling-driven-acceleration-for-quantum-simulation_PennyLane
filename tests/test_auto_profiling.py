import json
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from qusimsed.config import ExperimentConfig
from qusimsed.profiling import NsightStatus, collect_nsight, profile_qusimsed


class AutomaticProfilingTests(unittest.TestCase):
    def test_unavailable_profiler_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'profile'
            with patch('qusimsed.profiling.nsight_status', return_value=NsightStatus(False, None, 'missing nsys')):
                result = collect_nsight(['python', '--version'], base)
            self.assertFalse(result['collected'])
            self.assertEqual(json.loads(Path(result['manifest']).read_text())['reason'], 'missing nsys')

    def test_exit_zero_without_report_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('qusimsed.profiling.nsight_status', return_value=NsightStatus(True, 'nsys', 'found')):
                with patch('qusimsed.profiling.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')):
                    result = collect_nsight(['python', '--version'], Path(directory) / 'profile')
            self.assertFalse(result['collected'])

    def test_real_report_required_and_nested_runs_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'profile.v1'
            def run(command, **kwargs):
                self.assertEqual(kwargs['env']['QUSIMSED_UNDER_NSYS'], '1')
                Path(str(base) + '.nsys-rep').write_bytes(b'report')
                return subprocess.CompletedProcess(command, 0, '', '')
            with patch('qusimsed.profiling.nsight_status', return_value=NsightStatus(True, 'nsys', 'found')):
                with patch('qusimsed.profiling.subprocess.run', side_effect=run):
                    result = collect_nsight(['python', '--version'], base)
            self.assertTrue(result['collected'])
            self.assertTrue(result['report'].endswith('profile.v1.nsys-rep'))

    def test_stale_report_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'profile'
            Path(str(base) + '.nsys-rep').write_bytes(b'old report')
            with patch('qusimsed.profiling.nsight_status', return_value=NsightStatus(True, 'nsys', 'found')):
                with patch('qusimsed.profiling.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')):
                    self.assertFalse(collect_nsight(['python', '--version'], base)['collected'])

    def test_companion_preserves_workload_and_prevents_recursion(self):
        config = ExperimentConfig(qubits=7, layers=5, differentiation='adjoint', streams=3,
                                  gpu_device=1, seed=19, sm_demand=.25, memory_budget_bytes=1 << 30)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with patch('qusimsed.profiling.collect_nsight', return_value={'collected': False}) as collect:
                result = profile_qusimsed(config, Path(directory) / 'profile')
            command = collect.call_args.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertIn('--no-profile', command)
            for flag, expected in [('--mode', 'trace'), ('--strategy', 'qusimsed'),
                                   ('--qubits', '7'), ('--layers', '5'), ('--differentiation', 'adjoint'),
                                   ('--gpu-device', '1'), ('--sm-demand', '0.25'), ('--memory-gib', '1.0')]:
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertFalse(result['timing_samples_profiled'])

    def test_cpu_and_nested_profiles_skip_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            for config, environment in [(ExperimentConfig(execution_backend='torch-cpu'), {}),
                                        (ExperimentConfig(), {'QUSIMSED_UNDER_NSYS': '1'})]:
                with patch.dict(os.environ, environment, clear=True), patch('qusimsed.profiling.collect_nsight') as collect:
                    result = profile_qusimsed(config, Path(directory) / 'profile')
                    collect.assert_not_called()
                    self.assertFalse(result['collected'])
                    self.assertTrue(Path(result['manifest']).exists())

    def test_launch_error_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('qusimsed.profiling.nsight_status', return_value=NsightStatus(True, 'nsys', 'found')):
                with patch('qusimsed.profiling.subprocess.run', side_effect=OSError('cannot execute')):
                    result = collect_nsight(['python', '--version'], Path(directory) / 'profile')
            self.assertFalse(result['collected'])
            self.assertIn('cannot execute', result['reason'])

    def test_server_default_profiles_after_saving_primary_result(self):
        self._server_case([], expected=True)

    def test_server_opt_out_and_sequential_trace_do_not_profile(self):
        self._server_case(['--no-profile'], expected=False)
        self._server_case(['--strategy', 'sequential'], expected=False)

    def test_collector_managed_server_does_not_profile_again(self):
        self._server_case([], expected=False, environment={'QUSIMSED_UNDER_NSYS': '1'})

    def _server_case(self, flags, *, expected, environment=None):
        from qusimsed import server_benchmark
        cuda = SimpleNamespace(is_available=lambda: True,
                               get_device_properties=lambda _: SimpleNamespace(name='test GPU', total_memory=100, multi_processor_count=2),
                               device=lambda _: nullcontext(), synchronize=lambda _: None, empty_cache=lambda: None)
        torch = SimpleNamespace(cuda=cuda, __version__='test', version=SimpleNamespace(cuda='test'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run.json'
            def profile(config, prefix):
                self.assertTrue(output.exists(), 'primary result must survive profiling failure/interruption')
                self.assertEqual(Path(prefix), output.parent / 'profiling' / 'run' / 'qusimsed')
                return {'collected': False, 'reason': 'test'}
            with patch.dict(os.environ, environment or {}, clear=True), patch.dict(sys.modules, {'torch': torch}):
                with patch.object(sys, 'argv', ['server_benchmark', '--mode', 'trace', '--output', str(output), *flags]):
                    with patch.object(server_benchmark, '_workload', return_value=(None, None, None, None)):
                        with patch.object(server_benchmark, '_run_scheduled', return_value={'backend': 'test', 'trace': []}):
                            with patch.object(server_benchmark, 'profile_qusimsed', side_effect=profile) as collect:
                                with patch('builtins.print'):
                                    server_benchmark.main()
                                self.assertEqual(collect.called, expected)
            self.assertEqual('profiling' in json.loads(output.read_text()), expected)


if __name__ == '__main__':
    unittest.main()
