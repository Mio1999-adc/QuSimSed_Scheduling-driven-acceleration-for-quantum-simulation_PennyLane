"""Exercise widget wiring and charts with Streamlit's real app harness."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit required for UI tests')
class InterfaceTests(unittest.TestCase):
    def test_controls_reach_execution_and_results_render(self):
        from streamlit.testing.v1 import AppTest
        from qusimsed.result_collection import collect_results

        @collect_results('training')
        def training(config):
            captured.append(config)
            return {'configuration': config.metadata(), 'backend': 'test-backend',
                    'rows': [{'method': 'Sequential', 'training_seconds': 1., 'test_loss': .1,
                              'test_accuracy': .9, 'speedup_vs_sequential': 1., 'time_saving_percent': 0.}],
                    'trajectories': {'Sequential': [.3, .1]}, 'epoch_times_seconds': {'Sequential': [.5, .5]}}

        captured = []
        with tempfile.TemporaryDirectory() as directory:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app/streamlit_app.py')).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            def widget(kind, label):
                return next(item for item in getattr(app, kind) if item.label == label)
            widget('text_input', 'Output directory').set_value(directory)
            widget('number_input', 'Qubits').set_value(3)
            widget('number_input', 'Circuit depth (layers)').set_value(2)
            widget('selectbox', 'Execution backend').select('torch-cpu')
            widget('selectbox', 'Differentiation').select('adjoint')
            widget('checkbox', 'Train all rotation parameters').uncheck()
            widget('checkbox', 'Collect companion Nsight profile').uncheck()
            app.run()
            widget('number_input', 'Trainable parameters').set_value(4)
            widget('number_input', 'Learning rate').set_value(.02)
            widget('number_input', 'Maximum partition nodes').set_value(7)
            widget('number_input', 'Memory budget cap (GiB; 0 = automatic)').set_value(2.)
            widget('selectbox', 'Application dataset').select('wine')
            widget('number_input', 'Training epochs').set_value(2)
            with patch('qusimsed.real_benchmarks.run_real_benchmark', side_effect=training):
                widget('button', 'Run real QML benchmark').click()
                app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            config = captured[0]
            self.assertEqual((config.qubits, config.layers, config.trainable_parameters), (3, 2, 4))
            self.assertEqual((config.workload, config.iterations, config.differentiation), ('wine', 2, 'adjoint'))
            self.assertEqual(config.learning_rate, .02)
            self.assertEqual(config.partition_size, 7)
            self.assertEqual(config.memory_budget_bytes, 2 * (1 << 30))
            self.assertTrue(list(Path(directory).glob('experiments/*/result.json')))
            self.assertGreaterEqual(len(app.dataframe), 2)  # current result and saved-run viewer
            self.assertTrue(any(item.value == 'Training loss by epoch' for item in app.subheader))
            self.assertTrue(any(item.label == 'Saved run' for item in app.selectbox))
