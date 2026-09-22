import tempfile
import unittest
from unittest.mock import patch, MagicMock

from qusimsed.train import main, Dashboard


class TrainingLauncherTests(unittest.TestCase):
    def test_ui_starts_before_training_and_stops_on_exit(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            def train(config):
                events.append('training')
                self.assertEqual(config.workload, 'wine')
                self.assertEqual(config.iterations, 2)
                return {'status': 'completed', 'collection': {}, 'rows': []}
            with patch('sys.argv', ['train', '--dataset', 'wine', '--epochs', '2', '--port', '8512',
                                    '--output-dir', directory, '--exit-after-training']):
                with patch('qusimsed.train.Dashboard') as dashboard, patch('qusimsed.train.run_real_benchmark', side_effect=train):
                    instance = dashboard.return_value
                    instance.start.side_effect = lambda: events.append('dashboard') or instance
                    main()
                    self.assertEqual(events, ['dashboard', 'training'])
                    dashboard.assert_called_once_with(directory, 8512)
                    instance.stop.assert_called_once()

    def test_training_failure_stops_dashboard(self):
        with patch('sys.argv', ['train']), patch('qusimsed.train.Dashboard') as dashboard:
            with patch('qusimsed.train.run_real_benchmark', side_effect=RuntimeError('failed')):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    main()
            dashboard.return_value.start.return_value.stop.assert_called_once()

    def test_no_ui_does_not_start_a_server(self):
        with patch('sys.argv', ['train', '--no-ui']), patch('qusimsed.train.Dashboard') as dashboard:
            with patch('qusimsed.train.run_real_benchmark', return_value={'status': 'completed', 'collection': {}, 'rows': []}):
                main()
            dashboard.assert_not_called()
