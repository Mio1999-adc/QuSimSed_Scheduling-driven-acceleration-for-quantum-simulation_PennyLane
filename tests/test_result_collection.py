import csv
import json
from pathlib import Path
import tempfile
import unittest
from qusimsed.config import ExperimentConfig
from qusimsed.result_collection import collect_results


class AutomaticCollectionTests(unittest.TestCase):
    def test_training_is_saved_without_explicit_save_and_runs_are_unique(self):
        @collect_results('training')
        def training(config):
            return {'configuration': config.metadata(),
                    'rows': [{'method': 'Sequential', 'training_seconds': 2., 'speedup_vs_sequential': 1., 'time_saving_percent': 0.}],
                    'trajectories': {'Sequential': [.5, .25]},
                    'epoch_times_seconds': {'Sequential': [1., 1.]}}
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(output_dir=directory)
            first, second = training(config), training(config)
            self.assertNotEqual(first['collection'], second['collection'])
            path = Path(first['collection']['directory'])
            saved = json.loads((path / 'result.json').read_text())
            self.assertEqual(saved['status'], 'completed')
            rows = list(csv.DictReader((path / 'convergence.csv').read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[1]['training_loss']), .25)

    def test_failure_and_opt_out(self):
        @collect_results('correctness')
        def failing(config):
            raise ValueError('numerical failure')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'numerical failure'):
                failing(ExperimentConfig(output_dir=directory))
            files = list(Path(directory).glob('experiments/*/result.json'))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())['status'], 'failed')
            with self.assertRaises(ValueError):
                failing(ExperimentConfig(output_dir=directory, collect_results=False))
            self.assertEqual(len(list(Path(directory).glob('experiments/*/result.json'))), 1)
