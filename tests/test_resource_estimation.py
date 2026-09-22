import unittest
from qusimsed.resource_estimation import SMDemandEstimator
from qusimsed.config import ExperimentConfig


class ResourceEstimationTests(unittest.TestCase):
    def test_default_is_automatic(self):
        self.assertIsNone(ExperimentConfig().sm_demand)
        ExperimentConfig().validate()

    def test_workload_size_width_and_gpu_capacity_affect_admission(self):
        estimator = SMDemandEstimator(sm_count=108)
        small = estimator.estimate(10)
        self.assertLess(small['fraction'], 1.)
        self.assertGreater(estimator.estimate(10, gate_wires=4)['fraction'], small['fraction'])
        self.assertGreater(estimator.estimate(10, reverse=True)['fraction'], small['fraction'])
        self.assertGreater(SMDemandEstimator(sm_count=40).estimate(10)['fraction'], small['fraction'])
        self.assertEqual(estimator.estimate(25)['fraction'], 1.)
        self.assertFalse(small['measured_occupancy'])

    def test_override_fallback_and_bounds(self):
        self.assertEqual(SMDemandEstimator(override=.25).estimate(25)['fraction'], .25)
        self.assertEqual(SMDemandEstimator().estimate(2)['fraction'], 1.)
        for invalid in [0, -1, 2, float('nan')]:
            with self.assertRaises(ValueError):
                SMDemandEstimator(override=invalid)
        for qubits in range(1, 30):
            self.assertTrue(0 < SMDemandEstimator(sm_count=108).estimate(qubits)['fraction'] <= 1)
