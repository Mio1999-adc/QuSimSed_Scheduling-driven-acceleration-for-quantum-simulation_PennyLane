import time
import unittest
import tempfile
from pathlib import Path

from qusimsed.core.cds import CDSRecord, RecordPool, TaskState
from qusimsed.core.scheduler import ResourceAwareScheduler
from qusimsed.memory import live_memory_snapshot, metadata_memory, safe_qubit_counts
from qusimsed.profiling import collect_nsight
from qusimsed.validation import compare
from qusimsed.runtime import detect_runtime
from qusimsed.pennylane_experiments import METHODS
from qusimsed.real_benchmarks import DATASET_DESCRIPTIONS


class SchedulerTests(unittest.TestCase):
    def pool(self):
        pool = RecordPool()
        for key, graph in [("a", "circuit"), ("b", "circuit"), ("g", "autograd")]:
            pool.add(CDSRecord(key, graph, "op", memory_bytes=10))
        pool.add_edge("a", "g", cross_graph=True); pool.add_edge("b", "g", cross_graph=True)
        return pool

    def test_cross_graph_dependencies_and_completion(self):
        pool = self.pool(); scheduler = ResourceAwareScheduler(pool, 2, 100)
        trace = scheduler.run({key: lambda slot: time.sleep(.001) for key in pool.records})
        self.assertEqual({row.task_id for row in trace}, {"a", "b", "g"})
        self.assertEqual(pool.records["g"].state, TaskState.DONE)
        self.assertGreaterEqual(pool.records["g"].ready_counter, 0)

    def test_memory_gate(self):
        pool = self.pool()
        with self.assertRaises(MemoryError): ResourceAwareScheduler(pool, 2, 5).run({key: lambda slot: None for key in pool.records})

    def test_numerical_tolerance(self):
        self.assertTrue(compare([1.0, 2.0], [1.0 + 1e-8, 2.0]).within_tolerance)

    def test_metadata_memory_is_measured(self):
        report = metadata_memory(self.pool())
        self.assertGreater(report.total_metadata_bytes, 0)
        self.assertEqual(report.nodes, 3)

    def test_safe_qubit_planning_rejects_oom(self):
        rows = safe_qubit_counts([4, 10], total_memory_bytes=16_000, streams=2, safety_factor=0.8)
        self.assertTrue(rows[0]["admitted"])
        self.assertFalse(rows[1]["admitted"])

    def test_runtime_selection_is_explicit(self):
        runtime = detect_runtime()
        self.assertIn(runtime.selected_mode, {"cuda-streams", "cpu-threads"})
        self.assertTrue(runtime.reason)

    def test_memory_snapshot_has_explicit_fields(self):
        snapshot = live_memory_snapshot()
        self.assertIn("cuda_peak_allocated_bytes", snapshot)
        self.assertIn("process_rss_bytes", snapshot)

    def test_nsight_collector_writes_manifest_without_fabrication(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile"
            result = collect_nsight(["python", "--version"], output)
            self.assertTrue(output.with_suffix(".nsight.json").exists())
            self.assertIn("collected", result)

    def test_requested_fair_baselines_are_registered(self):
        self.assertEqual(METHODS, ("Sequential", "Batched parameter-shift", "Naive multi-stream", "QuSim-Sed"))

    def test_realistic_benchmark_suite_includes_pca_mnist(self):
        self.assertEqual(set(DATASET_DESCRIPTIONS), {"iris", "wine", "breast-cancer", "mnist-pca"})


if __name__ == "__main__": unittest.main()
