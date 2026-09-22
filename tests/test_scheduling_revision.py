import importlib.util
import threading
import time
import unittest

from qusimsed.core.cds import CDSRecord, RecordPool, TaskState
from qusimsed.core.scheduler import ResourceAwareScheduler


class SchedulingRevisionTests(unittest.TestCase):
    def test_partitions_preserve_every_edge_and_node_readiness(self):
        pool = RecordPool()
        for key in 'abcd':
            pool.add(CDSRecord(key, 'circuit', 'gate'))
        for parent, child in [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')]:
            pool.add_edge(parent, child)
        original = {r.node_id: r.parents.copy() for r in pool}
        groups = pool.partition(2)
        self.assertEqual(sum(map(len, groups.values())), 4)
        self.assertTrue(all(len(g) <= 2 for g in groups.values()))
        self.assertEqual(original, {r.node_id: r.parents for r in pool})
        order = []
        scheduler = ResourceAwareScheduler(pool, 2, 100)
        scheduler.run({r.node_id: lambda _, key=r.node_id: order.append(key) for r in pool})
        self.assertLess(order.index('a'), order.index('b'))
        self.assertLess(order.index('b'), order.index('d'))
        self.assertLess(order.index('c'), order.index('d'))

    def test_persistent_state_is_not_released_at_gate_boundary(self):
        pool = RecordPool()
        for group in ['x', 'y']:
            for key in ['init', 'gate', 'end']:
                pool.add(CDSRecord(f'{group}-{key}', 'circuit', key, resource_group=group,
                                   group_memory_bytes=60, priority=10 if key != 'init' else 0))
            pool.add_edge(f'{group}-init', f'{group}-gate')
            pool.add_edge(f'{group}-gate', f'{group}-end')
        order = []
        scheduler = ResourceAwareScheduler(pool, 4, 100, 1)
        scheduler.run({r.node_id: lambda _, key=r.node_id: order.append(key) for r in pool})
        self.assertEqual(scheduler.peak_reserved_bytes, 60)
        first = order[0][0]
        other = 'x' if first == 'y' else 'y'
        self.assertLess(order.index(first + '-end'), order.index(other + '-init'))

    def test_sm_admission_limits_concurrent_workers(self):
        pool = RecordPool()
        for i in range(5):
            pool.add(CDSRecord(str(i), 'circuit', 'gate', sm_demand=.6))
        lock = threading.Lock()
        active, peak = 0, 0
        def work(_):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            time.sleep(.002)
            with lock:
                active -= 1
        scheduler = ResourceAwareScheduler(pool, 4, 100)
        scheduler.run({r.node_id: work for r in pool})
        self.assertEqual(peak, 1)
        self.assertLessEqual(scheduler.peak_sm_demand, 1)

    def test_ready_gradient_preempts_unlaunched_circuits(self):
        pool = RecordPool()
        for key, graph, priority in [('a', 'circuit', 0), ('z', 'circuit', 0), ('grad', 'autograd', 20)]:
            pool.add(CDSRecord(key, graph, 'op', priority=priority))
        pool.add_edge('z', 'grad', cross_graph=True)
        order = []
        ResourceAwareScheduler(pool, 1, 100).run({r.node_id: lambda _, key=r.node_id: order.append(key) for r in pool})
        self.assertEqual(order, ['z', 'grad', 'a'])

    def test_child_prefers_parent_stream(self):
        pool = RecordPool()
        pool.add(CDSRecord('parent', 'circuit', 'gate'))
        pool.add(CDSRecord('child', 'autograd', 'gradient'))
        pool.add_edge('parent', 'child', cross_graph=True)
        trace = ResourceAwareScheduler(pool, 3, 100).run({r.node_id: lambda _: None for r in pool})
        self.assertEqual(trace[0].stream_id, trace[1].stream_id)

    def test_live_memory_pressure_fails_before_execution(self):
        pool = RecordPool()
        pool.add(CDSRecord('x', 'circuit', 'gate', memory_bytes=10))
        with self.assertRaises(MemoryError):
            ResourceAwareScheduler(pool, 2, 100, memory_available=lambda: 1).run({'x': lambda _: self.fail('launched')})

    def test_failure_never_unlocks_successor(self):
        pool = RecordPool()
        pool.add(CDSRecord('a', 'circuit', 'gate'))
        pool.add(CDSRecord('b', 'autograd', 'grad'))
        pool.add_edge('a', 'b', cross_graph=True)
        def fail(_):
            raise ValueError('device failed')
        with self.assertRaisesRegex(RuntimeError, 'device failed'):
            ResourceAwareScheduler(pool, 2, 100).run({'a': fail, 'b': lambda _: self.fail('unlocked')})
        self.assertEqual(pool.records['b'].state, TaskState.NOT_READY)


HAS_STACK = importlib.util.find_spec('pennylane') is not None and importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_STACK, 'PennyLane and Torch required')
class TapeExecutionTests(unittest.TestCase):
    def circuit(self, qubits=3):
        import pennylane as qml
        from pennylane import numpy as np
        theta = np.array([.2, -.4, .7], requires_grad=True)
        @qml.qnode(qml.device('default.qubit', wires=qubits), diff_method='parameter-shift')
        def circuit(theta):
            qml.RX(theta[0], wires=0)
            if qubits > 1:
                qml.Hadamard(wires=qubits-1)
                qml.CNOT(wires=[qubits-1, 0])
            qml.RY(theta[1], wires=qubits-1)
            qml.RZ(theta[2], wires=0)
            return qml.expval(qml.PauliX(0))
        return circuit, theta

    def test_both_gradient_methods_and_mse_update_match_pennylane(self):
        import numpy as np
        import pennylane as qml
        from qusimsed.pennylane_experiments import _tape
        from qusimsed.scheduled_vqc import ScheduledVQC
        for qubits in [1, 3]:
            circuit, theta = self.circuit(qubits)
            expected = float(circuit(theta))
            gradient = qml.grad(circuit)(theta)
            for method in ['parameter-shift', 'adjoint']:
                for streams in [1, 3]:
                    with self.subTest(qubits=qubits, method=method, streams=streams):
                        plan = ScheduledVQC(_tape(circuit, theta), differentiation=method, streams=streams,
                                            device='cpu', memory_budget_bytes=1 << 29, sm_demand=.3)
                        result = plan.add_training_step(target=.3, learning_rate=.1).run()
                        np.testing.assert_allclose(result['expectation'], expected, atol=1e-10)
                        np.testing.assert_allclose(result['gradient'], gradient, atol=1e-10)
                        loss_grad = 2 * (expected - .3) * gradient
                        np.testing.assert_allclose(result['loss_gradient'], loss_grad, atol=1e-10)
                        np.testing.assert_allclose(result['parameters'], theta - .1 * loss_grad, atol=1e-10)
                        self.assertTrue(result['classical_graph_metadata'])
                        self.assertEqual(result['backend'], 'torch-cpu-statevector-validation')
                        self.assertTrue(all(r['executor'] == 'host-thread' for r in result['trace']))
                        by_id = {r['task_id']: r for r in result['trace']}
                        for row in result['trace']:
                            for parent in row['dependencies']:
                                self.assertLessEqual(by_id[parent]['end_ns'], row['start_ns'])

    def test_extracted_wire_dependencies(self):
        from qusimsed.graph_adapter import extract_tape
        from qusimsed.pennylane_experiments import _tape
        circuit, theta = self.circuit()
        graph = extract_tape(_tape(circuit, theta))
        self.assertEqual(graph['operations'][2]['parents'], [0, 1])
        self.assertEqual(len(graph['parameters']), 3)

    def test_general_shift_recipe_and_adjoint_controlled_rotation(self):
        import numpy as np
        import pennylane as qml
        from pennylane import numpy as pnp
        from qusimsed.pennylane_experiments import _tape
        from qusimsed.scheduled_vqc import ScheduledVQC
        theta = pnp.array([.73], requires_grad=True)
        @qml.qnode(qml.device('default.qubit', wires=2))
        def circuit(x):
            qml.Hadamard(0)
            qml.CRX(x[0], wires=[0, 1])
            return qml.expval(qml.PauliZ(1))
        expected = qml.grad(circuit)(theta)
        for differentiation in ['parameter-shift', 'adjoint']:
            result = ScheduledVQC(_tape(circuit, theta), differentiation=differentiation,
                                  device='cpu', memory_budget_bytes=1 << 29, sm_demand=.25).run()
            np.testing.assert_allclose(result['gradient'], expected, atol=1e-10)

    def test_unsupported_state_preparation_fails_explicitly(self):
        import pennylane as qml
        from qusimsed.graph_adapter import extract_tape
        tape = qml.tape.QuantumScript([qml.StatePrep([1., 0.], wires=0)], [qml.expval(qml.PauliZ(0))])
        with self.assertRaisesRegex(ValueError, 'state-preparation'):
            extract_tape(tape)

    def test_forward_only_plan_and_baseline_timing(self):
        from qusimsed.config import ExperimentConfig
        from qusimsed.pennylane_experiments import _workload, evaluate_expectation, baseline_experiment
        config = ExperimentConfig(qubits=2, layers=1, execution_backend='torch-cpu', warmup=0, iterations=1)
        _, theta, circuit, _ = _workload(config)
        self.assertAlmostEqual(evaluate_expectation(circuit, theta, config), float(circuit(theta)), places=10)
        result = baseline_experiment(config)
        self.assertEqual([r['method'] for r in result['rows']], ['Sequential', 'QuSim-Sed'])
        self.assertTrue(all(r['mean_iteration_time_ms'] > 0 for r in result['rows']))

    def test_gpu_request_never_falls_back(self):
        import torch
        from qusimsed.tape_executor import TapeExecutor
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, 'refusing CPU fallback'):
                TapeExecutor(2)

    def test_no_adjoint_shift_tasks(self):
        from qusimsed.pennylane_experiments import _tape
        from qusimsed.scheduled_vqc import ScheduledVQC
        circuit, theta = self.circuit()
        plan = ScheduledVQC(_tape(circuit, theta), differentiation='adjoint', device='cpu', memory_budget_bytes=1 << 29)
        self.assertFalse(any('shift_' in r.node_id for r in plan.pool))
        result = plan.run()
        self.assertTrue(any(r['node_type'] == 'adjoint-reverse' for r in result['trace']))

    def test_real_state_footprint_rejected(self):
        from qusimsed.pennylane_experiments import _tape
        from qusimsed.scheduled_vqc import ScheduledVQC
        circuit, theta = self.circuit()
        plan = ScheduledVQC(_tape(circuit, theta), device='cpu', memory_budget_bytes=1024)
        with self.assertRaises(MemoryError):
            plan.run()

    def test_experiment_adjoint_config_is_obeyed(self):
        from qusimsed.config import ExperimentConfig
        from qusimsed.pennylane_experiments import correctness_experiment
        result = correctness_experiment(ExperimentConfig(qubits=2, layers=1, execution_backend='torch-cpu', differentiation='adjoint'))
        self.assertTrue(all(r['all_within_tolerance'] for r in result['rows']))
        self.assertTrue(any(row['node_type'] == 'adjoint-reverse' for row in result['traces']['QuSim-Sed']))

    def test_application_training_routes_through_configured_backend(self):
        import numpy as np
        from qusimsed.config import ExperimentConfig
        from qusimsed.pennylane_experiments import _imports, _device
        from qusimsed.real_benchmarks import _run_method
        config = ExperimentConfig(qubits=2, layers=1, execution_backend='torch-cpu',
                                  differentiation='adjoint', iterations=1, sm_demand=.25)
        qml, pnp = _imports()
        device, _ = _device(qml, 2, config)
        x, y = np.array([[.2, .3], [-.4, .1]]), np.array([0, 1])
        initial = pnp.array(np.arange(6) * .1, requires_grad=False)
        outcomes = [_run_method(method, qml, device, config, x, y, x, y, initial)
                    for method in ['Sequential', 'QuSim-Sed']]
        np.testing.assert_allclose(outcomes[0]['parameters'], outcomes[1]['parameters'], atol=1e-10)
        np.testing.assert_allclose(outcomes[0]['trajectory'], outcomes[1]['trajectory'], atol=1e-10)

    def test_cuda_execution_and_events_when_available(self):
        import torch
        import pennylane as qml
        import numpy as np
        from qusimsed.pennylane_experiments import _tape
        from qusimsed.scheduled_vqc import ScheduledVQC
        if not torch.cuda.is_available():
            self.skipTest('requires GPU server')
        circuit, theta = self.circuit()
        gradient = qml.grad(circuit)(theta)
        for method in ['parameter-shift', 'adjoint']:
            plan = ScheduledVQC(_tape(circuit, theta), differentiation=method, streams=3, sm_demand=.25)
            result = plan.add_training_step().run()
            np.testing.assert_allclose(result['gradient'], gradient, atol=1e-9)
            self.assertTrue(all(r['cuda_stream_id'] is not None for r in result['trace']))
            if method == 'parameter-shift':
                self.assertGreater(result['cross_stream_event_waits'], 0)


if __name__ == '__main__':
    unittest.main()
