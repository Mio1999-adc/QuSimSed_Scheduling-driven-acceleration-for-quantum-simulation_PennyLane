"""Resource invariants from Algorithm 2 and equations (20)-(21)."""
import threading
import unittest
from qusimsed.core.cds import CDSRecord, RecordPool
from qusimsed.core.scheduler import ResourceAwareScheduler


class JointResourceConstraintTests(unittest.TestCase):
    def test_both_constraints_allow_only_the_fitting_pair(self):
        pool = RecordPool()
        for key, memory, sm, priority in [('a', 60, .6, 100), ('memory_blocked', 60, .1, 0),
                                          ('sm_blocked', 10, .6, 0), ('fits', 40, .4, 0)]:
            pool.add(CDSRecord(key, 'circuit', 'gate', memory_bytes=memory, sm_demand=sm, priority=priority))
        concurrent = threading.Event()
        def first(_):
            self.assertTrue(concurrent.wait(2), 'a feasible concurrent task was never admitted')
        tasks = {key: lambda _: None for key in pool.records}
        tasks['a'] = first
        tasks['fits'] = lambda _: concurrent.set()
        scheduler = ResourceAwareScheduler(pool, 4, 100, 1., gpu_sm_count=108)
        trace = scheduler.run(tasks)
        fits = next(row for row in trace if row.task_id == 'fits').resource_admission
        self.assertEqual(fits['memory_reserved_before_bytes'], 60)
        self.assertAlmostEqual(fits['sm_used_before'], .6)
        self.assertAlmostEqual(fits['available_sm_equivalents'], 43.2)
        for row in trace:
            r = row.resource_admission
            self.assertLessEqual(r['memory_reserved_before_bytes'] + r['task_incremental_memory_bytes'], r['memory_limit_bytes'])
            self.assertLessEqual(r['sm_used_before'] + r['task_sm_demand'], r['sm_capacity'] + 1e-12)

    def test_live_free_memory_does_not_hide_unmaterialized_reservations(self):
        pool = RecordPool()
        for key in ['a', 'b']:
            pool.add(CDSRecord(key, 'circuit', 'gate', memory_bytes=40, sm_demand=.2))
        scheduler = ResourceAwareScheduler(pool, 2, 100, 1., memory_available=lambda: 70,
                                           memory_allocated=lambda: 0)
        scheduler.run({key: lambda _: None for key in pool.records})
        self.assertEqual(scheduler.peak_reserved_bytes, 40)

    def test_materialized_persistent_lease_is_not_double_counted(self):
        pool = RecordPool()
        for key in ['init', 'gate', 'end']:
            pool.add(CDSRecord(key, 'circuit', 'gate', resource_group='state', group_memory_bytes=80))
        pool.add_edge('init', 'gate'); pool.add_edge('gate', 'end')
        memory = {'allocated': 0}
        scheduler = ResourceAwareScheduler(pool, 2, 100, 1., memory_available=lambda: 100-memory['allocated'],
                                           memory_allocated=lambda: memory['allocated'])
        trace = scheduler.run({'init': lambda _: memory.update(allocated=80), 'gate': lambda _: None,
                               'end': lambda _: memory.update(allocated=0)})
        self.assertEqual(len(trace), 3)
        gate = next(row for row in trace if row.task_id == 'gate').resource_admission
        self.assertEqual(gate['materialized_run_bytes'], 80)
        self.assertEqual(gate['memory_limit_bytes'], 100)

    def test_ranking_considers_resource_cost_and_new_parallelism(self):
        pool = RecordPool()
        pool.add(CDSRecord('a_cheap', 'circuit', 'gate', memory_bytes=10, sm_demand=.1))
        pool.add(CDSRecord('z_costly', 'circuit', 'gate', memory_bytes=90, sm_demand=.9))
        order = []
        ResourceAwareScheduler(pool, 1, 100, 1.).run({key: lambda _, key=key: order.append(key) for key in pool.records})
        self.assertEqual(order[0], 'a_cheap')
        pool = RecordPool()
        for key in ['a_unlocks', 'z_leaf', 'child']:
            pool.add(CDSRecord(key, 'circuit', 'gate', sm_demand=.1))
        pool.add_edge('a_unlocks', 'child')
        order = []
        ResourceAwareScheduler(pool, 1, 100, 1.).run({key: lambda _, key=key: order.append(key) for key in pool.records})
        self.assertEqual(order[0], 'a_unlocks')
