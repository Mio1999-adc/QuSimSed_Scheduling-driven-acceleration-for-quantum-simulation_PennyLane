# Methodology revisions for the maintained implementation

This is a proposed manuscript revision, checked against `core/cds.py`,
`core/scheduler.py`, `scheduled_vqc.py`, `resource_estimation.py`, and
`tape_executor.py`. It does not modify the source PDF or establish new GPU
performance results. References to Algorithm 1, Algorithm 2 and equations
(14)–(22) refer to the supplied QuSim-Sed manuscript.

## Required changes

| Manuscript item | Revision needed |
|---|---|
| Algorithm 1: graph inputs | Describe actual PennyLane tape extraction, parameter-shift expansion or adjoint reverse-task construction, and Torch FX differentiation/loss arithmetic. This is not extraction of arbitrary internal Autograd/JAX graphs. |
| Algorithm 1: predecessor counting | Count distinct predecessors once, including cross-graph and shared-state storage edges. In the implementation, `ParentList` already includes cross-graph parents. |
| CDS representation | The current `RecordPool` is a dictionary keyed by node ID, not a contiguous array. State expected constant-time lookup; do not claim negligible overhead without measurement. |
| Partitioning | Specify deterministic topological grouping by state ownership and graph type, bounded by partition size. It is not an optimized graph-cut algorithm. |
| Algorithm 2: memory initialization | Initialize reservations with persistent outputs/FX workspace, not zero. Distinguish task scratch from state/workspace leases. |
| Algorithm 2: resource feasibility | Check both remaining SM budget and the current effective global-memory budget before every dispatch. |
| Algorithm 2: task selection | Define the actual priority, aging, affinity, potential-parallelism, synchronization and resource-pressure score. |
| Algorithm 2: completion | Release task scratch and SM demand after device completion; release a state lease only after its last owning task. Failed tasks never unlock successors. |
| SM demand | Document the automatic static model and optional manual override. An SM-equivalent estimate is not a physical SM reservation or measured occupancy. |
| CUDA execution | Describe scheduler-owned Torch streams and events, plus host workers that wait for individual completion events. The current backend is not Lightning-GPU/cuStateVec. |
| Figures and experiments | Label the current Torch executor separately from historical JAX/Lightning/Catalyst results. Catalyst combinations remain unsupported in this executor. |

## Revised Algorithm 1: construct and initialize CDS

```text
Input: supported analytic PennyLane tape T, differentiation method d,
       loss/update arithmetic, partition-size bound K,
       GPU properties and optional demand override
Output: RecordPool R, initial ready set Q, dependency-preserving partitions

1. Extract tape operations, wire dependencies, measurement and trainable indices.
2. Construct executable tasks:
     parameter-shift: shifted tape evaluations and recipe-based reductions;
     adjoint: forward tape and an ordered reverse/uncompute sweep;
     classical: traced Torch FX reduction/loss/update operations.
3. Create one record per executable task, keyed by unique task ID.
   Store graph/type, priority, state ownership, scratch/lease memory estimates,
   estimated SM demand and its source, and synchronization-cost metadata.
4. Insert each distinct intra-graph and cross-graph dependency (u, v):
     add u to R[v].ParentList;
     add v to R[u].ChildList or R[u].CrossGraphLinks.
5. Add storage-order dependencies between successive mutations of one state.
6. Validate reciprocal edges, nonnegative resources and acyclicity.
7. Traverse in deterministic topological order. Start a new partition when
   (state ownership, graph type) changes or the current partition reaches K.
   Assign partition IDs; do not contract edges or insert partition barriers.
8. For each v:
     ReadyCounter[v] = number of distinct members of ParentList[v];
     clear prior stream assignment;
     state[v] = READY if ReadyCounter[v] == 0, otherwise NOT_READY.
9. Q = all READY records. Return R, Q and partitions.
```

Cross-graph edges must not be counted again after step 4. A partition ID is an
affinity hint, while state ownership controls persistent allocation lifetime.
They are distinct concepts even when related nodes share both.

## Resource definitions for revised Algorithm 2

Let `B` be the persistent output/workspace reservation, `s_tau` task scratch,
`L_g` the maximum lease estimate of tasks in state group g, and `H` the set of
currently leased groups. Let `M_res` include B, leased groups and running scratch.
The additional reservation for a candidate is

```text
delta_m(tau) = s_tau + (L_g if g exists and g is not in H, otherwise 0).
```

The current VQC executor estimates each state as `2**n * 16` bytes (complex128).
Forward evaluations reserve eight state equivalents plus 1 MiB slack; adjoint
evaluations reserve twelve plus 1 MiB. These are implementation-specific
workspace estimates, not universal simulator requirements.

The initial memory limit is `M0 = floor(alpha * min(available_at_start, user_cap))`,
with the cap omitted when absent. Available memory includes reusable Torch
cache. Before dispatch, read live allocatable memory F and the increase A in
Torch allocated memory since scheduling began. The implementation samples
allocation before and after F and uses the smaller nonnegative increase:

```text
A = max(0, min(allocated_before, allocated_after) - allocated_at_start)
M_limit(t) = min(M0, floor(alpha * max(0, F + A)))
```

Adding A back avoids charging materialized state twice; pending reservations
remain charged in M_res. Without allocation telemetry, use A = 0; without live
memory telemetry, use M0. This is admission bookkeeping, not an atomic GPU
memory reservation: external allocations can still race a check.

Normalize full device SM capacity to `C = 1`. Let `U` be the sum of running
task demands. The two hard conditions are

```text
M_res + delta_m(tau) <= M_limit(t)
U + r_tau <= C
```

Available SM capacity is `C - U`. Multiplying normalized capacity/demand by
the detected SM count yields estimated SM equivalents. CUDA does not assign
that number of physical SMs to a stream.

### Automatic demand estimate: report it explicitly

For a state-owning task, the implemented static model is

```text
W = 2**n * 2**max(0, k-1) * (2 for adjoint-reverse tasks, otherwise 1)
u = max(1, ceil(W / 256))
r_tau = min(1, u / N_SM)
```

Here k is the task's gate width when supplied; otherwise the maximum tape
operation/observable width is used conservatively. For non-state classical
tasks W = 1. Without GPU properties, demand defaults to 1. Manual overrides
replace the model (non-state classical tasks use min(0.05, override)).

The constant 256 defines model work units. It is not a measured kernel block
size. This model uses workload metadata and hardware capacity but does not
estimate actual cuBLAS occupancy, bandwidth consumption, or interference.
Keep those distinctions in the paper; the exact coefficients require GPU
evaluation if presented as quantitatively accurate resource predictions.

## Actual trade-off selection score

For each feasible task/available-stream pair `(tau, s)`, the implementation uses

```text
score = priority(tau) + w_age * age(tau)
      + w_aff * home_match(tau, s)
      + w_par * min(S, newly_ready_successors(tau)) / S
      - w_sync * sync_cost(tau) * cross_stream_parent_count(tau, s)
      - w_res * [delta_m(tau) / max(1, M_limit - M_res)
                 + r_tau / max(epsilon, C - U)]
```

S is the configured stream count. A successor contributes to potential
parallelism if its ready counter is currently one. Age is measured in dispatch
ticks, not wall-clock time. Home affinity uses the state group when present,
otherwise the partition. Defaults are 1 for affinity, parallelism,
synchronization and resource weights, and 0.01 for aging. The current sync-cost
coefficient defaults to 1 per parent edge; it is not calibrated event latency.
Ties prefer older readiness, then lexicographically larger task ID, then the
lower stream index. No claim of global optimality is needed.

The manuscript's scheduling-overhead term is not independently estimated per
task in this implementation. Say so, rather than claiming the runtime directly
optimizes every term of equation (15).

## Revised Algorithm 2: completion-driven, jointly constrained dispatch

```text
Input: validated/partitioned CDS R, ready set Q, stream pool S,
       memory budget and telemetry, normalized SM capacity C, score weights

1. Compute each group's lease L_g and remaining owning-task count.
2. Reject tasks that cannot fit the initial memory or SM capacity.
3. M_res = B; U = 0; H = empty; Running = empty; all streams available.
4. While Q is nonempty or Running is nonempty:
5.   While Q is nonempty and an available stream exists:
6.     Refresh live memory limit; compute memory and SM headroom.
7.     F = all ready task/available-stream pairs passing both constraints.
8.     If F is empty: break the dispatch loop.
9.     Select the highest-scoring pair (tau, s).
10.    Remove tau from Q; mark RUNNING; reserve s and any new state lease.
11.    M_res += delta_m(tau); U += r_tau; update affinity and admission trace.
12.    Submit worker on s:
         insert waits for predecessor events from different streams;
         launch task work on the owned CUDA stream;
         record completion event and wait for that event before returning.
13.  If Running is empty: report infeasible/live-pressure state when Q is
     nonempty; otherwise finish the loop.
14.  Wait for at least one worker's device-complete result.
15.  For every completed task tau:
16.    Release its stream; M_res -= s_tau; U -= r_tau.
17.    On failure: mark REJECTED and abort; do not unlock successors.
18.    Mark DONE. Decrement the remaining count of its state group.
19.    If that count becomes zero: release L_g from M_res and remove g from H.
20.    For each distinct child in ChildList union CrossGraphLinks:
         decrement ReadyCounter; insert into Q when the count reaches zero.
21. Verify every task is DONE; return traces and resource accounting.
```

The host may wait for a task's own event while other stream workers continue;
there is no device-wide barrier at every completion. This implementation does
not enqueue dependent tasks before observing their predecessors' completion.
Do not describe it as an entirely device-driven scheduler.

## Revise the objective's interpretation

Equation (14), `integral |A(t)| dt`, equals the sum of task-active durations.
For fixed task durations, it does not distinguish serial execution from
parallel execution. With contention, slower tasks can even increase it.
Therefore it should not be presented as a standalone overlap/throughput
objective that this implementation maximizes.

A clearer high-level objective is minimizing execution makespan (or maximizing
completed-work throughput) subject to dependencies, memory, SM demand and stream
limits. Present the score above as a practical heuristic toward that objective.
If retaining equation (14), label it aggregate active-task time and discuss its
limitations. Use measured makespan, speedup and GPU overlap as evaluation
metrics; use newly exposed successors only as the scheduler's proxy for useful
parallelism. Do not imply that the scheduler computes future makespan exactly.

## Manuscript consistency checklist

- Update Figures 3/4 backend and CDS labels for the implementation being reported.
- Distinguish proposed architecture, implemented heuristic and measured results.
- Keep historical JAX/Lightning/Catalyst numbers separate from this Torch backend.
- Describe one analytic expectation, supported unitary tapes, separate shifted
  evaluations and the ordered adjoint reverse sweep.
- Avoid generalizing the selected baseline's sequential behavior to every
  PennyLane device/interface/differentiation configuration.
- Explain that resource constraints always apply; profiling verifies realized
  kernel overlap and helps evaluate estimator quality.
- Include `resource_admission` trace fields and resource-constraint tests as
  validation evidence; CPU tests do not establish GPU performance.
