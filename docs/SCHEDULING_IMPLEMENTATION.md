# Scheduling implementation and GPU-server validation

The maintained entry point is `python -m qusimsed.server_benchmark`. The older
`qusimsed_merged_benchmark*.py` files remain historical thread/vmap proxies;
they do not invoke this implementation.

## Backend identity

`torch-cuda-statevector` executes PennyLane tape operations on complex128 Torch
CUDA state vectors. It owns the CUDA streams, state buffers, events, and
classical gradient kernels. It is **not Lightning-GPU/cuStateVec**, and timing
results must retain this backend label. Placing a Lightning QNode inside a
Torch stream context is not treated as evidence that Lightning uses that
stream. A future cuStateVec adapter must explicitly bind its handle to the
scheduler's stream and obey the same completion contract.

`torch-cpu-statevector-validation` is opt-in (`--cpu-validation`) and exercises
the same arithmetic and dependencies locally. A CUDA request fails without
CUDA; it never falls back. PennyLane's `default.qubit` is used for the small,
independent numerical reference outside timed GPU execution. The GPU entry
point includes no CPU quantum simulation during benchmark/trace execution;
tape construction and small gate-matrix construction still occur on the host.

## Methodology mapping

| Paper concept | Implementation |
|---|---|
| Circuit metadata | `graph_adapter.extract_tape`: reads operations, wire dependencies, measurement dependencies, and trainable parameter ownership from the actual PennyLane tape |
| Classical gradient metadata | `trace_shift_reduction` and `trace_training_step` construct Torch FX graphs; `extract_fx` reads their actual edges, and `ScheduledVQC._fx` lowers those nodes into executable CDS tasks |
| Joint dependencies | Measurement-to-gradient edges, gradient stack, loss gradient, and SGD update are linked by the CDS; no quantum/classical phase barrier |
| Partitioning | `RecordPool.partition`: deterministic contiguous topological groups, bounded in size and separated by state ownership/graph type; original edges are retained |
| Affinity and synchronization cost | Ready task/stream candidates are ranked by priority + age + affinity + newly exposed parallelism - synchronization cost - resource pressure |
| SM admission | Sum of running task demand fractions must remain <= 1.0; these are estimates, not physical SM reservations |
| Memory admission | Live allocatable CUDA memory and an optional user cap bound the budget; safety factor is applied; state/workspace leases persist across gates until the entire owning evaluation finishes |
| CUDA execution | One owned Torch CUDA stream per slot, predecessor-event waits for cross-stream edges, per-task completion events, allocator `record_stream`, and NVTX task labels |
| Adjoint | Forward tape execution followed by a dependency-ordered reverse sweep using PennyLane operation derivatives; both primal and adjoint states are uncomputed using inverse gates |

The classical graph is the executable differentiation/loss arithmetic used by
this backend. It is not an extraction of arbitrary internal JAX or PennyLane
Autograd engine graphs. Adjoint reverse nodes are derived from the actual tape
and operation derivatives. The adapter accepts independent tape parameters;
it does not infer classical Jacobians for shared/transformed QNode arguments.
The repository's RX/RY/RZ ansatz uses one independent input per trainable gate.

Every circuit owns a separate state vector. Gates on disjoint wires still need
storage-order edges when they update that same state. Different circuit copies
can run concurrently. Subgraph boundaries are affinity hints, not barriers;
an individual ready node may run before other nodes in its partition finish.
Ready reductions have higher priority than unstarted circuits, allowing
classical gradient work to overlap remaining quantum work when resource
estimates permit. Partitioning does not fuse kernels.

Workers synchronize only their own completion event before returning. Thus
successor readiness and resource release mean **device completion**, not host
submission. Cross-stream events are additionally inserted to make ownership
and dependencies explicit. This conservative host-observed implementation
can incur substantial per-gate overhead; a speedup is not assumed. Host trace
intervals include dispatch and completion-observation overhead. Use Nsight for
actual kernel concurrency and GPU idle time.

## Resource model

A state uses `2**n * 16` bytes. Each independent forward evaluation reserves
8 state equivalents plus 1 MiB of matrix/allocator slack. Adjoint reserves 12
state equivalents plus the same slack. These bounds cover permutation copies,
matrix multiplication outputs, and primal/adjoint temporaries in this executor;
they are not claimed to describe Lightning's allocations. Scalar/FX outputs
have a separate conservative lifetime reservation. The executor reports Torch
peak allocated memory as well as scheduler reservations. CUDA allocatable
memory includes the reusable Torch cache and excludes other live allocations.
Other processes can allocate concurrently after an admission check, so runtime
allocation failures are surfaced, not silently retried or hidden.

SM demand is automatic by default (`sm_demand=None`): the static workload
estimator uses GPU SM count, state-vector size, gate width and reverse work.
A numeric `--sm-demand` is an explicit override, not a required tuning step.
The estimate is a normalized fraction of the GPU capacity, not an occupancy
measurement; `fraction * gpu_sm_count` gives estimated SM equivalents.

### Algorithm 2: jointly enforce memory and SM constraints

Before every dispatch, both of the paper's constraints (20)-(21) must pass:

```text
reserved_memory + task_incremental_memory <= effective_memory_limit
used_sm + task_sm_demand <= sm_capacity
```

An available stream and satisfied dependencies are also required. `used_sm`
is the sum of demands of dispatched, unfinished tasks; available SM capacity
is `sm_capacity - used_sm`. The stream executor releases these reservations
only after its CUDA completion event. This is the scheduler's available
capacity, not a query for physically idle SMs or an SM reservation in CUDA.

The memory limit starts with the smaller of available GPU memory and the
optional user cap, multiplied by the safety factor. It is rechecked against
live allocatable global memory before each dispatch. With allocator telemetry,
`effective_memory_limit = min(initial_limit, alpha * (live_allocatable +
materialized_run_bytes))`. Materialized bytes are the nonnegative increase in
Torch allocated memory since the run began. This adds back already allocated
state once, while still charging all pending reservations. Without allocator
telemetry the live check is conservative and adds nothing back. Persistent
state/workspace leases remain reserved across gate boundaries; only task
scratch is released at each gate completion.

Among feasible task/stream pairs, ranking additionally rewards successors
whose last unfinished predecessor would complete, and penalizes the task's
fractions of remaining memory and SM capacity. The exact score is:

```text
priority + aging + affinity_weight * home_stream_match
+ parallelism_weight * min(streams, newly_ready_successors) / streams
- sync_weight * sync_cost * cross_stream_parent_count
- resource_weight * (incremental_memory / memory_headroom + sm_demand / sm_headroom)
```

The two new weights default to 1 and are configurable in the CLIs/UI. This is
an explicit heuristic for the paper's trade-off objective, not an exact solver;
uniform dispatch overhead is not separately profiled per task. GPU profiling
checks realized overlap and throughput, not whether constraints should apply.

Every trace row records `resource_admission`: reserved/incremental memory,
live memory limit, used/available SM demand, GPU SM count, SM equivalents,
selection score, resource pressure and potential-parallelism benefit. These
records allow the two inequalities to be checked for every dispatch.


## Supported workload scope

Analytic, unitary tapes with one expectation measurement; small dense gates
and observables (up to four wires); scalar, single-parameter trainable gates
with parameter-shift frequencies/recipes and, for adjoint, an available generator.
Unsupported state preparation, noise, shots, parameter broadcasting, large
dense observables, and multi-parameter trainable operations fail explicitly.
This covers the repository's full-width RX/RY/RZ + CNOT VQC. Hamiltonian-sum
VQE support and Catalyst stream integration are not added by this change.

The maintained Torch benchmark compares **Sequential and QuSim-Sed using the
same kernels**. Legacy PennyLane batching is a separate backend, with no
cross-backend speedup calculation. The previous "Naive multi-stream" label is
not offered until it has an independent, validated implementation. Do not mix
these new results with the older paper figures without identifying the backend.

## Automatic profiling

The server CLI automatically collects one companion QuSim-Sed trace after
successful correctness/benchmark runs or a QuSim-Sed trace run. Primary timing
samples remain unprofiled. The extra execution has the same circuit, seed,
differentiation, device, resource settings, and stream count. Before replay,
the CLI releases unused Torch allocator cache so the child can use the GPU's
available memory. Primary results are saved first and augmented with a
`profiling` field afterward.

For `--output results/server/run.json`, the default prefix is
`results/server/profiling/run/qusimsed`, producing `.nsys-rep`, `.nsight.json`,
and `-trace.json` files when successful. Use `--profile-output` to override
that prefix or `--no-profile` to disable collection. Missing/failed Nsight
produces an explicit failure manifest. CPU validation produces a skip manifest.
Only a new, nonempty report plus successful process exit counts as collection.
Collector-managed child processes and the replay's `--no-profile` flag prevent
recursive profiling. The automation is attached to the server CLI; direct API
and Streamlit application calls are unchanged.

## Server commands

Install a CUDA-enabled Torch build matching the server using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then
install `requirements-gpu.txt`. A CPU-only Torch wheel will correctly fail the
GPU capability check. The implementation was locally tested with PennyLane
0.44.0 and Torch 2.5.1; retain server package and driver versions with results.

```bash
python -m unittest discover -s tests -v

# Independent numerical reference versus the actual scheduled execution.
python -m qusimsed.server_benchmark --mode correctness --qubits 4 --layers 2 \
  --differentiation parameter-shift --output results/server/shift-correctness.json
python -m qusimsed.server_benchmark --mode correctness --qubits 4 --layers 2 \
  --differentiation adjoint --output results/server/adjoint-correctness.json

# Use a justified SM estimate; 0.25 below is an explicit example estimate.
python -m qusimsed.server_benchmark --mode benchmark --qubits 10 --layers 3 \
  --streams 4 --sm-demand 0.25 --warmup 1 --iterations 30 \
  --output results/server/shift-timing.json

# Profile the real VQC scheduler, not the Torch matmul smoke demo.
python -m qusimsed.collectors nsight --output results/nsight/qusimsed -- \
  python -m qusimsed.server_benchmark --mode trace --strategy qusimsed \
  --qubits 10 --layers 3 --streams 4 --sm-demand 0.25 \
  --output results/server/qusimsed-trace.json
python -m qusimsed.collectors nsight --output results/nsight/sequential -- \
  python -m qusimsed.server_benchmark --mode trace --strategy sequential \
  --qubits 10 --layers 3 --output results/server/sequential-trace.json
```

Start with correctness and a small trace before scaling to 20/25 qubits. Timing
includes graph construction, forward execution, differentiation, MSE/SGD, and
device completion. Trace runs export CDS nodes, native wire metadata, traced
classical graphs, partitions, resource estimates, stream handles, and event-wait
counts. Timing runs retain the final trace per method. Both differentiation
methods honor configuration; correctness failures produce a nonzero CLI exit.

The Streamlit UI exposes backend choice and SM demand. Selecting `torch-cpu`
is explicitly local validation. The framework-independent demo remains a
scheduler smoke test, not a VQC performance result.
