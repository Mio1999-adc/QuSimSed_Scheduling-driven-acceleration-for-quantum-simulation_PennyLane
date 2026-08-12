# QuSim-Sed — Scheduling-Driven Acceleration for Hybrid Classical-Quantum Simulation on GPU with PennyLane

This repository accompanies the paper **"QuSim-Sed: Scheduling-Driven Acceleration for PennyLane-based Hybrid Classical Quantum Simulation on GPU"** and contains the benchmark code, Catalyst comparison, and an interactive console used to reproduce and explore its results.

---

## 1. Paper overview

### The problem
Variational Quantum Circuit (VQC) training on GPUs alternates between two phases every iteration:

- **Forward / Circuit Graph** — encode input, run the parameterized circuit, measure an expectation value.
- **Backward / AutoGrad Graph** — differentiate (parameter-shift or adjoint), aggregate gradients, update parameters.

Frameworks like PennyLane, TorchQuantum, and TensorFlow Quantum execute these two graphs **sequentially and on a single CUDA stream**: `T = T_enc → T_fwd → T_grad → T_opt`. Even though parameter-shift gradients require `2P` circuit evaluations that are mathematically *independent* of one another, the framework dispatches them one at a time with a synchronization barrier between the forward and backward phase. The result: idle SMs, low kernel overlap, and a GPU that is only nominally "GPU-accelerated."

### The idea
QuSim-Sed adds a **Coordinating Layer** between PennyLane's Circuit Graph (`G_circ`) and AutoGrad Graph (`G_grad`) *without modifying either*. It:

1. Builds a lightweight **Coordinating Data Structure (CDS)** — one `CDSRecord` per node of both graphs, stored in a flat `RecordPool` (constant-time access, negligible memory overhead). Each record tracks `ParentList`/`ChildList` (intra-graph deps), `CrossGraphLinks` (e.g. measurement `M_i` → gradient `G_i`), a `ReadyCounter`, `ExecutionState`, `TargetStream`, `Priority`, and `ResourceMetadata`.
2. Runs a **single-pass topological scheduler** over the CDS: whenever a node's `ReadyCounter` hits zero, it's pushed onto a ready queue and dispatched asynchronously to a free CUDA stream — regardless of which graph it belongs to. This is what lets an independent shifted-circuit evaluation and a gradient-accumulation step run *concurrently* instead of waiting on a global sync barrier.
3. Gatekeeps dispatch with **resource-aware scheduling** (Algorithm 2): before launching a task, it estimates the task's memory (`m_state = 2^n × sizeof(complex)`, larger under Adjoint due to retained intermediate states) and only dispatches if `M_allocated + m_req < α·M_total`; otherwise it waits for a running task to finish and free memory. This prevents OOM without needing to shrink the circuit or the qubit count.

### Headline results (NVIDIA A100, 80GB, PennyLane + JAX + Lightning-GPU, Table III setup)
| Differentiation | Result |
|---|---|
| Parameter-shift | Up to **6.24× speedup**, **80.3–83.6%** execution-time reduction vs. sequential PennyLane; consistently beats both Gradient-only and Quantum-only, and consistently beats Catalyst alone (~2.1–2.9× across qubit counts, ~5.15–5.17× across layer depths) |
| Adjoint | Only **marginal** improvement (~1.06×, <6% time reduction) for every method — the reverse-state propagation's strict forward→backward dependency leaves almost no independent work to schedule concurrently |
| QuSim-Sed + Catalyst | Combining scheduling (macro-level) with JIT compilation (micro-level) gives the largest gains under parameter-shift (e.g. ~3.6–3.9× at low qubit counts), since the two optimizations are complementary rather than redundant |

**Experimental setup (Table III):** qubits `{10, 15, 20, 25}`, layers `{3, 5, 7}`, both differentiation methods, 30 timed iterations after 1 warm-up, synthetic random inputs/parameters (so results reflect *scheduling*, not learning).

---

## 2. New data structures (proposal)

This is the core data-structure proposal of the paper: a way to give a scheduler visibility into *both* PennyLane's Circuit Graph and its AutoGrad Graph at once, without touching either graph's own representation. Two structures make this possible.

### 2.1 `CDSRecord` — one record per computation node

Every node of `G_circ` (state init, encoding, variational gates, measurement) and every node of `G_grad` (differentiation op, gradient accumulation, optimizer update) gets exactly one `CDSRecord`:

```
class CDSRecord:
    node_id             # unique identifier for this node
    graph_id            # which graph the node belongs to: Circuit or AutoGrad
    node_type           # operator kind (e.g. gate, measurement, grad-op, opt-update)
    parent_list         # intra-graph predecessors (same graph as this node)
    child_list           # intra-graph successors
    cross_graph_links    # edges to nodes in the *other* graph (e.g. measurement M_i -> gradient G_i)
    ready_counter        # unresolved-dependency count; node is dispatchable once this hits 0
    execution_state       # NotReady / Ready / Running / Done
    target_stream         # which CUDA stream this node is assigned to
    priority              # scheduling priority used when multiple nodes are ready at once
    resource_metadata     # estimated memory footprint / kernel characteristics, for the resource-aware gate
```

`parent_list`/`child_list` capture ordinary within-graph dependencies (the kind PennyLane and the AutoGrad engine already know about). `cross_graph_links` is the new part: it's what lets a *gradient* node depend on a *quantum-circuit* node's output (or vice versa) without merging the two graphs into one — the two graphs stay exactly as PennyLane/JAX produced them; the CDS is a side-table of relationships layered on top.

### 2.2 `RecordPool` — the scheduler's flat runtime view

```
RecordPool = [CDSRecord()] * MAX
```

All `CDSRecord`s live in one contiguous array rather than being scattered across two separate graph objects. This is what gives the scheduler:

- **Constant-time record access** (`RecordPool[node_id]`) instead of re-walking `G_circ`/`G_grad` on every scheduling decision.
- **Negligible memory overhead** — each record is small, fixed-size metadata, not a copy of the actual tensors/quantum state.
- **A single unified view for topological scheduling** — the scheduler never needs to know or care which of the two original graphs a ready node came from; it just sees "some record with `ready_counter == 0`."

### 2.3 How the two structures are built and used

**Construction (Algorithm 1).** One traversal of `G_circ ∪ G_grad` creates a `CDSRecord` per node, copies its `NodeID`/`GraphID`/`NodeType`, and fills in `ParentList`/`ChildList` from the graphs' own edges. A second pass walks the framework-reported data dependencies *between* the two graphs (e.g. "this measurement feeds this gradient computation") and records them as `CrossGraphLinks`. Finally every record's `ReadyCounter` is initialized to `|ParentList| + |CrossGraphLinks|`, and its `ExecutionState` set to `NotReady`.

**Scheduling (Algorithm 2).** The scheduler repeatedly pops the highest-priority record from a ready queue `Q`, estimates its memory need from `resource_metadata`, and — only if a stream is free *and* `M_allocated + m_req < α·M_total` — dispatches it asynchronously to that stream. If no stream/memory is available, it instead waits for a running task to finish, releases that task's stream and memory, marks it `Done`, and decrements the `ReadyCounter` of every node in its `ChildList` **and** its `CrossGraphLinks`. Any node whose counter reaches zero is pushed onto `Q`. Because `ChildList` and `CrossGraphLinks` are decremented through the same mechanism, an independent quantum-circuit evaluation and an independent gradient computation can become ready and get dispatched to different streams *at the same time* — this is precisely what turns the framework's synchronous `T_fwd → T_grad` pipeline into overlapping, concurrent execution.

### 2.4 Why this design, specifically

- **No framework modification.** Because the CDS is a side-structure built from graphs PennyLane/JAX already construct, QuSim-Sed doesn't need to patch PennyLane's QNode, JAX's autodiff, or the simulator backend — it only needs read access to both graphs' edges plus the cross-graph data dependencies the framework already exposes at runtime.
- **Cross-graph dependencies are explicit, not inferred.** Without `CrossGraphLinks`, a scheduler operating on `G_circ` and `G_grad` separately (which is what "Gradient-only" and "Quantum-only" effectively do) can only parallelize *within* one graph. Making the `M_i → G_i`-style links first-class metadata is what lets a single scheduler pass see opportunities that span both graphs — the core mechanism behind Cross-graph/QuSim-Sed's advantage over either single-graph optimization alone.
- **Resource-awareness is a gate, not a rewrite.** `resource_metadata` + the `M_allocated + m_req < α·M_total` check in Algorithm 2 bounds concurrency to what the GPU can actually hold; it never changes what a node computes, only when it's allowed to start. That's an important distinction from the qubit-block-splitting shortcut used in one of this repo's earlier scripts (§5) — the CDS's resource-awareness throttles *how much runs at once*, it never shrinks the *problem* being run.

---

## 3. Codebase overview

| File | Paper concept it implements | Notes |
|---|---|---|
| `qusimsed_four_config_benchmark.py` | Sequential / Gradient-only / Quantum-only / Cross-graph, swept across qubits and layers | Earlier iteration — its Quantum-only/Cross-graph split the qubit register into smaller independent blocks. This shrinks the actual state space being simulated, which inflates speedups at large qubit counts for a reason unrelated to scheduling (see §5). Kept for reference; superseded by the merged script below for anything you want to trust numerically. |
| `qusimsed_catalyst_benchmark.py` | Adds Catalyst QJIT and a QuSim-Sed+Catalyst "Merged" strategy, compared honestly against Sequential/Gradient-only/Cross-graph on one fixed circuit | This is the fair design later mirrored in the current paper's Figs. 9–10. |
| `qusimsed_merged_benchmark.py` | **The reference implementation.** All six strategies — Sequential, Gradient-only, Quantum-only, Cross-graph (QuSim-Sed proxy), Catalyst QJIT, and Merged (QuSim-Sed+Catalyst) — run the *exact same* full-width circuit at a given (qubits, layers). Only the dispatch of the `2P` independent shift evaluations differs: loop / vmap-batch / thread-pool concurrency / compiled / compiled+async. | Use this one to reproduce numbers in the realistic 1×–3× range the paper reports, rather than the inflated numbers a block-decomposed circuit produces. |
| `qusimsed_console_simple.html` | An interactive, in-browser cost-model console mirroring `qusimsed_merged_benchmark.py`'s scheduling logic | Sliders for qubits, layers, differentiation method, concurrent streams, and chunk size; shows per-iteration time and speedup for all six strategies live, plus a qubit-vs-parameter-count speedup sweep table. Useful for intuition-building without needing a GPU. |

### What the CDS / Algorithm 1–2 correspond to in code
The paper's `CDSRecord`/`RecordPool`/topological scheduler is a general **mechanism**; the benchmark scripts implement its *effect* directly rather than the generic data structure, since the goal is measuring scheduling *outcomes* (Sequential vs Gradient-only vs Quantum-only vs Cross-graph), not the CDS's own overhead:

- **Gradient-only** ≈ Coordinating Layer batches only within the AutoGrad Graph → `jax.vmap` over the `2P` shifted circuit calls in one call (or in `chunk_size` groups).
- **Quantum-only** ≈ Coordinating Layer dispatches only Circuit Graph nodes concurrently → a `ThreadPoolExecutor` with `n_streams` workers, each processing an assigned slice of shift-pairs sequentially (no batching) — approximating separate CUDA streams for independent circuit evaluations.
- **Cross-graph (QuSim-Sed)** ≈ both graphs scheduled jointly → the same `n_streams` concurrent groups as Quantum-only, but each group is *also* `vmap`-batched internally, so both the Circuit Graph's stream-level concurrency and the AutoGrad Graph's batching are exploited at once — this is the direct analogue of Algorithm 1/2's `ReadyCounter`-driven concurrent dispatch, minus the generic bookkeeping structure.
- **Resource-awareness** (`m_state = 2^n × sizeof(complex)`, throttled dispatch) is modeled in `benchmark_four_configs_qubit_scaling`/`safe_chunk_size()`-style helpers that cap `chunk_size`/`n_streams` once the concurrent memory footprint approaches a GPU memory budget.

---

## 4. How to set up and run the tests

### 3.1 Environment
The paper's results were produced on **Google Colab** with an **NVIDIA A100 (80GB, 108 SMs)**. Any CUDA GPU with `pennylane-lightning[gpu]` support will work; the scripts fall back to CPU (`lightning.qubit`) automatically if no GPU device is found, so you can sanity-check the code path without a GPU (just expect much longer wall-clock times and no meaningful speedup at large qubit counts).

```bash
# Core simulation stack
pip install pennylane pennylane-lightning[gpu] custatevec-cu12 -q

# JAX (match to your CUDA version — see https://docs.jax.dev/en/latest/installation.html)
pip install --upgrade "jax[cuda12]" -q

# Catalyst + optimizer, needed for the Catalyst / Merged strategies
pip install pennylane-catalyst optax -q

# Plotting / data handling
pip install pandas matplotlib -q
```

If `pennylane-catalyst` isn't installed, `qusimsed_merged_benchmark.py` and `qusimsed_catalyst_benchmark.py` will print a warning and simply skip the Catalyst/Merged rows — Sequential/Gradient-only/Quantum-only/Cross-graph still run.

### 3.2 Running the benchmarks

```bash
# Reference benchmark: all 6 strategies, Table 6 sweep {4,6,8} qubits x {3,5} layers
python qusimsed_merged_benchmark.py
#  -> merged_table6_results.csv
#  -> merged_table6_all_strategies.png

# Earlier 4-config-only sweep (qubits 10-30) — useful for qualitative trends,
# but see the block-decomposition caveat in §5 before trusting absolute speedups
python qusimsed_four_config_benchmark.py
#  -> method_comparison_qubits_layers.png
#  -> speedup_vs_qubit_parameter_scaling.png

# Catalyst-focused comparison (Sequential / Gradient-only / Cross-graph / Catalyst / Merged)
python qusimsed_catalyst_benchmark.py
```

To reproduce the **paper's exact Table III setup** (qubits `{10,15,20,25}`, layers `{3,5,7}`, 30 timed iterations, 1 warm-up), edit the sweep parameters at the top of `benchmark_table6()` (or the equivalent sweep function) in `qusimsed_merged_benchmark.py` before running — the shipped default uses a smaller `{4,6,8}×{3,5}` sweep so the script finishes quickly on shared/Colab GPUs.

Each script prints per-configuration timings and speedups to stdout as it runs, and saves a CSV + PNG summary at the end.

### 3.3 Exploring without a GPU: the console
Open `qusimsed_console_simple.html` directly in a browser (no server needed). Use it to:

- Move the **Qubits** / **Layers** sliders and toggle **Parameter-Shift vs. Adjoint** to see per-iteration time and speedup for all six strategies update live.
- Expand **Advanced** to change GPU profile (A100 / V100 / CPU), concurrent streams, and vmap chunk size.
- Read off the **speedup vs. parameter-scaling table** (qubits swept 4→30 at your current layer count) to see how each strategy's advantage grows or plateaus as the parameter count increases — the console's analogue of the paper's Figs. 7–8.

This is a cost model, not live telemetry — treat it as intuition-building alongside the real benchmark scripts, not a substitute for them.

---

## 5. Known caveat: block-decomposition inflation (fixed in the merged script)

If you compare `qusimsed_four_config_benchmark.py`'s numbers at large qubit counts against `qusimsed_merged_benchmark.py`, you'll see the former reporting far larger ("thousands-of-x") speedups for Cross-graph. That's because its Quantum-only/Cross-graph implementations **split the qubit register into smaller independent blocks**, which reduces the actual `2^n` state size being simulated — an exponential reduction in *work*, not a scheduling win. `qusimsed_merged_benchmark.py` fixes this: every strategy runs the identical full-width circuit, so its speedups (and the paper's) stay in the realistic ~1×–6× band. Prefer the merged script (and the console, which mirrors it) for any numbers you intend to cite or compare against the paper.

---

## 6. Citation

```
QuSim-Sed: Scheduling-Driven Acceleration for PennyLane-based
Hybrid Classical Quantum Simulation on GPU.
(Working paper — SC'26 submission draft.)
```
