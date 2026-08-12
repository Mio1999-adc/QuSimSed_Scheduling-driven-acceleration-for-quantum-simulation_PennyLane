"""
QuSim-Sed — Merged Benchmark: Four Configs + Catalyst + Catalyst-Merged
=========================================================================

WHY THIS FILE EXISTS (root-cause of the ~4000x vs ~2-3x discrepancy)
----------------------------------------------------------------------
The original `qusimsed_four_config_benchmark.py` builds "Quantum-only" and
"Cross-graph" out of a **block-decomposed circuit**: it splits the n-qubit
register into n_blocks independent sub-registers of width w = n/n_blocks,
with NO entangling gates crossing block boundaries, and runs each block on
its OWN device.

That looks like "splitting the Quantum Graph into independent subtrees"
(which is a legitimate scheduling idea), but it has a side effect that has
nothing to do with scheduling: a statevector simulator's cost scales as
2^n. Running n_blocks blocks of width w=n/n_blocks instead of one circuit
of width n replaces a single 2^n-sized computation with n_blocks separate
2^w-sized computations. Because 2^n grows exponentially and 2^w does not,
the "speedup" from block-splitting explodes at large qubit counts for a
reason that has NOTHING to do with concurrent GPU dispatch — it's doing
exponentially LESS total arithmetic, not the same arithmetic faster. That
is exactly why the four-config script can report absurd numbers like
~1000-4000x at 26-30 qubits: at that scale, block-splitting isn't
approximating a scheduler, it's approximating a different, much smaller
problem.

`qusimsed_catalyst_benchmark.py` (the Catalyst-comparison version) does
NOT do this: every strategy — Sequential, Gradient-Only, Cross-Graph,
Catalyst, Merged — runs the exact SAME full-width, fully-entangled
n-qubit circuit (`FlexibleVQC.circuit`, `CatalystVQC.circuit`,
`MergedVQC.circuit`, at Table 6's {4,6,8} qubits). The only thing that
differs between strategies is HOW the mathematically-independent 2P
shift evaluations are dispatched (one at a time / vmap-batched / compiled
+ async). That's why its Cross-Graph numbers land at a modest, paper-
consistent ~1.3-2x rather than thousands-of-x: it is actually measuring
scheduling, not measuring a smaller quantum state.

THE FIX applied in this merged file
----------------------------------------------------------------------
Every strategy below — Sequential, Gradient-Only, Quantum-Only, Cross-
Graph, Catalyst QJIT, and Merged (QuSim-Sed + Catalyst) — runs on ONE
shared, unmodified, full-width circuit at a given (n_qubits, n_layers).
No strategy ever changes the qubit count or entanglement structure being
simulated. Only the DISPATCH of the 2P independent shift-circuit
evaluations changes across strategies:

  1. Sequential      - one shift-pair at a time, plain Python loop.
  2. Gradient-Only    - all 2P shifts vmap-batched in one call (batches
                        the Gradient Graph; still a single GPU stream).
  3. Quantum-Only     - the SAME 2P shift-pairs, split into n_streams
                        groups and dispatched CONCURRENTLY via a thread
                        pool (approximating separate CUDA streams), with
                        NO vmap batching inside a group (parallelizes the
                        Quantum Graph's dispatch, not the Gradient Graph).
  4. Cross-Graph      - QuSim-Sed proxy: the n_streams groups from (3),
                        but each group is ALSO vmap-batched internally,
                        like (2). Combines both graphs' parallelism, as
                        in the paper's Coordinating Layer, without ever
                        touching qubit count.
  5. Catalyst QJIT    - whole gradient computation compiled once with
                        `qml.qjit`, removing per-iteration Python/host
                        dispatch overhead (a different bottleneck than
                        1-4: compilation, not concurrency).
  6. Merged           - one `@qjit(async_qnodes=True)` program containing
                        the same independent shift-circuit calls as (1),
                        so Catalyst's async runtime both compiles away
                        Python overhead AND concurrently dispatches the
                        independent evaluations - a real combination of
                        (4)'s idea and (5)'s idea, not just their speedups
                        added together.

Because every strategy shares the identical circuit, speedups reported
here are apples-to-apples and should stay in the small (roughly 1x-3x)
range reported in the paper (Figs. 6-11) rather than the several-
thousand-x seen when block-decomposition silently shrinks the problem.

Table 6 experimental setup (kept identical to both source files):
    Platform:            Google Colab
    GPU:                 NVIDIA A100 (80GB, 108 SMs)
    Number of Qubits:    {4, 6, 8}
    Number of Layers:    {3, 5}
    Circuit Type:        Variational Quantum Circuit (VQC)
    Differentiation:     Parameter-shift, Adjoint
    Execution Phases:    Forward, Backward

Run on Google Colab with an A100 GPU:
    !pip install pennylane pennylane-lightning[gpu] custatevec-cu12 -q
    !pip install pennylane-catalyst optax pandas matplotlib seaborn -q
"""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pennylane as qml
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.tree_util import tree_map

print("=" * 80)
print("QuSim-Sed MERGED BENCHMARK: 4 CONFIGS + CATALYST + CATALYST-MERGED")
print("=" * 80)
print(f"PennyLane: {qml.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 80)
print()

# ============================================================================
# Optional imports: Catalyst + Optax (only needed for strategies 5-6)
# ============================================================================
try:
    import catalyst
    from catalyst import qjit
    import optax
    CATALYST_AVAILABLE = True
except Exception as e:
    CATALYST_AVAILABLE = False
    warnings.warn(
        f"Catalyst/Optax not available ({e}). Strategies 1-4 will still run; "
        f"strategies 5-6 (Catalyst QJIT, Merged) will be skipped. "
        f"Install with: pip install pennylane-catalyst optax"
    )


# ============================================================================
# DEVICE SELECTION — mirrors Table 6 (NVIDIA A100 GPU, lightning.gpu)
# ============================================================================

def make_device(n_qubits):
    """lightning.gpu first (cuQuantum-accelerated statevector sim on A100),
    falling back to lightning.qubit (CPU) if no GPU device is available."""
    try:
        dev = qml.device('lightning.gpu', wires=n_qubits)
        return dev, 'lightning.gpu'
    except Exception as e1:
        try:
            dev = qml.device('lightning.qubit', wires=n_qubits)
            warnings.warn(
                f"lightning.gpu unavailable ({e1}); falling back to lightning.qubit (CPU)."
            )
            return dev, 'lightning.qubit'
        except Exception as e2:
            dev = qml.device('default.qubit', wires=n_qubits)
            warnings.warn(f"lightning.qubit also unavailable ({e2}); using default.qubit.")
            return dev, 'default.qubit'


_probe_dev, _probe_name = make_device(4)
print(f"Selected simulator backend: {_probe_name}")
print(f"Catalyst available: {CATALYST_AVAILABLE}")
print("=" * 80)
print()


# ============================================================================
# Shared VQC ansatz — IDENTICAL circuit used by every single strategy below.
# No strategy is allowed to change n_qubits, n_layers, or entanglement
# structure; that invariant is what keeps the comparison fair.
# ============================================================================

def build_ansatz(n_qubits, n_layers):
    def ansatz(params, x=None):
        if x is not None:
            for i in range(n_qubits):
                qml.RY(x[i % len(x)], wires=i)
        idx = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                qml.RX(params[idx], wires=q); idx += 1
                qml.RY(params[idx], wires=q); idx += 1
                qml.RZ(params[idx], wires=q); idx += 1
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
    return ansatz


# ============================================================================
# STRATEGIES 1-4: Sequential / Gradient-Only / Quantum-Only / Cross-Graph
# All four share ONE QNode on ONE device at the full n_qubits width.
# ============================================================================

class FlexibleVQC:
    """
    Parameter-shift and adjoint gradients under four dispatch strategies,
    all operating on the same full-width circuit (self.circuit). Only the
    dispatch pattern of the 2P independent shift-circuit evaluations
    differs between strategies — n_qubits and the ansatz never change.
    """

    def __init__(self, n_qubits, n_layers, diff_method="parameter-shift"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 3
        self.diff_method = diff_method
        self.dev, self.backend_name = make_device(n_qubits)

        ansatz = build_ansatz(n_qubits, n_layers)

        @qml.qnode(self.dev, interface='jax', diff_method=diff_method)
        def circuit(params, x=None):
            ansatz(params, x)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    # ---- Strategy 1: Sequential (naive, one shifted circuit at a time) ----
    def gradients_sequential(self, params, x=None):
        if self.diff_method == "adjoint":
            return grad(lambda p: self.circuit(p, x))(params)
        shift = jnp.pi / 2
        grads = []
        for i in range(self.n_params):
            p_plus = params.at[i].add(shift)
            p_minus = params.at[i].add(-shift)
            grads.append((self.circuit(p_plus, x) - self.circuit(p_minus, x)) / 2)
        return jnp.stack(grads)

    # ---- Strategy 2: Gradient-Only parallel (single vmap batch) ----
    # Batches the Gradient Graph (all 2P shifts at once) but still a
    # single dispatch call / single stream — no concurrent threads.
    def gradients_gradient_only(self, params, x=None):
        if self.diff_method == "adjoint":
            return grad(lambda p: self.circuit(p, x))(params)
        shift = jnp.pi / 2
        idx = jnp.arange(self.n_params)
        plus = vmap(lambda i: params.at[i].add(shift))(idx)
        minus = vmap(lambda i: params.at[i].add(-shift))(idx)
        vals_p = vmap(lambda p: self.circuit(p, x))(plus)
        vals_m = vmap(lambda p: self.circuit(p, x))(minus)
        return (vals_p - vals_m) / 2

    # ---- Strategy 3: Quantum-Only parallel ----
    # Splits the SAME 2P shift-pairs into n_streams groups, dispatched
    # CONCURRENTLY via a thread pool (approximating separate CUDA
    # streams). No vmap batching within a group -> parallelizes the
    # Quantum Graph's dispatch only, not the Gradient Graph. The circuit
    # and qubit count are untouched; only concurrency of independent
    # circuit *calls* changes.
    def _group_gradient_sequential(self, params, x, idx_slice):
        if self.diff_method == "adjoint":
            # No independent shift-copies exist under adjoint; a "group"
            # degenerates to the single reverse-mode pass.
            return grad(lambda p: self.circuit(p, x))(params)
        shift = jnp.pi / 2
        grads = []
        for i in idx_slice:
            p_plus = params.at[i].add(shift)
            p_minus = params.at[i].add(-shift)
            grads.append((self.circuit(p_plus, x) - self.circuit(p_minus, x)) / 2)
        return jnp.stack(grads) if grads else jnp.array(grads)

    def gradients_quantum_only(self, params, x=None, n_streams=4):
        if self.diff_method == "adjoint":
            return grad(lambda p: self.circuit(p, x))(params)
        n = self.n_params
        idx_groups = [list(range(s, n, n_streams)) for s in range(n_streams)]
        # Interleaved grouping keeps each group's workload balanced; the
        # concatenation order below matches jnp.arange(n) so results line
        # up with the other strategies' gradient ordering.
        with ThreadPoolExecutor(max_workers=n_streams) as pool:
            futures = [pool.submit(self._group_gradient_sequential, params, x, g)
                       for g in idx_groups if g]
            group_results = [f.result() for f in futures]
        out = jnp.zeros(n)
        for g, res in zip([g for g in idx_groups if g], group_results):
            out = out.at[jnp.array(g)].set(res)
        return out

    # ---- Strategy 4: Cross-Graph (QuSim-Sed proxy) ----
    # The same n_streams concurrent groups as Quantum-Only, but each
    # group is ALSO vmap-batched internally (like Gradient-Only). This
    # jointly exploits Quantum Graph concurrency (streams) and Gradient
    # Graph batching (vmap) — the paper's cross-graph scheduling idea —
    # without ever changing the circuit's qubit count.
    def _group_gradient_chunked(self, params, x, idx_slice, chunk_size):
        if self.diff_method == "adjoint":
            return grad(lambda p: self.circuit(p, x))(params)
        if not idx_slice:
            return jnp.array([])
        shift = jnp.pi / 2
        idx_arr = jnp.array(idx_slice)
        n = len(idx_slice)
        n_chunks = (n + chunk_size - 1) // chunk_size
        all_grads = []
        for c in range(n_chunks):
            cs, ce = c * chunk_size, min((c + 1) * chunk_size, n)
            chunk_idx = idx_arr[cs:ce]
            plus = vmap(lambda i: params.at[i].add(shift))(chunk_idx)
            minus = vmap(lambda i: params.at[i].add(-shift))(chunk_idx)
            vp = vmap(lambda p: self.circuit(p, x))(plus)
            vm = vmap(lambda p: self.circuit(p, x))(minus)
            all_grads.append((vp - vm) / 2)
        return jnp.concatenate(all_grads)

    def gradients_cross_graph(self, params, x=None, n_streams=4, chunk_size=4):
        if self.diff_method == "adjoint":
            # Table 1/3: strict forward-backward dependency, no independent
            # shift-copies to batch or split; cross-graph collapses to the
            # single reverse-mode pass, same as every other strategy here.
            return grad(lambda p: self.circuit(p, x))(params)
        n = self.n_params
        idx_groups = [list(range(s, n, n_streams)) for s in range(n_streams)]
        idx_groups = [g for g in idx_groups if g]
        with ThreadPoolExecutor(max_workers=len(idx_groups)) as pool:
            futures = [pool.submit(self._group_gradient_chunked, params, x, g, chunk_size)
                       for g in idx_groups]
            group_results = [f.result() for f in futures]
        out = jnp.zeros(n)
        for g, res in zip(idx_groups, group_results):
            out = out.at[jnp.array(g)].set(res)
        return out


# ============================================================================
# STRATEGY 5: Catalyst QJIT — whole gradient computation compiled once
# ============================================================================

class CatalystVQC:
    """
    Compiles circuit + gradient into a SINGLE executable with qml.qjit,
    eliminating per-iteration Python/host dispatch overhead. Requires
    pennylane-catalyst + optax, and a Catalyst-compatible device
    (lightning.qubit or lightning.gpu / lightning.kokkos). Uses the SAME
    ansatz/qubit count as FlexibleVQC above.
    """

    def __init__(self, n_qubits, n_layers, diff_method="adjoint", lr=0.01):
        if not CATALYST_AVAILABLE:
            raise RuntimeError("Catalyst/Optax not installed.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 3
        self.diff_method = diff_method

        try:
            self.dev = qml.device("lightning.gpu", wires=n_qubits)
            self.backend_name = "lightning.gpu"
        except Exception as e:
            warnings.warn(f"lightning.gpu unavailable for Catalyst ({e}); using lightning.qubit.")
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
            self.backend_name = "lightning.qubit"

        ansatz = build_ansatz(n_qubits, n_layers)

        @qml.qnode(self.dev, diff_method=diff_method)
        def circuit(params, x):
            ansatz(params, x)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        self.optimizer = optax.adam(lr)

        # A single compiled gradient call (apples-to-apples grad timing
        # against strategies 1-4).
        self.qjit_grad = qjit(catalyst.grad(circuit, argnums=0))

        # The whole training STEP (forward + backward + optimizer update)
        # compiled as one executable.
        opt = self.optimizer

        @qjit
        def compiled_train_step(params, opt_state, x):
            grads = catalyst.grad(circuit, argnums=0)(params, x)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        self.compiled_train_step = compiled_train_step

    def gradient(self, params, x):
        return self.qjit_grad(params, x)

    def bench_train_loop(self, params, x, n_iters=50):
        opt_state = self.optimizer.init(params)
        params, opt_state = self.compiled_train_step(params, opt_state, x)
        start = time.perf_counter()
        for _ in range(n_iters):
            params, opt_state = self.compiled_train_step(params, opt_state, x)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms / n_iters


# ============================================================================
# STRATEGY 6: MERGED — QuSim-Sed's cross-graph independence + Catalyst's
# whole-program compilation, via Catalyst's native async QNode dispatch
# ============================================================================

class MergedVQC:
    """
    Combines both ideas for real, instead of just adding their speedups:
      - Catalyst QJIT removes host/Python dispatch overhead by compiling
        the whole gradient computation into one executable.
      - QuSim-Sed's cross-graph scheduling exploits the fact that the 2P
        shifted circuit evaluations of parameter-shift are mathematically
        INDEPENDENT and dispatches them concurrently instead of
        serializing them on one stream.
    Catalyst exposes exactly this second idea natively via
    `qjit(async_qnodes=True)`. Same ansatz/qubit count as every other
    strategy in this file.
    """

    def __init__(self, n_qubits, n_layers, diff_method="parameter-shift"):
        if not CATALYST_AVAILABLE:
            raise RuntimeError("Catalyst/Optax not installed.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 3
        self.diff_method = diff_method

        try:
            self.dev = qml.device("lightning.gpu", wires=n_qubits)
            self.backend_name = "lightning.gpu"
        except Exception as e:
            warnings.warn(f"lightning.gpu unavailable for merged strategy ({e}); using lightning.qubit.")
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
            self.backend_name = "lightning.qubit"

        ansatz = build_ansatz(n_qubits, n_layers)

        @qml.qnode(self.dev, diff_method=diff_method)
        def circuit(params, x):
            ansatz(params, x)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        if diff_method == "adjoint":
            # Strict forward->backward dependency (Table 1/3): no
            # independent shift-copies left for async dispatch to
            # overlap. Merged collapses to plain Catalyst QJIT here —
            # expected, not a bug (mirrors the paper's own finding that
            # cross-graph scheduling only helps the ~35% forward-pass
            # sliver under adjoint differentiation).
            @qjit(async_qnodes=True)
            def merged_grad(params, x):
                return catalyst.grad(circuit, argnums=0)(params, x)
        else:
            shift = jnp.pi / 2
            n_params = self.n_params

            @qjit(async_qnodes=True)
            def merged_grad(params, x):
                grads = []
                for i in range(n_params):
                    p_plus = params.at[i].add(shift)
                    p_minus = params.at[i].add(-shift)
                    v_plus = circuit(p_plus, x)
                    v_minus = circuit(p_minus, x)
                    grads.append((v_plus - v_minus) / 2)
                return jnp.stack(grads)

        self.merged_grad = merged_grad

    def gradient(self, params, x):
        return self.merged_grad(params, x)


# ============================================================================
# BENCHMARK: Table 6 sweep — qubits {4,6,8} x layers {3,5} x diff methods,
# ALL SIX strategies, ALL on the same shared circuit.
# ============================================================================

def benchmark_table6(n_iters_grad=20, n_iters_train=30, n_streams=4, chunk_size=4):
    print("\n" + "=" * 80)
    print("BENCHMARK: TABLE 6 SETUP (qubits {4,6,8}, layers {3,5}) - 6 STRATEGIES")
    print("=" * 80)

    qubit_counts = [4, 6, 8]
    layer_counts = [3, 5]
    diff_methods = ["parameter-shift", "adjoint"]

    results = []

    for n_qubits in qubit_counts:
        for n_layers in layer_counts:
            for diff_method in diff_methods:
                label = f"{n_qubits}Q-{n_layers}L-{diff_method}"
                print(f"\n-- {label} --")

                vqc = FlexibleVQC(n_qubits, n_layers, diff_method=diff_method)
                n_params = vqc.n_params
                key = jax.random.PRNGKey(42)
                params = jax.random.normal(key, (n_params,)) * 0.1
                x = jax.random.normal(key, (n_qubits,))

                def timeit(fn, n_iters=n_iters_grad):
                    warm = fn(params, x)
                    (warm.block_until_ready() if hasattr(warm, "block_until_ready")
                     else tree_map(lambda g: g.block_until_ready(), warm))
                    start = time.perf_counter()
                    for _ in range(n_iters):
                        out = fn(params, x)
                        tree_map(lambda g: g.block_until_ready(), out)
                    return (time.perf_counter() - start) * 1000 / n_iters

                t_seq = timeit(vqc.gradients_sequential)
                t_grad_only = timeit(jit(vqc.gradients_gradient_only))
                t_quantum_only = timeit(lambda p, xx: vqc.gradients_quantum_only(p, xx, n_streams=n_streams))
                t_cross_graph = timeit(lambda p, xx: vqc.gradients_cross_graph(
                    p, xx, n_streams=n_streams, chunk_size=chunk_size))

                row = {
                    "config": label,
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                    "n_params": n_params,
                    "diff_method": diff_method,
                    "Sequential_ms": t_seq,
                    "GradientOnly_ms": t_grad_only,
                    "QuantumOnly_ms": t_quantum_only,
                    "CrossGraph_QuSimSed_ms": t_cross_graph,
                    "GradientOnly_speedup": t_seq / t_grad_only,
                    "QuantumOnly_speedup": t_seq / t_quantum_only,
                    "CrossGraph_speedup": t_seq / t_cross_graph,
                }

                if CATALYST_AVAILABLE:
                    try:
                        cvqc = CatalystVQC(n_qubits, n_layers, diff_method=diff_method)
                        _ = cvqc.gradient(params, x)  # trigger + amortize compilation
                        start = time.perf_counter()
                        for _ in range(n_iters_grad):
                            cvqc.gradient(params, x)
                        t_catalyst_grad = (time.perf_counter() - start) * 1000 / n_iters_grad
                        t_catalyst_train = cvqc.bench_train_loop(params, x, n_iters=n_iters_train)

                        row["Catalyst_grad_ms"] = t_catalyst_grad
                        row["Catalyst_grad_speedup"] = t_seq / t_catalyst_grad
                        row["Catalyst_train_step_ms"] = t_catalyst_train
                    except Exception as e:
                        print(f"    Catalyst strategy failed for {label}: {e}")
                        row["Catalyst_grad_ms"] = np.nan
                        row["Catalyst_grad_speedup"] = np.nan
                        row["Catalyst_train_step_ms"] = np.nan

                    try:
                        mvqc = MergedVQC(n_qubits, n_layers, diff_method=diff_method)
                        _ = mvqc.gradient(params, x)  # trigger compilation
                        start = time.perf_counter()
                        for _ in range(n_iters_grad):
                            mvqc.gradient(params, x)
                        t_merged = (time.perf_counter() - start) * 1000 / n_iters_grad
                        row["Merged_QuSimSed_Catalyst_ms"] = t_merged
                        row["Merged_speedup"] = t_seq / t_merged
                    except Exception as e:
                        print(f"    Merged strategy failed for {label}: {e}")
                        row["Merged_QuSimSed_Catalyst_ms"] = np.nan
                        row["Merged_speedup"] = np.nan
                else:
                    row["Catalyst_grad_ms"] = np.nan
                    row["Catalyst_grad_speedup"] = np.nan
                    row["Catalyst_train_step_ms"] = np.nan
                    row["Merged_QuSimSed_Catalyst_ms"] = np.nan
                    row["Merged_speedup"] = np.nan

                results.append(row)
                print(f"    Sequential:  {t_seq:8.3f} ms")
                print(f"    GradOnly:    {t_grad_only:8.3f} ms  ({row['GradientOnly_speedup']:.2f}x)")
                print(f"    QuantumOnly: {t_quantum_only:8.3f} ms  ({row['QuantumOnly_speedup']:.2f}x)")
                print(f"    CrossGraph:  {t_cross_graph:8.3f} ms  ({row['CrossGraph_speedup']:.2f}x)")
                if not np.isnan(row["Catalyst_grad_ms"]):
                    print(f"    Catalyst:    {row['Catalyst_grad_ms']:8.3f} ms  ({row['Catalyst_grad_speedup']:.2f}x)")
                    print(f"    Catalyst compiled train step: {row['Catalyst_train_step_ms']:8.3f} ms/iter")
                if not np.isnan(row.get("Merged_QuSimSed_Catalyst_ms", np.nan)):
                    print(f"    Merged (QuSim-Sed+Catalyst): {row['Merged_QuSimSed_Catalyst_ms']:8.3f} ms  ({row['Merged_speedup']:.2f}x)")

    return pd.DataFrame(results)


def visualize_table6(df):
    if df.empty:
        print("No results to plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    bar_specs = [
        ("Sequential_ms", "Sequential"),
        ("GradientOnly_ms", "Gradient-Only"),
        ("QuantumOnly_ms", "Quantum-Only"),
        ("CrossGraph_QuSimSed_ms", "Cross-Graph (QuSim-Sed)"),
        ("Catalyst_grad_ms", "Catalyst QJIT"),
        ("Merged_QuSimSed_Catalyst_ms", "Merged (QuSim-Sed+Catalyst)"),
    ]
    n_bars = len(bar_specs)
    width = 0.8 / n_bars

    for diff_method, ax in zip(["parameter-shift", "adjoint"], axes):
        sub = df[df["diff_method"] == diff_method]
        x = np.arange(len(sub))
        for i, (col, label) in enumerate(bar_specs):
            offset = (i - (n_bars - 1) / 2) * width
            ax.bar(x + offset, sub[col], width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["config"], rotation=45, ha="right")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Gradient time — {diff_method}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("merged_table6_all_strategies.png", dpi=150, bbox_inches="tight")
    print("\nSaved: merged_table6_all_strategies.png")
    plt.show()


def main():
    df = benchmark_table6()
    df.to_csv("merged_table6_results.csv", index=False)
    print("\nSaved: merged_table6_results.csv")
    visualize_table6(df)
    print("\n" + "=" * 80)
    print("MERGED BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
