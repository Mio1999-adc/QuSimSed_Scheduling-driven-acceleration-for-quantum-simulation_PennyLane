"""
QuSim-Sed vs Catalyst QJIT — GPU Benchmark (Table 6 experimental setup)
========================================================================

Extends the original graph-split / chunked parameter-shift benchmark with a
fourth strategy: PennyLane **Catalyst** (Quantum Just-In-Time compilation,
https://pennylane.ai/blog/2023/03/introducing-catalyst-quantum-just-in-time-compilation/).

Four strategies are compared, all under the SAME experimental setup used in
the QuSim-Sed paper's evaluation section (Table 6):

    Platform:            Google Colab
    GPU:                 NVIDIA A100 (80GB, 108 SMs)
    Number of Qubits:    {4, 6, 8}
    Number of Layers:    {3, 5}
    Circuit Type:        Variational Quantum Circuit (VQC)
    Differentiation:     Parameter-shift, Adjoint
    Execution Phases:    Forward, Backward

Strategies:
    1. Sequential          - naive PennyLane baseline (no parallel dispatch)
    2. Gradient-Only        - vmap-batched parameter-shift, no cross-graph
                              scheduling (proxy for the paper's "Gradient-Only
                              Parallel" bar)
    3. Cross-Graph (QuSim-Sed) - graph-split scheduling across quantum +
                              gradient graphs (proxy for the paper's "Cross-
                              Graph Full" bar; this repo approximates the
                              Coordinating Layer with chunked/split vmap
                              dispatch since a literal multi-stream CUDA
                              scheduler is outside PennyLane's public API)
    4. Catalyst QJIT        - whole hybrid program (circuit + grad + optax
                              update) compiled ONCE via `qml.qjit`, removing
                              Python dispatch overhead every iteration.
    5. Merged (QuSim-Sed +  - ONE `@qjit(async_qnodes=True)` program: the
       Catalyst)              same whole-program compilation as (4), but the
                              2P shifted circuit calls are left as ordinary
                              independent QNode calls so Catalyst's async
                              runtime dispatches them concurrently across
                              threads/streams — the real mechanism behind
                              (3)'s cross-graph concurrency, now running
                              inside compiled code instead of eager Python.

IMPORTANT: Catalyst and QuSim-Sed optimize DIFFERENT bottlenecks:
  - QuSim-Sed overlaps independent GPU kernels across CUDA streams
    (concurrency).
  - Catalyst removes the Python/host dispatch overhead by compiling the
    entire training step into a single executable (no concurrency by
    itself, but zero per-iteration interpreter overhead).
  They are complementary, not mutually exclusive (see paper's Future Work:
  "extend this scheduling to multi-GPU environments and circuit
  compilation").

Run on Google Colab with an A100 GPU:
    !pip install pennylane pennylane-lightning[gpu] custatevec-cu12 -q
    !pip install pennylane-catalyst optax pandas matplotlib seaborn -q
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pennylane as qml
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.tree_util import tree_map

print("=" * 80)
print("QuSim-Sed vs CATALYST QJIT — TABLE 6 EXPERIMENTAL SETUP")
print("=" * 80)
print(f"PennyLane: {qml.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 80)
print()

# ============================================================================
# Optional imports: Catalyst + Optax (only needed for strategy 4)
# ============================================================================
try:
    import catalyst
    from catalyst import qjit
    import optax
    CATALYST_AVAILABLE = True
except Exception as e:
    CATALYST_AVAILABLE = False
    warnings.warn(
        f"Catalyst/Optax not available ({e}). Strategies 1-3 will still run; "
        f"strategy 4 (Catalyst QJIT) will be skipped. "
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
# Shared VQC ansatz (identical circuit used by every strategy, so timings
# are apples-to-apples)
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
# STRATEGIES 1-3: Sequential / Gradient-Only / Cross-Graph (QuSim-Sed proxy)
# ============================================================================

class FlexibleVQC:
    """Parameter-shift and adjoint gradients under three dispatch strategies."""

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

    # ---- Strategy 2: Gradient-Only parallel (single vmap batch, no split) ----
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

    # ---- Strategy 3: Cross-Graph / QuSim-Sed proxy (chunked + split dispatch,
    #      approximating the Coordinating Layer's multi-stream scheduling) ----
    def gradients_cross_graph(self, params, x=None, n_splits=4, max_chunk=4):
        if self.diff_method == "adjoint":
            # Adjoint has a strict forward-backward dependency (Table 1/3),
            # so cross-graph scheduling only overlaps the ~35% forward-pass
            # sliver; still computed via jax.grad, timed the same way.
            return grad(lambda p: self.circuit(p, x))(params)
        shift = jnp.pi / 2
        n = self.n_params
        per_split = (n + n_splits - 1) // n_splits
        all_grads = []
        for s in range(n_splits):
            start, end = s * per_split, min((s + 1) * per_split, n)
            if start >= n:
                continue
            for c_start in range(start, end, max_chunk):
                c_end = min(c_start + max_chunk, end)
                idx = jnp.arange(c_start, c_end)
                plus = vmap(lambda i: params.at[i].add(shift))(idx)
                minus = vmap(lambda i: params.at[i].add(-shift))(idx)
                vals_p = vmap(lambda p: self.circuit(p, x))(plus)
                vals_m = vmap(lambda p: self.circuit(p, x))(minus)
                all_grads.append((vals_p - vals_m) / 2)
        return jnp.concatenate(all_grads)


# ============================================================================
# STRATEGY 4: Catalyst QJIT — whole train step compiled once
# ============================================================================

class CatalystVQC:
    """
    Compiles circuit + gradient + optimizer update into a SINGLE executable
    with qml.qjit, eliminating per-iteration Python/host dispatch overhead.
    Requires pennylane-catalyst + optax, and a Catalyst-compatible device
    (lightning.qubit or lightning.gpu / lightning.kokkos).
    """

    def __init__(self, n_qubits, n_layers, diff_method="adjoint", lr=0.01):
        if not CATALYST_AVAILABLE:
            raise RuntimeError("Catalyst/Optax not installed.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 3
        self.diff_method = diff_method

        # Catalyst's device support mirrors PennyLane's lightning family.
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

        # A single compiled gradient call (for apples-to-apples grad timing
        # against strategies 1-3).
        self.qjit_grad = qjit(catalyst.grad(circuit, argnums=0))

        # The whole training STEP (forward + backward + optimizer update)
        # compiled as one executable — this is where Catalyst's advantage
        # over eager PennyLane compounds over many iterations.
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
        # Trigger + amortize one-time compilation before timing.
        params, opt_state = self.compiled_train_step(params, opt_state, x)
        start = time.perf_counter()
        for _ in range(n_iters):
            params, opt_state = self.compiled_train_step(params, opt_state, x)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms / n_iters


# ============================================================================
# STRATEGY 5: MERGED — QuSim-Sed's cross-graph independence + Catalyst's
# whole-program compilation, via Catalyst's native async QNode dispatch
# ============================================================================

class MergedVQC:
    """
    Combines both ideas for real, instead of just adding their speedups:

      - Catalyst QJIT removes host/Python dispatch overhead by compiling
        the whole gradient computation into one executable (same as
        CatalystVQC above).
      - QuSim-Sed's cross-graph scheduling exploits the fact that the 2P
        shifted circuit evaluations of parameter-shift are mathematically
        INDEPENDENT (C_j ⊥ C_k, Table 1 of the paper) and dispatches them
        concurrently instead of serializing them on one stream.

    Catalyst exposes exactly this second idea natively via
    `qjit(async_qnodes=True)`: independent QNode calls found in the traced
    program are automatically dispatched across multiple threads/streams
    by the compiled runtime (see PennyLane v0.34 / Catalyst v0.4 release
    notes). So the merged strategy is simply: write the shifted-circuit
    loop as ordinary independent QNode calls, and let ONE
    `@qjit(async_qnodes=True)` program both compile away the Python
    overhead (Catalyst's win) AND concurrently dispatch the independent
    evaluations (QuSim-Sed's win) — a single-pass realization of the
    paper's own "Future Work: extend scheduling to circuit compilation."
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
            # Adjoint enforces a strict forward->backward dependency inside
            # a SINGLE gradient call (Table 1/3): there is nothing
            # independent left for async dispatch to overlap. The merged
            # strategy collapses to plain Catalyst QJIT here — this is
            # expected, not a bug, and matches the paper's own finding that
            # cross-graph scheduling only helps the ~35% forward-pass
            # sliver under adjoint differentiation.
            @qjit(async_qnodes=True)
            def merged_grad(params, x):
                return catalyst.grad(circuit, argnums=0)(params, x)
        else:
            shift = jnp.pi / 2
            n_params = self.n_params

            # All 2P shifted evaluations are independent QNode calls inside
            # ONE compiled program -> Catalyst's async runtime dispatches
            # them concurrently, while qjit removes the per-call Python
            # dispatch overhead that a hand-rolled vmap/loop still pays.
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
# BENCHMARK: Table 6 sweep — qubits {4,6,8} x layers {3,5} x diff methods
# ============================================================================

def benchmark_table6(n_iters_grad=20, n_iters_train=30):
    print("\n" + "=" * 80)
    print("BENCHMARK: TABLE 6 SETUP (qubits {4,6,8}, layers {3,5})")
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
                    fn(params, x).block_until_ready() if hasattr(fn(params, x), "block_until_ready") \
                        else tree_map(lambda g: g.block_until_ready(), fn(params, x))
                    start = time.perf_counter()
                    for _ in range(n_iters):
                        out = fn(params, x)
                        tree_map(lambda g: g.block_until_ready(), out)
                    return (time.perf_counter() - start) * 1000 / n_iters

                t_seq = timeit(vqc.gradients_sequential)
                t_grad_only = timeit(jit(vqc.gradients_gradient_only))
                t_cross_graph = timeit(lambda p, x: vqc.gradients_cross_graph(p, x))

                row = {
                    "config": label,
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                    "n_params": n_params,
                    "diff_method": diff_method,
                    "Sequential_ms": t_seq,
                    "GradientOnly_ms": t_grad_only,
                    "CrossGraph_QuSimSed_ms": t_cross_graph,
                    "GradientOnly_speedup": t_seq / t_grad_only,
                    "CrossGraph_speedup": t_seq / t_cross_graph,
                }

                if CATALYST_AVAILABLE:
                    try:
                        cvqc = CatalystVQC(n_qubits, n_layers, diff_method=diff_method)
                        t_catalyst_grad = None
                        # single compiled gradient call, timed after warmup
                        g0 = cvqc.gradient(params, x)
                        start = time.perf_counter()
                        for _ in range(n_iters_grad):
                            cvqc.gradient(params, x)
                        t_catalyst_grad = (time.perf_counter() - start) * 1000 / n_iters_grad

                        t_catalyst_train = cvqc.bench_train_loop(params, x, n_iters=n_iters_train)

                        row["Catalyst_grad_ms"] = t_catalyst_grad
                        row["Catalyst_grad_speedup"] = t_seq / t_catalyst_grad
                        row["Catalyst_train_step_ms"] = t_catalyst_train
                    except Exception as e:
                        print(f"    ✗ Catalyst strategy failed for {label}: {e}")
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
                        print(f"    ✗ Merged strategy failed for {label}: {e}")
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for diff_method, ax in zip(["parameter-shift", "adjoint"], axes):
        sub = df[df["diff_method"] == diff_method]
        x = np.arange(len(sub))
        width = 0.16
        ax.bar(x - 2 * width, sub["Sequential_ms"], width, label="Sequential")
        ax.bar(x - 1 * width, sub["CrossGraph_QuSimSed_ms"], width, label="Cross-Graph (QuSim-Sed)")
        ax.bar(x + 0 * width, sub["Catalyst_grad_ms"], width, label="Catalyst QJIT")
        ax.bar(x + 1 * width, sub["Merged_QuSimSed_Catalyst_ms"], width, label="Merged (QuSim-Sed+Catalyst)")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["config"], rotation=45, ha="right")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Gradient time — {diff_method}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("table6_catalyst_vs_qusimsed.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved: table6_catalyst_vs_qusimsed.png")
    plt.show()


def main():
    df = benchmark_table6()
    df.to_csv("table6_catalyst_vs_qusimsed_results.csv", index=False)
    print("\n✓ Saved: table6_catalyst_vs_qusimsed_results.csv")
    visualize_table6(df)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
