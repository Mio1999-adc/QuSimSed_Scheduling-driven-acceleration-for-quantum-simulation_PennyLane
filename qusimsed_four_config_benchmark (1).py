"""
QuSim-Sed: Four-Configuration GPU Scheduling Benchmark
=======================================================

Implements and benchmarks the four scheduling configurations discussed
in the QuSim-Sed paper, scaled to 20-30 qubits using
PennyLane-Lightning-GPU (cuQuantum backend):

  1. Sequential
     No parallelism. Monolithic (full-width, fully-entangled) circuit.
     Each of the 2*n_params shifted circuit evaluations is executed one
     at a time, no batching. This is the framework-driven baseline
     described in the paper (T_total ~= T_fwd + sum_j T_grad^(j)).

  2. Gradient-only parallelism
     Parallelizes the GRADIENT GRAPH only. Same monolithic circuit as
     (1), but the +shift/-shift evaluations are batched together via
     jax.vmap (chunked to bound memory). The Quantum Graph itself is
     still evaluated as a single, undivided circuit per shot.

  3. Quantum-only parallelism  (NEW)
     Parallelizes the QUANTUM GRAPH only. The circuit is structurally
     split into independent qubit blocks/subtrees: entangling gates are
     confined within each block (no cross-block CNOTs), so each block
     is a genuinely independent sub-circuit ("Circuit DAG" subtree).
     Each block runs on its own device/QNode and blocks are dispatched
     concurrently (via a thread pool, approximating separate GPU
     streams). Within each block, gradients are still computed with a
     plain sequential loop (no vmap) -> only the Quantum Graph is
     parallelized, not the Gradient Graph.

  4. Cross-graph (full) parallelism  -- QuSim-Sed's proposal
     Combines (2) and (3): the same independent qubit-block subtrees
     from (3), but each block ALSO uses vmap-chunked shift-batching
     internally like (2). Blocks are dispatched concurrently. This
     jointly exploits Quantum Graph and Gradient Graph parallelism, in
     the spirit of QuSim-Sed's Coordinating Layer / cross-graph
     scheduler.

DESIGN NOTE on (3)/(4) vs (1)/(2):
Quantum-only and Cross-graph use a *block-decomposed* circuit (no
entanglement across blocks) so that blocks are truly independent
subtrees that can be dispatched to separate streams/threads. Sequential
and Gradient-only use the original *monolithic* fully-entangled circuit.
This means (3)/(4) are not numerically identical to (1)/(2) — they
represent a different point in the paper's design space (splitting the
Quantum Graph itself), exactly as requested. All four configs still use
the same total qubit/parameter counts for a fair timing comparison.

IMPORTANT MEMORY NOTE:
Statevector size grows as 2^n complex128 values (16 bytes each):
  20 qubits ->    16 MB
  24 qubits ->   256 MB
  28 qubits ->     4 GB
  30 qubits ->    16 GB
Sequential/Gradient-only operate on the FULL n-qubit statevector.
Quantum-only/Cross-graph operate on smaller per-block statevectors
(block width < n_qubits), which is itself one of the practical benefits
of Quantum Graph splitting at large qubit counts.

Run on Google Colab with GPU enabled (A100 recommended for 28-30 qubits).
"""

# ============================================================================
# SETUP
# ============================================================================
"""
!pip install pennylane pandas matplotlib seaborn -q
!pip install custatevec-cu12 -q
!pip install pennylane-lightning[gpu] -q
# lightning.gpu requires an NVIDIA GPU + cuQuantum (custatevec). If this
# install fails or the device can't be created, the code below falls back
# to lightning.qubit (CPU) automatically and prints a warning.
"""

import pennylane as qml
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.tree_util import tree_map
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from concurrent.futures import ThreadPoolExecutor

print("=" * 80)
print("QuSim-Sed FOUR-CONFIG SCHEDULING BENCHMARK (16-30 QUBITS)")
print("=" * 80)
print(f"PennyLane: {qml.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print("=" * 80)
print()


# ============================================================================
# DEVICE SELECTION: lightning.gpu with safe fallback
# ============================================================================

def make_device(n_qubits):
    """
    Try lightning.gpu first (cuQuantum-accelerated statevector sim).
    Fall back to lightning.qubit (fast CPU) if GPU device unavailable.
    Falls back further to default.qubit only as last resort.
    """
    try:
        dev = qml.device('lightning.gpu', wires=n_qubits)
        return dev, 'lightning.gpu'
    except Exception as e1:
        try:
            dev = qml.device('lightning.qubit', wires=n_qubits)
            warnings.warn(
                f"lightning.gpu unavailable ({e1}); falling back to lightning.qubit (CPU). "
                f"Timings will NOT reflect GPU performance."
            )
            return dev, 'lightning.qubit'
        except Exception as e2:
            dev = qml.device('default.qubit', wires=n_qubits)
            warnings.warn(
                f"lightning.qubit also unavailable ({e2}); falling back to default.qubit. "
                f"This will be SLOW and memory-hungry above ~20 qubits."
            )
            return dev, 'default.qubit'


_probe_dev, _probe_name = make_device(4)
print(f"Selected simulator backend: {_probe_name}")
if _probe_name != 'lightning.gpu':
    print("WARNING: lightning.gpu not active. Results below will NOT reflect")
    print("   real GPU-accelerated statevector performance. Install pennylane-lightning[gpu]")
    print("   and custatevec-cu12, and confirm you're on an NVIDIA GPU runtime.")
print("=" * 80)
print()


# ============================================================================
# MEMORY HELPERS
# ============================================================================

def estimate_statevector_memory_gb(n_qubits, n_copies=1):
    """Estimate memory in GB for n_copies statevectors of n_qubits."""
    bytes_per_amplitude = 16  # complex128
    return (2 ** n_qubits) * bytes_per_amplitude * n_copies / (1024 ** 3)


def safe_chunk_size(n_qubits, memory_budget_gb=4.0):
    """
    Pick a chunk_size such that 2*chunk_size statevectors fit within
    memory_budget_gb. Ensures at least 1.
    """
    per_state_gb = estimate_statevector_memory_gb(n_qubits, 1)
    if per_state_gb <= 0:
        return 8
    max_copies = max(1, int(memory_budget_gb / (2 * per_state_gb)))
    return max(1, min(max_copies, 8))  # cap at 8 for sanity


def partition_qubits(n_qubits, n_blocks):
    """
    Split n_qubits into n_blocks contiguous, near-equal-sized wire
    ranges. Returns a list of (start, end) tuples (end exclusive).
    Used to build the independent Quantum Graph subtrees for the
    Quantum-only and Cross-graph configs.
    """
    n_blocks = max(1, min(n_blocks, n_qubits))
    base = n_qubits // n_blocks
    rem = n_qubits % n_blocks
    blocks = []
    start = 0
    for i in range(n_blocks):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue
        blocks.append((start, start + size))
        start += size
    return blocks


def default_n_blocks(n_qubits, target_block_size=6):
    """Choose a block count that keeps each block's statevector small."""
    return max(1, round(n_qubits / target_block_size))


# ============================================================================
# FOUR-CONFIG VQC
# ============================================================================

class FourConfigVQC:
    """
    Builds both the monolithic circuit (used by Sequential and
    Gradient-only) and the block-decomposed circuit (used by
    Quantum-only and Cross-graph), and exposes one gradient method per
    configuration.

    diff_method: 'parameter-shift' or 'adjoint'
      - parameter-shift: 2*n_params independent +/-shift circuit evals
        (Table 1/3 in the paper). Sequential loops over them one at a
        time; Gradient-only batches them with vmap; Quantum-only /
        Cross-graph additionally split the circuit into independent
        qubit blocks.
      - adjoint: single reverse-mode pass per circuit (jax.grad through
        the QNode's native adjoint rule). There are no independent
        shift-copies to batch, so Gradient-only collapses to the same
        single pass as Sequential (matching the paper's Table 1: "strict
        forward-backward dependency"). Quantum-only / Cross-graph still
        benefit from splitting the circuit into independent blocks that
        can be dispatched concurrently.
    """

    def __init__(self, n_qubits, n_layers, n_blocks=None, diff_method='parameter-shift'):
        assert diff_method in ('parameter-shift', 'adjoint'), \
            "diff_method must be 'parameter-shift' or 'adjoint'"
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 3
        self.n_blocks = n_blocks if n_blocks is not None else default_n_blocks(n_qubits)
        self.blocks = partition_qubits(n_qubits, self.n_blocks)
        self.diff_method = diff_method

        # ---- Monolithic circuit (full-width, fully entangled) --------
        self.mono_dev, self.mono_backend = make_device(n_qubits)

        @qml.qnode(self.mono_dev, interface='jax', diff_method=diff_method)
        def mono_circuit(params, x=None):
            if x is not None:
                for i in range(n_qubits):
                    qml.RY(x[i % len(x)], wires=i)
            idx = 0
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RX(params[idx], wires=qubit); idx += 1
                    qml.RY(params[idx], wires=qubit); idx += 1
                    qml.RZ(params[idx], wires=qubit); idx += 1
                for qubit in range(n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
            return qml.expval(qml.PauliZ(0))

        self.mono_circuit = mono_circuit

        # ---- Block-decomposed circuits (independent subtrees) --------
        # Global params vector is laid out block-by-block: block b owns
        # a contiguous slice of size (block_width * n_layers * 3), using
        # its own local layer-major ordering (RX,RY,RZ per qubit per
        # layer), with entangling CNOTs confined to that block's wires.
        self.block_specs = []  # (dev, backend_name, qnode, param_slice, x_slice, width)
        param_cursor = 0
        for (start, end) in self.blocks:
            width = end - start
            n_block_params = width * n_layers * 3
            dev, backend_name = make_device(width)

            def make_block_circuit(width=width, n_layers=n_layers):
                @qml.qnode(dev, interface='jax', diff_method=diff_method)
                def block_circuit(params, x=None):
                    if x is not None:
                        for i in range(width):
                            qml.RY(x[i % len(x)], wires=i)
                    idx = 0
                    for layer in range(n_layers):
                        for qubit in range(width):
                            qml.RX(params[idx], wires=qubit); idx += 1
                            qml.RY(params[idx], wires=qubit); idx += 1
                            qml.RZ(params[idx], wires=qubit); idx += 1
                        for qubit in range(width):
                            qml.CNOT(wires=[qubit, (qubit + 1) % width])
                    return qml.expval(qml.PauliZ(0))
                return block_circuit

            qnode = make_block_circuit()
            self.block_specs.append({
                'dev': dev,
                'backend': backend_name,
                'qnode': qnode,
                'param_start': param_cursor,
                'param_end': param_cursor + n_block_params,
                'wire_start': start,
                'wire_end': end,
                'width': width,
                'n_block_params': n_block_params,
            })
            param_cursor += n_block_params

    # ------------------------------------------------------------------
    # FORWARD PASS ONLY — single unshifted circuit evaluation. Used both
    # standalone and to build the Figure-3-style forward/backward
    # breakdown (see time_forward_backward_split() below).
    # ------------------------------------------------------------------
    def forward_only(self, params, x=None):
        return self.mono_circuit(params, x)

    def forward_only_for_config(self, config_name, params, x=None):
        """
        Forward-Phase reference matched to the circuit topology actually
        used by `config_name`'s gradient computation:
          - Sequential / Gradient-only  -> one pass on the monolithic,
            full-width circuit (they never split the Quantum Graph).
          - Quantum-only / Cross-graph  -> one pass PER independent block,
            dispatched concurrently (they DO split the Quantum Graph into
            block subtrees, each with its own smaller statevector).
        Using a mismatched reference (e.g. always the full-width circuit)
        would make a "before vs. after scheduling" forward/backward
        breakdown misleading once the block-parallel configs get fast
        enough that their total time is smaller than a full-width
        forward pass.
        """
        if config_name in ('Sequential', 'Gradient-only'):
            return self.mono_circuit(params, x)

        def _block_forward(spec):
            s, e = spec['param_start'], spec['param_end']
            block_params = params[s:e]
            block_x = None if x is None else x[spec['wire_start']:spec['wire_end']]
            return spec['qnode'](block_params, block_x)

        with ThreadPoolExecutor(max_workers=len(self.block_specs)) as pool:
            futures = [pool.submit(_block_forward, spec) for spec in self.block_specs]
            results = [f.result() for f in futures]
        return jnp.stack(results)

    # ------------------------------------------------------------------
    # 1) SEQUENTIAL — no parallelism, monolithic circuit
    # ------------------------------------------------------------------
    def gradients_sequential(self, params, x=None):
        if self.diff_method == 'adjoint':
            # Single reverse-mode pass; no shift copies exist to loop over.
            return jax.grad(lambda p: self.mono_circuit(p, x))(params)
        shift = jnp.pi / 2
        grads = []
        for i in range(self.n_params):
            p_plus = params.at[i].add(shift)
            p_minus = params.at[i].add(-shift)
            v_plus = self.mono_circuit(p_plus, x)
            v_minus = self.mono_circuit(p_minus, x)
            grads.append((v_plus - v_minus) / 2)
        return jnp.array(grads)

    # ------------------------------------------------------------------
    # 2) GRADIENT-ONLY parallelism — monolithic circuit, vmap-batched
    #    shift evaluations (chunked to bound memory).
    #    Under Adjoint there are no independent shift-copies to batch
    #    (Table 1: "strict forward-backward dependency"), so this
    #    collapses to the same single reverse-mode pass as Sequential.
    # ------------------------------------------------------------------
    def gradients_gradient_only(self, params, x=None, chunk_size=4):
        if self.diff_method == 'adjoint':
            return self.gradients_sequential(params, x)
        shift = jnp.pi / 2
        n = self.n_params
        n_chunks = (n + chunk_size - 1) // chunk_size
        all_grads = []
        for c in range(n_chunks):
            s, e = c * chunk_size, min((c + 1) * chunk_size, n)
            idxs = jnp.arange(s, e)
            plus = vmap(lambda i: params.at[i].add(shift))(idxs)
            minus = vmap(lambda i: params.at[i].add(-shift))(idxs)
            vp = vmap(lambda p: self.mono_circuit(p, x))(plus)
            vm = vmap(lambda p: self.mono_circuit(p, x))(minus)
            all_grads.append((vp - vm) / 2)
        return jnp.concatenate(all_grads)

    # ------------------------------------------------------------------
    # 3) QUANTUM-ONLY parallelism — independent qubit-block subtrees
    #    dispatched concurrently; sequential (no vmap) gradient loop
    #    WITHIN each block
    # ------------------------------------------------------------------
    def _block_gradient_sequential(self, spec, params, x):
        s, e = spec['param_start'], spec['param_end']
        block_params = params[s:e]
        block_x = None if x is None else x[spec['wire_start']:spec['wire_end']]
        if self.diff_method == 'adjoint':
            # Single reverse-mode pass through this block's QNode.
            return jax.grad(lambda p: spec['qnode'](p, block_x))(block_params)
        shift = jnp.pi / 2
        grads = []
        for i in range(spec['n_block_params']):
            p_plus = block_params.at[i].add(shift)
            p_minus = block_params.at[i].add(-shift)
            v_plus = spec['qnode'](p_plus, block_x)
            v_minus = spec['qnode'](p_minus, block_x)
            grads.append((v_plus - v_minus) / 2)
        return jnp.array(grads)

    def gradients_quantum_only(self, params, x=None):
        with ThreadPoolExecutor(max_workers=len(self.block_specs)) as pool:
            futures = [
                pool.submit(self._block_gradient_sequential, spec, params, x)
                for spec in self.block_specs
            ]
            block_results = [f.result() for f in futures]
        return jnp.concatenate(block_results)

    # ------------------------------------------------------------------
    # 4) CROSS-GRAPH (full) parallelism — independent qubit-block
    #    subtrees dispatched concurrently, EACH using vmap-chunked
    #    shift-batching internally (QuSim-Sed's proposal)
    # ------------------------------------------------------------------
    def _block_gradient_chunked(self, spec, params, x, chunk_size):
        if self.diff_method == 'adjoint':
            # No shift-copies to batch under Adjoint; the cross-graph
            # benefit here comes purely from concurrent block dispatch,
            # so this is identical to the quantum-only block gradient.
            return self._block_gradient_sequential(spec, params, x)
        shift = jnp.pi / 2
        s, e = spec['param_start'], spec['param_end']
        block_params = params[s:e]
        block_x = None if x is None else x[spec['wire_start']:spec['wire_end']]
        n = spec['n_block_params']
        n_chunks = (n + chunk_size - 1) // chunk_size
        all_grads = []
        for c in range(n_chunks):
            cs, ce = c * chunk_size, min((c + 1) * chunk_size, n)
            idxs = jnp.arange(cs, ce)
            plus = vmap(lambda i: block_params.at[i].add(shift))(idxs)
            minus = vmap(lambda i: block_params.at[i].add(-shift))(idxs)
            vp = vmap(lambda p: spec['qnode'](p, block_x))(plus)
            vm = vmap(lambda p: spec['qnode'](p, block_x))(minus)
            all_grads.append((vp - vm) / 2)
        return jnp.concatenate(all_grads)

    def gradients_cross_graph(self, params, x=None, chunk_size=4):
        with ThreadPoolExecutor(max_workers=len(self.block_specs)) as pool:
            futures = [
                pool.submit(self._block_gradient_chunked, spec, params, x, chunk_size)
                for spec in self.block_specs
            ]
            block_results = [f.result() for f in futures]
        return jnp.concatenate(block_results)

    # ------------------------------------------------------------------
    # Dispatch helper so callers can select a config by name.
    # ------------------------------------------------------------------
    def gradients(self, config_name, params, x=None, chunk_size=4):
        fn_map = {
            'Sequential': lambda p: self.gradients_sequential(p, x),
            'Gradient-only': lambda p: self.gradients_gradient_only(p, x, chunk_size=chunk_size),
            'Quantum-only': lambda p: self.gradients_quantum_only(p, x),
            'Cross-graph': lambda p: self.gradients_cross_graph(p, x, chunk_size=chunk_size),
        }
        return fn_map[config_name](params)


# ============================================================================
# FORWARD / BACKWARD TIME BREAKDOWN  (Figure-3-style, before vs after
# QuSim-Sed's cross-graph scheduling is applied)
# ============================================================================

def time_forward_backward_split(vqc, params, x, config_name, chunk_size=4,
                                 n_warmup=2, n_iter=5):
    """
    Separately times:
      - the Forward Phase: one plain (unshifted) circuit evaluation
        (Fig. 2's "Forward" box), on whichever circuit topology the given
        config actually uses (see forward_only_for_config()).
      - the Backward Phase: the FULL gradient computation for the given
        scheduling config (Sequential / Gradient-only / Quantum-only /
        Cross-graph), which under parameter-shift already subsumes many
        forward-like circuit evals, and under adjoint is the reverse-mode
        pass.

    Mirrors the paper's Figure 3 (Time Breakdown: Forward vs Backward)
    for the *conventional* (Sequential) configuration, and additionally
    reports the same breakdown for Cross-graph so the effect of
    QuSim-Sed's scheduling on the forward/backward split can be seen
    directly ("breakdown after acceleration"). The forward reference is
    matched to each config's own circuit topology via
    forward_only_for_config() (full-width for Sequential/Gradient-only,
    concurrent per-block for Quantum-only/Cross-graph) so "before" and
    "after" are an apples-to-apples comparison rather than both being
    measured against the full monolithic circuit.
    """
    # ---- Forward Phase timing (matched to this config's topology) ----
    for _ in range(n_warmup):
        tree_map(lambda r: r.block_until_ready(),
                 vqc.forward_only_for_config(config_name, params, x))
    fwd_times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        r = vqc.forward_only_for_config(config_name, params, x)
        tree_map(lambda rr: rr.block_until_ready(), r)
        fwd_times.append((time.perf_counter() - start) * 1000)
    fwd_ms = float(np.mean(fwd_times))

    # ---- Full gradient computation timing for this config ----
    for _ in range(n_warmup):
        tree_map(lambda g: g.block_until_ready(), vqc.gradients(config_name, params, x, chunk_size))
    total_times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        g = vqc.gradients(config_name, params, x, chunk_size)
        tree_map(lambda gg: gg.block_until_ready(), g)
        total_times.append((time.perf_counter() - start) * 1000)
    total_ms = float(np.mean(total_times))

    # Backward Phase = total gradient-computation time attributable to
    # differentiation beyond one reference forward pass. Floored at 1% of
    # total so the bar never goes to zero/negative from timing noise.
    bwd_ms = max(total_ms - fwd_ms, total_ms * 0.01)
    # Re-normalize so forward + backward == total_ms exactly (for a clean
    # stacked-bar chart, matching how Fig. 3 always sums to 100%).
    fwd_ms_norm = max(total_ms - bwd_ms, total_ms * 0.005)
    return {
        'config': config_name,
        'forward_ms': fwd_ms_norm,
        'backward_ms': total_ms - fwd_ms_norm,
        'total_ms': total_ms,
        'forward_pct': 100.0 * fwd_ms_norm / total_ms if total_ms > 0 else np.nan,
        'backward_pct': 100.0 * (total_ms - fwd_ms_norm) / total_ms if total_ms > 0 else np.nan,
    }


def benchmark_forward_backward_breakdown(qubit_layer_pairs=((10, 3), (15, 3), (18, 3)),
                                          diff_methods=('parameter-shift', 'adjoint'),
                                          chunk_size=4):
    """
    Reproduces Figure 3 (Sequential, "before") and extends it with the
    same breakdown for Cross-graph ("after" QuSim-Sed's scheduling is
    applied), for both differentiation methods, across a few qubit/layer
    configurations. Returns a tidy DataFrame.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK: FORWARD/BACKWARD BREAKDOWN — BEFORE vs AFTER SCHEDULING")
    print("=" * 80)
    rows = []
    for method in diff_methods:
        for (n_qubits, n_layers) in qubit_layer_pairs:
            n_blocks = default_n_blocks(n_qubits)
            try:
                vqc = FourConfigVQC(n_qubits=n_qubits, n_layers=n_layers,
                                     n_blocks=n_blocks, diff_method=method)
            except Exception as e:
                print(f"  Skipping {method} {n_qubits}Q-{n_layers}L - device creation failed: {e}")
                continue
            key = jax.random.PRNGKey(42)
            params = jax.random.normal(key, (vqc.n_params,)) * 0.1
            x = jax.random.normal(key, (n_qubits,))
            for config_name in ('Sequential', 'Cross-graph'):
                try:
                    res = time_forward_backward_split(vqc, params, x, config_name,
                                                        chunk_size=chunk_size)
                except Exception as e:
                    print(f"    {method:16s} {n_qubits:2d}Q-{n_layers}L {config_name:12s} FAILED ({e})")
                    continue
                res.update({'method': method, 'n_qubits': n_qubits, 'n_layers': n_layers})
                rows.append(res)
                print(f"    {method:16s} {n_qubits:2d}Q-{n_layers}L {config_name:12s} "
                      f"fwd={res['forward_pct']:5.1f}%  bwd={res['backward_pct']:5.1f}%  "
                      f"total={res['total_ms']:8.2f} ms")
    return pd.DataFrame(rows)


def visualize_forward_backward_breakdown(df):
    """Stacked horizontal bars, Sequential ('before') vs Cross-graph
    ('after'), one panel per differentiation method — the console/paper
    Figure 3 style, extended to show the effect of QuSim-Sed's
    scheduling on the forward/backward split."""
    if df.empty:
        print("No forward/backward breakdown data to visualize.")
        return
    methods = sorted(df['method'].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(8 * len(methods), 5), squeeze=False)
    axes = axes[0]
    for ax, method in zip(axes, methods):
        sub = df[df['method'] == method].copy()
        sub['label'] = sub.apply(
            lambda r: f"{int(r['n_qubits'])}Q-{int(r['n_layers'])}L\n{r['config']}", axis=1)
        sub = sub.sort_values(['n_qubits', 'config'])
        y = np.arange(len(sub))
        ax.barh(y, sub['forward_pct'], color='#8b6fd6', label='Forward')
        ax.barh(y, sub['backward_pct'], left=sub['forward_pct'], color='#e8b04b', label='Backward')
        for i, (_, r) in enumerate(sub.iterrows()):
            if r['forward_pct'] > 6:
                ax.text(r['forward_pct'] / 2, i, f"{r['forward_pct']:.0f}%",
                        va='center', ha='center', fontsize=8, color='white')
            ax.text(r['forward_pct'] + r['backward_pct'] / 2, i, f"{r['backward_pct']:.0f}%",
                    va='center', ha='center', fontsize=8, color='#222')
        ax.set_yticks(y)
        ax.set_yticklabels(sub['label'], fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Share of gradient-computation time (%)', fontweight='bold')
        ax.set_title(f'Forward vs Backward — {method}\n(Sequential = before, Cross-graph = after)',
                     fontweight='bold', fontsize=11)
        ax.legend(loc='lower right')
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('forward_backward_breakdown.png', dpi=150, bbox_inches='tight')
    print("\nSaved: forward_backward_breakdown.png")
    plt.show()


# ============================================================================
# BENCHMARK: DIFFERENTIATION METHOD COMPARISON ACROSS QUBITS AND LAYERS
# ============================================================================

def benchmark_method_vs_qubits(n_layers=3, qubit_counts=(10, 12, 14, 16, 18, 20),
                                diff_methods=('parameter-shift', 'adjoint'),
                                chunk_size=4, n_warmup=2, n_iter=3):
    """Sweeps qubit count at fixed depth, for both differentiation
    methods, recording Sequential/Cross-graph time and speedup."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: PARAMETER-SHIFT vs ADJOINT ACROSS QUBITS (L={n_layers})")
    print("=" * 80)
    rows = []
    for method in diff_methods:
        for n_qubits in qubit_counts:
            n_blocks = default_n_blocks(n_qubits)
            try:
                vqc = FourConfigVQC(n_qubits=n_qubits, n_layers=n_layers,
                                     n_blocks=n_blocks, diff_method=method)
            except Exception as e:
                print(f"  Skipping {method} {n_qubits}Q: device creation failed: {e}")
                continue
            row = _time_seq_vs_cross(vqc, n_qubits, n_layers, method, chunk_size, n_warmup, n_iter)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def benchmark_method_vs_layers(n_qubits=16, layer_counts=(1, 2, 3, 4, 5, 6),
                                diff_methods=('parameter-shift', 'adjoint'),
                                chunk_size=4, n_warmup=2, n_iter=3):
    """Sweeps circuit depth at fixed qubit count, for both differentiation
    methods, recording Sequential/Cross-graph time and speedup."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: PARAMETER-SHIFT vs ADJOINT ACROSS LAYERS (Q={n_qubits})")
    print("=" * 80)
    rows = []
    n_blocks = default_n_blocks(n_qubits)
    for method in diff_methods:
        for n_layers in layer_counts:
            try:
                vqc = FourConfigVQC(n_qubits=n_qubits, n_layers=n_layers,
                                     n_blocks=n_blocks, diff_method=method)
            except Exception as e:
                print(f"  Skipping {method} L={n_layers}: device creation failed: {e}")
                continue
            row = _time_seq_vs_cross(vqc, n_qubits, n_layers, method, chunk_size, n_warmup, n_iter)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def _time_seq_vs_cross(vqc, n_qubits, n_layers, method, chunk_size, n_warmup, n_iter):
    key = jax.random.PRNGKey(42)
    params = jax.random.normal(key, (vqc.n_params,)) * 0.1
    x = jax.random.normal(key, (n_qubits,))
    row = {'method': method, 'n_qubits': n_qubits, 'n_layers': n_layers, 'n_params': vqc.n_params}
    try:
        for name in ('Sequential', 'Cross-graph'):
            for _ in range(n_warmup):
                tree_map(lambda g: g.block_until_ready(), vqc.gradients(name, params, x, chunk_size))
            timings = []
            for _ in range(n_iter):
                start = time.perf_counter()
                g = vqc.gradients(name, params, x, chunk_size)
                tree_map(lambda gg: gg.block_until_ready(), g)
                timings.append((time.perf_counter() - start) * 1000)
            row[name] = float(np.mean(timings))
        row['speedup'] = row['Sequential'] / row['Cross-graph'] if row['Cross-graph'] > 0 else np.nan
        print(f"    {method:16s} {n_qubits:2d}Q-{n_layers}L: Seq={row['Sequential']:8.2f} ms  "
              f"Cross={row['Cross-graph']:8.2f} ms  speedup={row['speedup']:.2f}x")
        return row
    except Exception as e:
        print(f"    {method:16s} {n_qubits:2d}Q-{n_layers}L: FAILED ({e})")
        return None


def visualize_method_comparison(df_q, df_l):
    """Two-panel comparison: Cross-graph speedup vs Sequential, for
    Parameter-Shift vs Adjoint, swept across (a) qubit count and
    (b) layer count."""
    if df_q.empty and df_l.empty:
        print("No method-comparison data to visualize.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {'parameter-shift': '#5b8def', 'adjoint': '#e67e22'}

    ax = axes[0]
    for method in df_q['method'].unique() if not df_q.empty else []:
        sub = df_q[df_q['method'] == method].sort_values('n_qubits')
        ax.plot(sub['n_qubits'], sub['speedup'], marker='o', linewidth=2,
                label=method, color=colors.get(method, '#999'))
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='No speedup')
    ax.set_xlabel('Number of Qubits', fontweight='bold')
    ax.set_ylabel('Cross-graph Speedup vs Sequential (x)', fontweight='bold')
    ax.set_title('Method Comparison Across Qubit Count', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for method in df_l['method'].unique() if not df_l.empty else []:
        sub = df_l[df_l['method'] == method].sort_values('n_layers')
        ax.plot(sub['n_layers'], sub['speedup'], marker='D', linewidth=2,
                label=method, color=colors.get(method, '#999'))
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='No speedup')
    ax.set_xlabel('Number of Layers', fontweight='bold')
    ax.set_ylabel('Cross-graph Speedup vs Sequential (x)', fontweight='bold')
    ax.set_title('Method Comparison Across Circuit Depth', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('method_comparison_qubits_layers.png', dpi=150, bbox_inches='tight')
    print("\nSaved: method_comparison_qubits_layers.png")
    plt.show()


# ============================================================================
# BENCHMARK: SPEEDUP + PARAMETER SCALING ACROSS QUBIT COUNT (ALL CONFIGS)
# ============================================================================

def benchmark_speedup_scaling_qubits(n_layers=3,
                                      qubit_counts=(10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
                                      diff_methods=('parameter-shift', 'adjoint'),
                                      chunk_size=4, n_warmup=2, n_iter=3,
                                      sequential_qubit_limit=24):
    """
    Sweeps qubit count at a fixed layer depth, for each differentiation
    method, and records the speedup of Gradient-only / Quantum-only /
    Cross-graph vs Sequential *together with* the parameter count at that
    qubit count (n_params = n_layers * n_qubits * 3, Section 3.1.1 / Table
    5). This isolates how speedup scales as parameter count grows purely
    from adding qubits (layer depth held fixed), complementing
    benchmark_method_vs_layers() which instead grows parameters by adding
    layers at fixed qubit count.

    Sequential is skipped above `sequential_qubit_limit` qubits (looping
    2*n_params individual shifted circuits one at a time becomes too slow
    to run repeatedly at large qubit counts); above that limit, raw times
    for the three parallel configs are still collected but speedups vs
    Sequential are reported as NaN for those rows.

    Returns a tidy DataFrame with one row per (method, n_qubits), columns:
      method, n_qubits, n_layers, n_params,
      Sequential, Gradient-only, Quantum-only, Cross-graph  (ms, may be NaN),
      Gradient-only_speedup, Quantum-only_speedup, Cross-graph_speedup
    """
    print("\n" + "=" * 80)
    print("BENCHMARK: SPEEDUP vs PARAMETER SCALING ACROSS QUBIT COUNT (ALL CONFIGS)")
    print("=" * 80)
    print(f"Fixed layers L={n_layers}. Parameters grow as n_params = L * n_qubits * 3.")
    print()

    rows = []
    for method in diff_methods:
        print(f"  -- {method} --")
        for n_qubits in qubit_counts:
            n_blocks = default_n_blocks(n_qubits)
            try:
                vqc = FourConfigVQC(n_qubits=n_qubits, n_layers=n_layers,
                                     n_blocks=n_blocks, diff_method=method)
            except Exception as e:
                print(f"    Skipping {n_qubits}Q: device creation failed: {e}")
                continue

            n_params = vqc.n_params
            key = jax.random.PRNGKey(42)
            params = jax.random.normal(key, (n_params,)) * 0.1
            x = jax.random.normal(key, (n_qubits,))

            run_sequential = n_qubits <= sequential_qubit_limit
            configs = (['Sequential'] if run_sequential else []) + \
                      ['Gradient-only', 'Quantum-only', 'Cross-graph']

            row = {'method': method, 'n_qubits': n_qubits, 'n_layers': n_layers,
                   'n_params': n_params}
            for name in configs:
                try:
                    for _ in range(n_warmup):
                        tree_map(lambda g: g.block_until_ready(),
                                 vqc.gradients(name, params, x, chunk_size))
                    timings = []
                    for _ in range(n_iter):
                        start = time.perf_counter()
                        g = vqc.gradients(name, params, x, chunk_size)
                        tree_map(lambda gg: gg.block_until_ready(), g)
                        timings.append((time.perf_counter() - start) * 1000)
                    row[name] = float(np.mean(timings))
                except Exception as e:
                    print(f"    {name:14s} FAILED at {n_qubits}Q ({e})")
                    row[name] = np.nan
            if not run_sequential:
                row['Sequential'] = np.nan

            base = row.get('Sequential', np.nan)
            for name in ['Gradient-only', 'Quantum-only', 'Cross-graph']:
                if not np.isnan(base) and name in row and not np.isnan(row[name]) and row[name] > 0:
                    row[f'{name}_speedup'] = base / row[name]
                else:
                    row[f'{name}_speedup'] = np.nan

            sp_str = "  ".join(
                f"{name}={row[f'{name}_speedup']:.2f}x" if not np.isnan(row[f'{name}_speedup']) else f"{name}=n/a"
                for name in ['Gradient-only', 'Quantum-only', 'Cross-graph']
            )
            print(f"    {n_qubits:2d}Q ({n_params:3d} params): {sp_str}")

            rows.append(row)
    return pd.DataFrame(rows)


def print_speedup_scaling_table(df):
    """Prints a plain-text Qubits | Params | speedup-per-config table, one
    block per differentiation method, sorted by qubit count."""
    if df.empty:
        print("No speedup-scaling data collected.")
        return
    print("\n" + "=" * 80)
    print("TABLE: SPEEDUP vs PARAMETER SCALING (ALL CONFIGS)")
    print("=" * 80)
    for method in sorted(df['method'].unique()):
        sub = df[df['method'] == method].sort_values('n_qubits')
        print(f"\n  {method}")
        print(f"  {'Qubits':>7s}  {'Params':>7s}  {'Gradient-only':>14s}  {'Quantum-only':>13s}  {'Cross-graph':>12s}")
        for _, r in sub.iterrows():
            def fmt_sp(name):
                v = r.get(f'{name}_speedup', np.nan)
                return f"{v:.2f}x" if not pd.isna(v) else "n/a"
            print(f"  {int(r['n_qubits']):7d}  {int(r['n_params']):7d}  "
                  f"{fmt_sp('Gradient-only'):>14s}  {fmt_sp('Quantum-only'):>13s}  {fmt_sp('Cross-graph'):>12s}")


def visualize_speedup_scaling(df):
    """Two-panel plot (one per differentiation method) of speedup vs
    Sequential for all three parallel configs, plotted against parameter
    count (which grows with qubit count at fixed layers); qubit count is
    annotated at each marker so both axes of scaling are visible at once."""
    if df.empty:
        print("No speedup-scaling data to visualize.")
        return
    methods = sorted(df['method'].unique())
    colors = {'Gradient-only': '#3498db', 'Quantum-only': '#e67e22', 'Cross-graph': '#2ecc71'}
    fig, axes = plt.subplots(1, len(methods), figsize=(8 * len(methods), 5.5), squeeze=False)
    axes = axes[0]
    for ax, method in zip(axes, methods):
        sub = df[df['method'] == method].sort_values('n_params')
        for name in ['Gradient-only', 'Quantum-only', 'Cross-graph']:
            col = f'{name}_speedup'
            if col in sub.columns:
                ax.plot(sub['n_params'], sub[col], marker='o', linewidth=2,
                        label=name, color=colors[name])
        # Annotate qubit count at each Cross-graph point for readability.
        if 'Cross-graph_speedup' in sub.columns:
            for _, r in sub.iterrows():
                v = r.get('Cross-graph_speedup', np.nan)
                if not pd.isna(v):
                    ax.annotate(f"{int(r['n_qubits'])}Q", (r['n_params'], v),
                                textcoords="offset points", xytext=(4, 6), fontsize=8, color='#666')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='No speedup')
        ax.set_xlabel('Number of Parameters (grown via qubit count, fixed layers)', fontweight='bold')
        ax.set_ylabel('Speedup vs Sequential (x)', fontweight='bold')
        ax.set_title(f'Speedup vs Parameter Scaling — {method}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('speedup_vs_qubit_parameter_scaling.png', dpi=150, bbox_inches='tight')
    print("\nSaved: speedup_vs_qubit_parameter_scaling.png")
    plt.show()


# ============================================================================
# BENCHMARK: QUBIT SCALING 16 -> 30, ALL FOUR CONFIGS
# ============================================================================

def benchmark_four_configs_qubit_scaling():
    print("\n" + "=" * 80)
    print("BENCHMARK: FOUR CONFIGS ACROSS QUBITS (16 -> 30)")
    print("=" * 80)
    print("Configs: Sequential | Gradient-only | Quantum-only | Cross-graph")
    print()

    n_layers = 3
    qubit_counts = [16, 18, 20, 22, 24, 26, 28, 30]
    results = []

    for n_qubits in qubit_counts:
        mem_per_state = estimate_statevector_memory_gb(n_qubits)
        chunk = safe_chunk_size(n_qubits, memory_budget_gb=4.0)
        n_blocks = default_n_blocks(n_qubits)

        print(f"  Qubits: {n_qubits:2d} | State: {mem_per_state:8.3f} GB/copy | "
              f"chunk_size={chunk} | n_blocks={n_blocks}")

        try:
            vqc = FourConfigVQC(n_qubits=n_qubits, n_layers=n_layers, n_blocks=n_blocks)
        except Exception as e:
            print(f"    Skipping {n_qubits} qubits - device creation failed: {e}")
            continue

        n_params = vqc.n_params
        key = jax.random.PRNGKey(42)
        params = jax.random.normal(key, (n_params,)) * 0.1
        x = jax.random.normal(key, (n_qubits,))

        # Sequential is O(n_params) individual circuit calls -> expensive.
        # Skip it (or run once) at very large qubit counts to keep runtime sane.
        run_sequential = n_qubits <= 24
        n_warmup = 2 if n_qubits < 24 else 1
        n_iter = 5 if n_qubits < 24 else 3

        fns = {
            'Gradient-only': lambda p: vqc.gradients_gradient_only(p, x, chunk_size=chunk),
            'Quantum-only': lambda p: vqc.gradients_quantum_only(p, x),
            'Cross-graph': lambda p: vqc.gradients_cross_graph(p, x, chunk_size=chunk),
        }
        if run_sequential:
            fns['Sequential'] = lambda p: vqc.gradients_sequential(p, x)

        row = {
            'n_qubits': n_qubits, 'n_params': n_params,
            'state_gb': mem_per_state, 'chunk_size': chunk, 'n_blocks': n_blocks,
        }
        failed = False
        for name, fn in fns.items():
            try:
                for _ in range(n_warmup):
                    tree_map(lambda g: g.block_until_ready(), fn(params))
                timings = []
                for _ in range(n_iter):
                    start = time.perf_counter()
                    result = fn(params)
                    tree_map(lambda g: g.block_until_ready(), result)
                    timings.append((time.perf_counter() - start) * 1000)
                row[name] = float(np.mean(timings))
                row[f'{name}_std'] = float(np.std(timings))
                print(f"    {name:14s}: {row[name]:8.2f} ms")
            except Exception as e:
                print(f"    {name:14s}: FAILED ({e})")
                row[name] = np.nan
                failed = True

        if not run_sequential:
            row['Sequential'] = np.nan  # not measured at this scale

        # Speedups relative to Sequential where available
        base = row.get('Sequential', np.nan)
        for name in ['Gradient-only', 'Quantum-only', 'Cross-graph']:
            if not np.isnan(base) and name in row and not np.isnan(row[name]) and row[name] > 0:
                row[f'{name}_speedup'] = base / row[name]
            else:
                row[f'{name}_speedup'] = np.nan

        results.append(row)

    if not results:
        print("\n  No configurations completed successfully. Check GPU memory / lightning.gpu install.")

    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_four_configs(df):
    if df.empty:
        print("No data to visualize.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    configs = ['Sequential', 'Gradient-only', 'Quantum-only', 'Cross-graph']
    colors = {'Sequential': '#95a5a6', 'Gradient-only': '#3498db',
              'Quantum-only': '#e67e22', 'Cross-graph': '#2ecc71'}

    ax = axes[0, 0]
    for c in configs:
        if c in df.columns:
            ax.plot(df['n_qubits'], df[c], marker='o', linewidth=2, label=c, color=colors[c])
    ax.set_xlabel('Number of Qubits', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Absolute Time by Config: 16-30 Qubits', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for c in ['Gradient-only', 'Quantum-only', 'Cross-graph']:
        col = f'{c}_speedup'
        if col in df.columns:
            ax.plot(df['n_qubits'], df[col], marker='D', linewidth=2, label=c, color=colors[c])
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Number of Qubits', fontweight='bold')
    ax.set_ylabel('Speedup vs Sequential (x)', fontweight='bold')
    ax.set_title('Speedup vs Qubit Count', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(df['n_qubits'], df['state_gb'], marker='o', linewidth=2, color='#e74c3c')
    ax.set_xlabel('Number of Qubits', fontweight='bold')
    ax.set_ylabel('Statevector Size (GB, log scale)', fontweight='bold')
    ax.set_title('Memory Growth (2^n scaling)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    last = df.dropna(subset=[c for c in configs if c in df.columns], how='all').iloc[-1]
    bars = [c for c in configs if c in df.columns and not pd.isna(last.get(c, np.nan))]
    vals = [last[c] for c in bars]
    ax.bar(bars, vals, color=[colors[c] for c in bars])
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title(f'Config Comparison at {int(last["n_qubits"])} Qubits', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('four_config_scaling_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved: four_config_scaling_analysis.png")
    plt.show()


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_four_configs(df):
    print("\n" + "=" * 80)
    print("ANALYSIS: FOUR-CONFIG SCHEDULING COMPARISON")
    print("=" * 80)

    if df.empty:
        print("  No results collected.")
        return

    for _, row in df.iterrows():
        print(f"\n  {int(row['n_qubits'])} qubits:")
        for c in ['Gradient-only', 'Quantum-only', 'Cross-graph']:
            sp = row.get(f'{c}_speedup', np.nan)
            if not pd.isna(sp):
                print(f"    {c:14s} speedup vs Sequential: {sp:.2f}x")
            elif c in row and not pd.isna(row[c]):
                print(f"    {c:14s} time: {row[c]:.2f} ms (Sequential not measured at this scale)")

    print()
    print("  Interpretation:")
    print("  - Gradient-only parallelizes the Gradient Graph (vmap-batched shifts)")
    print("    on the SAME monolithic, fully-entangled circuit as Sequential.")
    print("  - Quantum-only parallelizes the Quantum Graph itself: independent")
    print("    qubit-block subtrees dispatched concurrently, each with its own")
    print("    (smaller) statevector - no shift-batching within a block.")
    print("  - Cross-graph combines both and should show the largest speedup,")
    print("    consistent with QuSim-Sed's cross-graph scheduling proposal.")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\nRunning four-config (Sequential / Gradient-only / Quantum-only /")
    print("Cross-graph) benchmarks across 16-30 qubits. This can take a while")
    print("and will use significant GPU memory - monitor with `!nvidia-smi`.\n")

    df = benchmark_four_configs_qubit_scaling()
    analyze_four_configs(df)
    visualize_four_configs(df)

    # ---- Parameter-Shift vs Adjoint, across qubits and layers --------
    df_method_q = benchmark_method_vs_qubits()
    df_method_l = benchmark_method_vs_layers()
    visualize_method_comparison(df_method_q, df_method_l)

    # ---- Speedup of all three parallel configs vs parameter scaling --
    # (parameters grown via qubit count, layers held fixed)
    df_speedup_scaling = benchmark_speedup_scaling_qubits()
    print_speedup_scaling_table(df_speedup_scaling)
    visualize_speedup_scaling(df_speedup_scaling)

    # ---- Forward/backward breakdown, before vs after scheduling ------
    df_breakdown = benchmark_forward_backward_breakdown()
    visualize_forward_backward_breakdown(df_breakdown)

    print("\n" + "=" * 80)
    print("FOUR-CONFIG BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
