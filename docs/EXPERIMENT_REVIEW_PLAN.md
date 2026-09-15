# Experiment revision plan

This document addresses the experiment reviews on slides 6-7. It does not
claim results before they are collected on a supported GPU.

| Review issue | Evidence to collect | Implementation | Output | Status |
|---|---|---|---|---|
| Numerical correctness | expectation, gradient, loss, update errors vs sequential for every method | `pennylane_experiments.correctness_experiment` | correctness CSV + Streamlit chart | implemented; requires PennyLane to collect |
| Multi-stream execution | task ID, graph, dependencies, logical/CUDA stream ID, timestamps | `core.scheduler` | `stream_timeline.svg`, trace CSV/JSON | implemented; CUDA proof requires a CUDA executor |
| Kernel overlap | Nsight CUDA timeline and utilization | `qusimsed.profiling` | `.nsys-rep` plus exported report | capability-gated |
| Strong baselines | sequential, batched parameter-shift, naive multi-stream, QuSim-Sed on identical VQC | `pennylane_experiments.baseline_experiment` | timing CSV + Streamlit chart | implemented; requires PennyLane to collect |
| Scheduler clarity | dependency/resource/dispatch/completion workflow | `visualization.scheduler_workflow_svg` | workflow SVG | implemented |
| CDS overhead | measured deep size of CDS/scheduler metadata | `qusimsed.memory` | memory CSV/plot | implemented |
| Real workload | Iris, Wine, Breast Cancer, and PCA-MNIST: time, loss, accuracy, convergence/correctness | `real_benchmarks.run_real_benchmark` | training CSV + Streamlit charts | implemented; requires scikit-learn + PennyLane |
| >30 qubits | safe sweep with observed OOM status and peak VRAM | configuration + CUDA executor | scalability CSV/plot | pending supported GPU |

## Exact commands

```powershell
python -m unittest discover -s tests -v
python -m qusimsed.demo --output-dir results/demo --streams 2
# optional CUDA plumbing check; this runs CUDA matmuls, not a VQC
python -m qusimsed.demo --output-dir results/cuda-smoke --streams 2 --cuda-smoke
# strict GPU-only PennyLane correctness experiment (never falls back to CPU)
python -m qusimsed.gpu_correctness --qubits 4 --layers 2 --output results/gpu-correctness.json
# Persist CDS metadata plus live host/CUDA memory fields
python -m qusimsed.collectors memory-demo --output results/memory/cds_memory.json
# Profile a real GPU benchmark when Nsight Systems is installed
python -m qusimsed.collectors nsight --output results/nsight/parameter_shift -- python -m qusimsed.gpu_correctness --qubits 4 --layers 2
```

At startup, `qusimsed.runtime.detect_runtime()` chooses `cuda-streams` only
when a usable CUDA Python runtime is present. Otherwise the scheduler continues
as `cpu-threads`; this uses multiple CPU workers but is never labelled as CUDA
streams or GPU overlap. The demo produces genuine host-thread overlap and labels
it as such. On a GPU host, collect Nsight with
the command returned by `qusimsed.profiling.nsight_command`; preserve the raw
`.nsys-rep` file alongside exported CUDA timelines.
If `nsys` is unavailable, `collect_nsight` saves an explicit manifest instead
of profiler results. CUDA memory fields are `null` when no usable CUDA Python
runtime exists; `null` does not mean zero allocation.

## Implementation-vs-paper gap analysis

The original scripts make identical VQCs for the merged benchmark, but
`Quantum-only` and `Cross-graph` use Python thread pools and JAX `vmap` as
proxies. They neither build `CDSRecord`/`RecordPool` nor bind task dispatch to
CUDA streams, and therefore cannot substantiate the manuscript's claims about
cross-graph dependency scheduling, stream IDs, or kernel overlap. The older
four-config script also contains block decomposition which changes the simulated
problem size and must not be used for fair speedup claims.
