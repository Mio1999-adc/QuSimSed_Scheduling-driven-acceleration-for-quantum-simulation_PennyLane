# QuSim-Sed: Scheduling-Driven Acceleration for Hybrid Quantum-Classical Simulation on GPUs

QuSim-Sed coordinates quantum-circuit execution and classical gradient work
through a Coordinating Data Structure (CDS). The maintained implementation
extracts PennyLane tape dependencies and Torch FX arithmetic graphs, then
schedules ready operations subject to memory and estimated SM capacity.

**For training with a remote dashboard, start with `python -m qusimsed.train`.**
See [training and SSH access](#interactive-ui-and-scheduler-diagnostics) for
the server command and local tunnel. Use `python -m qusimsed.server_benchmark`
for controlled correctness checks and synthetic timing.
The current GPU executor is
`torch-cuda-statevector`: it executes PennyLane tapes using complex128 Torch
CUDA kernels on scheduler-owned streams. It is a separate backend from
Lightning-GPU/cuStateVec and the older JAX/Catalyst benchmark scripts.

Local validation has covered numerical agreement with PennyLane, dependencies,
resource admission, and application training. **GPU-server kernel overlap,
resource calibration, and performance still require validation.** The paper's
reported speedups are not measurements of this new executor.

## Current features

| Feature | Implemented behavior |
|---|---|
| Circuit metadata extraction | Reads actual tape operations, wire dependencies, measurements, and trainable parameter ownership |
| Classical graph extraction | Traces parameter-shift reduction and MSE/SGD arithmetic with Torch FX and lowers its dependency edges into CDS tasks |
| Cross-graph scheduling | Links measurements, gradient reductions, gradient assembly, loss, and parameter updates; successors become ready after predecessor completion |
| Dependency-preserving partitioning | Groups contiguous topological nodes by state ownership and graph type; preserves individual dependencies without adding partition-wide barriers |
| Affinity and synchronization costs | Ranks feasible task/stream pairs using priority, aging, affinity, and estimated cross-stream dependency cost |
| SM-demand admission | Bounds the sum of running tasks' estimated fractions of whole-device SM capacity |
| GPU memory admission | Uses live allocatable CUDA memory, an optional budget cap, and a safety factor; keeps state/workspace reservations until the owning evaluation finishes |
| Explicit CUDA execution | Owns CUDA streams, inserts cross-stream event waits, observes device completion before releasing resources, and records tensor stream usage |
| Parameter-shift | Builds shifted evaluations from PennyLane recipes/frequencies and schedules their reductions as dependencies become ready |
| Adjoint | Schedules the forward circuit followed by an ordered reverse sweep, uncomputing primal and adjoint states |
| Correctness and profiling | Compares expectations, loss gradients, losses, and updates against PennyLane; exports traces, NVTX labels, memory estimates, and observed Torch peak allocation |

The CDS uses a dictionary indexed by node ID. Each record stores dependencies,
readiness, execution state, partition/stream assignment, resource estimates,
and framework metadata. Partitioning supplies scheduling hints; it does not
fuse kernels. Each independent circuit evaluation owns its state vector, while
gates updating the same state retain storage-order dependencies.

The extracted classical graph is the differentiation/loss arithmetic used by
this backend. Arbitrary internal JAX or PennyLane Autograd graphs and classical
Jacobians for shared/transformed QNode arguments are outside the adapter's scope.
See [Scheduling implementation](docs/SCHEDULING_IMPLEMENTATION.md) for the
algorithm, memory model, and execution contract.

## Execution backends and available comparisons

| Configuration | Execution | Available methods |
|---|---|---|
| `execution_backend="torch-cuda"` | Explicit CUDA state-vector executor; GPU required | Sequential, QuSim-Sed; parameter-shift and adjoint |
| `execution_backend="torch-cpu"` | Explicit CPU validation of the same arithmetic and dependencies | Sequential, QuSim-Sed; parameter-shift and adjoint |
| `execution_backend="pennylane"` | Separate Lightning-GPU experiment path | Sequential and Batched parameter-shift for parameter-shift; Sequential for adjoint |

The CLIs select Torch CUDA by default and fail if CUDA is unavailable.
For synthetic runs, `server_benchmark --cpu-validation` explicitly selects CPU
validation. The training launcher instead accepts `--backend torch-cpu` for
CPU checks or `--backend pennylane` for the separate Lightning-GPU path.
Backend selection is also available in the Python API and Streamlit UI.

The maintained Torch comparison uses **identical kernels and circuits**:
Sequential uses one scheduler stream, and QuSim-Sed uses the configured stream
count. This Sequential baseline still includes CDS construction and scheduling
overhead; it is not conventional sequential PennyLane. Speedup is computed
within each backend. Naive multi-stream, separate Gradient-only/Quantum-only
ablations, and Catalyst comparisons are not implemented in this maintained
Torch comparison.

## GPU-server setup

The local numerical tests used Python 3.11, PennyLane 0.44.0, and Torch 2.5.1.
The dependency file is not a lockfile; record the versions installed on the
server with each experiment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled Torch build appropriate for the server using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Then install the repository dependencies:

```bash
python -m pip install -r requirements-gpu.txt

# Required for application training with the dashboard:
python -m pip install scikit-learn streamlit

# Optional process-memory telemetry:
python -m pip install psutil

nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m qusimsed.server_benchmark --help
python -m qusimsed.train --help
```

`requirements-gpu.txt` includes Lightning-GPU for the separate PennyLane path.
The Torch executor does not require JAX or Catalyst. Nsight Systems (`nsys`)
must be installed on the server for CUDA timeline collection.

## Experiments

GPU server runs containing QuSim-Sed now **automatically collect a companion
Nsight profile** after the primary run. This adds one QuSim-Sed trace execution
with the same circuit and scheduling configuration; it does not instrument the
reported timing samples. The primary result is saved before profiling begins.
Automatic profiling applies to server CLI correctness/benchmark and QuSim-Sed
trace runs, synthetic sweep cases, and synthetic UI runs when companion
profiling is enabled. Application training through `qusimsed.train` saves
results automatically but does not launch a profiler. Application UI profiling
is recorded as skipped; direct Python calls do not launch profiling either.

For `--output results/server/run.json`, automatic artifacts are:

```text
results/server/run.json
results/server/profiling/run/qusimsed.nsys-rep
results/server/profiling/run/qusimsed.nsight.json
results/server/profiling/run/qusimsed-trace.json
```

The result JSON's `profiling` field and terminal output point to these files.
`--profile-output results/nsight/my-run` changes the profile prefix;
`--no-profile` disables the companion run. Missing Nsight or a profiler failure
is recorded as `collected: false` without discarding the primary result. CPU
validation records a skipped-profile manifest. An unavailable CUDA runtime
still fails the requested GPU computation rather than falling back to CPU.
An existing collector-managed Nsight session suppresses the automatic replay
to avoid nested profiling. Numerical correctness failures also skip the replay.


### 1. Tests and numerical correctness

```bash
python -m unittest discover -s tests -v

python -m qusimsed.server_benchmark --mode correctness \
  --qubits 4 --layers 2 --differentiation parameter-shift \
  --output results/server/shift-correctness.json

python -m qusimsed.server_benchmark --mode correctness \
  --qubits 4 --layers 2 --differentiation adjoint \
  --output results/server/adjoint-correctness.json
```

Correctness mode compares each scheduled method with an independent PennyLane
reference. For the Torch backend, that small reference uses `default.qubit` on
CPU outside benchmark timing; the candidate computations use the requested
backend. The checks cover expectation, MSE loss, the **loss gradient**, and the
SGD parameter update. A failed numerical comparison causes a nonzero CLI exit.
The default synthetic loss is `(expectation - 1)**2` with learning rate `0.05`.

Local-only validation is explicit:

```bash
python -m qusimsed.server_benchmark --cpu-validation --mode correctness \
  --qubits 4 --layers 2 --differentiation adjoint \
  --output results/local/adjoint-correctness.json
```

The scheduling suite was locally validated with its CUDA-only test skipped
because CUDA was unavailable. Additional profiling tests cover missing tools,
report validation, configuration forwarding, and recursion prevention. Passing CPU tests establishes neither
GPU kernel overlap nor GPU speedup. Tests requiring PennyLane/Torch also skip
when those packages are absent; inspect the test summary on the server.

### 2. Controlled synthetic timing

The VQC uses RY feature encoding, per-layer RX/RY/RZ rotations, a CNOT ring,
and a Pauli-Z expectation. All methods receive the same full-width circuit,
features, and initial parameters. By default, `P = 3 * qubits * layers`.
Set `--parameters P` to train only the first P rotations while freezing the
remaining angles, preserving the circuit's gates and initial function.

```bash
python -m qusimsed.server_benchmark --mode benchmark \
  --qubits 10 --layers 3 --differentiation parameter-shift \
  --streams 4 --warmup 1 --iterations 30 --seed 7 \
  --output results/server/10q-3l-shift.json
```

Benchmark mode always runs both Sequential and QuSim-Sed. It reports mean and
standard deviation of wall-clock iteration time and speedup over Sequential.
Each timed iteration reuses the same initial workload and includes tape/graph
construction, executor setup, scheduling, forward evaluation, differentiation,
MSE/SGD computation, device completion, and result handling. It measures a
repeated training-step workload, not a convergence trajectory or kernel-only time.

**SM demand is estimated automatically by default.** Omit `--sm-demand`
(or use `sm_demand=None` in Python). Each scheduled task receives a workload
estimate using the actual GPU SM count, state-vector size and gate width;
adjoint reverse work receives a larger estimate. The scheduler admits ready
work within the summed demand and live memory budgets, then uses dependency,
affinity and synchronization costs to select streams.

This is a conservative static model, **not measured CUDA occupancy**. It uses
one model work unit per 256 complex amplitudes, scaled by gate width and reverse
work, and saturates admission at one unit per SM. Torch/cuBLAS launch geometry
and bandwidth are not measured by this model. Large state-vector tasks may
therefore remain serial. CPU validation or unavailable GPU properties use a
conservative full-demand fallback. GPU throughput tuning and automatic
profile-based calibration remain future work; maximum overlap is not guaranteed.

**Both resources constrain every dispatch**, as in Algorithm 2 and equations
(20)-(21): `reserved_memory + task_memory <= effective_memory_limit` and
`used_sm + task_sm_demand <= sm_capacity`. Available SM capacity is the device
capacity minus demands reserved by unfinished tasks. Live global-memory
checks include pending reservations and avoid counting already allocated
state twice. State leases remain held across gates until their evaluation ends.

Feasible tasks are ranked by priority, affinity, newly exposed parallelism,
synchronization cost and memory/SM pressure. `--parallelism-weight` and
`--resource-weight` control the last two additions. Each trace row includes a
`resource_admission` snapshot with the before-dispatch budgets and demands,
SM equivalents, and selection score. Profiling is for verifying achieved
kernel overlap; it is not required to activate these constraints. See
[resource model details](docs/SCHEDULING_IMPLEMENTATION.md#resource-model).

Trace results expose per-task `resource_estimates`, source and inputs, along
with `sm_demand_mode`. `--sm-demand 1.0` explicitly forces conservative serial
quantum admission; `--sm-demand 0.25` overrides the model to allow up to four
quantum tasks, subject to dependencies, streams and memory. The UI defaults to
automatic estimation and exposes an optional manual override.

Useful server options:

| Option | CLI default | Meaning |
|---|---|---|
| `--gpu-device` | `0` | CUDA device index |
| `--streams` | `4` | QuSim-Sed stream count; Sequential uses one |
| `--sm-demand` | Unset (automatic) | Optional manual admission-demand override |
| `--memory-gib` | Unset | Optional cap on the budget derived from live allocatable memory |
| `--memory-safety-factor` | `0.8` | Fraction of the budget admitted by the scheduler |
| `--partition-size` | `16` | Maximum nodes per topological partition |
| `--affinity-weight`, `--sync-weight` | `1.0` each | Task/stream selection weights |
| `--warmup`, `--iterations` | `1`, `30` | Warm-up and timed iteration counts |
| `--strategy` | `qusimsed` | Selects one method in **trace mode only** |
| `--no-profile` | Unset | Disable automatic companion profiling |
| `--profile-output` | Derived from `--output` | Override the Nsight output prefix |

The Python `ExperimentConfig` default is 10 iterations; set it explicitly to
30 when matching the CLI timing protocol. The memory safety factor applies
after the optional budget cap. Resource estimates reduce over-admission but
cannot prevent another process from allocating GPU memory after a check.

### 3. Scaling and scheduler sensitivity

The paper's experiment axes can guide new server runs:

| Experiment | Configuration |
|---|---|
| Qubit scaling | 10, 15, 20, 25 qubits at 3 layers |
| Depth scaling | 20 qubits at 3, 5, 7 layers |
| Differentiation | Repeat both axes for parameter-shift and adjoint |
| Scheduler sensitivity | Vary streams, partition size, affinity/synchronization weights, and justified SM estimates while holding the circuit fixed |

For example, this Bash sweep runs the qubit-scaling axis for both methods.
The example below uses automatic resource estimation. Validate a small configuration
and its memory requirements before launching the larger cases.

```bash
for differentiation in parameter-shift adjoint; do
  for qubits in 10 15 20 25; do
    python -m qusimsed.server_benchmark --mode benchmark \
      --qubits "$qubits" --layers 3 --differentiation "$differentiation" \
      --streams 4 --warmup 1 --iterations 30 \
      --output "results/scaling/${differentiation}-${qubits}q-3l.json" || exit 1
  done
done
```

These runs use the current Torch executor, so they do not reproduce the paper's
PennyLane/JAX/Lightning-GPU measurements directly. Use the collection runner
below for independent parameter-count sweeps at fixed qubits and depth.

#### Collect experiment results automatically

**No extra command is needed to save results from normal runs.**
`correctness_experiment`, `baseline_experiment`, and `run_real_benchmark`
automatically save results when called from the server CLI, Streamlit, or Python.
Each invocation gets a unique folder under
`<config.output_dir>/experiments/<UTC-time>-<run-kind>-<unique-id>/`.
The server CLI uses the parent directory of its existing `--output` file;
Streamlit uses its Output directory setting; Python defaults to `results`.
The returned `result["collection"]` contains the exact saved paths.

Each folder contains `result.json`, `summary.csv`, `errors.csv`, `timings.csv`,
and `convergence.csv`. Applicable tables include raw iteration times, speedup,
time savings, errors, and per-epoch training loss/time; inapplicable tables
are empty. JSON retains configuration, runtime metadata, traces and final
parameters supplied by the run. Failures are recorded and then raised normally.
File writes occur outside measured iterations/training loops. Repeated calls
preserve previous archives. Server profiling links are added to its archive
after profiling finishes. A standalone timing call explicitly records that it
did not perform an independent correctness check. Unit-test pass/fail counts
are software validation, not experiment performance measurements.

For example, the existing call `result = run_real_benchmark(config)` now saves
its results without requiring `save_real_result` or another command. Explicit
save functions still work for named copies. Set `collect_results=False` in
`ExperimentConfig` only when a caller manages persistence itself (as the sweep
runner does).

**Optional full sweeps:** the command below is only for launching many
configurations automatically. Run it from the repository root in the GPU
environment. It is not a required postprocessing or collection step.
The collection runner validates numerical correctness before timing each case,
saves results incrementally, and exports tables for plotting. On the GPU server:

```bash
# Inspect the matrix and method support without executing circuits.
python -m qusimsed.experiment_suite --output results/experiments --plan-only
# Execute exactly that matrix; resume also retries failed/interrupted cases.
python -m qusimsed.experiment_suite --output results/experiments --resume
```

Defaults cover both parameter-shift and adjoint: qubits 10/15/20/25 at depth 3,
depths 3/5/7 at 20 qubits, and trainable parameters 36/72/108/144/180/216/270
at 20 qubits and depth 5. Duplicate points are merged (26 cases per seed).
Parameter sweeps freeze a suffix of the existing rotations; they hold gate
count and initial angles fixed. `P` cannot exceed `3 * qubits * layers`.
Use `--seeds 7 11 19` for independent seeded runs, `--axes` to choose sweeps,
and `--iterations`/`--warmup` to override 30 measured steps and one warm-up.
Each seed remains a separate result; the runner does not pool uncertainty.

The default backend is Torch CUDA, with an independent `lightning.gpu`
correctness reference. Missing CUDA or reference dependencies cause explicit
failures. `--reference-backend default.qubit` selects a CPU reference if desired.
SM demand is estimated automatically; `--sm-demand` is an optional override. A completed synthetic case automatically receives a separate Nsight
companion run; `--no-profile` disables it. Profiling is outside measured samples.
Application profiling is currently recorded as skipped.

| Saved file under the output directory | Contents |
|---|---|
| `manifest.json`, `capabilities.json` | Exact matrix, environment/package/GPU metadata, progress, supported and missing methods |
| `summary.json`, `summary.csv` | Per-method status, correctness, timing statistics, speedup and time-saving percentage |
| `errors.csv` | Absolute/relative errors, RMSE and tolerance checks for expectation, loss gradient, loss and updated parameters |
| `timings.csv` | Individual synthetic iteration times, with configuration and seed |
| `convergence.csv` | Application training loss and elapsed time for every epoch |
| `comparisons.csv` | Pairwise speedup and time savings within the same case, backend and timing scope |
| `cases/<case-id>/` | Full case, correctness, benchmark or training JSON; optional traces and `profiling/qusimsed.*` artifacts |

Speedup is baseline time / method time; time saving is
`100 * (1 - method time / baseline time)`. Slowdowns retain negative savings.
Numerical failures suppress aggregate performance comparisons; failure details
and any raw measurements remain saved. Resume requires the same matrix/options;
use another output directory for a different configuration.

Torch currently measures Sequential and QuSim-Sed. With `--backend pennylane`,
the available methods are Sequential and Batched parameter-shift (the latter
only for parameter-shift differentiation). Requested missing methods, including
Catalyst and QuSim-Sed + Catalyst, receive `unsupported` rows without timing
values. Historical proxy results are never imported. `--methods` selects report
entries; underlying validation/benchmark routines still run their supported
method set. Cross-backend speedups are not computed automatically.

For application training sweeps, for example:

```bash
python -m qusimsed.experiment_suite \
  --axes qubits depth --qubits 4 6 8 --depths 1 2 3 \
  --fixed-qubits 4 --fixed-layers 2 --workloads iris mnist-pca \
  --samples 50 --epochs 3 --output results/application-experiments
```

These report actual multi-epoch training time, test errors/accuracy and loss
curves. Synthetic runs measure repeated training steps from the same initial
parameters. These timing scopes are labeled separately in the exports.
Dataset dependencies and MNIST cache/network access are required for applications.

For a small local validation run (not GPU performance evidence):

```bash
python -m qusimsed.experiment_suite --cpu-validation \
  --axes parameters --parameter-qubits 2 --parameter-layers 1 \
  --parameters 1 3 6 --iterations 1 --warmup 0 --no-profile \
  --output results/collection-smoke
```

### 4. Application benchmarks

The application suite trains a binary VQC classifier with full-batch updates.
It reports training time, test MSE, test accuracy, speedup, and parameter/training
loss-trajectory deviations from Sequential.

| Workload key | Task |
|---|---|
| `iris` | Iris classes 0 versus 1 |
| `wine` | Wine classes 0 versus 1 |
| `breast-cancer` | Binary diagnostic classification |
| `mnist-pca` | MNIST digits 3 versus 5, reduced to qubit-sized features |

A stratified 80/20 split precedes scaling and PCA; preprocessing is fitted on
the selected training samples only. `samples` caps training size, and the test
subset is capped separately. MNIST is fetched from OpenML on first use and
requires dataset access/cache availability.

Run applications through the **Real QML benchmarks** UI tab or the Python API:

```python
from qusimsed.config import ExperimentConfig
from qusimsed.real_benchmarks import run_real_benchmark, save_real_result

config = ExperimentConfig(
    workload="iris",              # Use "mnist-pca" for the image benchmark.
    qubits=4,
    layers=2,
    differentiation="adjoint",    # "parameter-shift" is also supported.
    execution_backend="torch-cuda",
    streams=4,
    sm_demand=None,               # Automatic workload/hardware estimate.
    samples=50,
    iterations=3,                 # Application iterations are training epochs.
    seed=7,
)
result = run_real_benchmark(config)
save_real_result(result, "results/applications/iris-adjoint.json")
```

Application timing covers the training loop and per-epoch training-loss
measurement; it excludes dataset loading/preprocessing and final test
prediction. It does not use the synthetic warm-up protocol. Samples and
batch accumulation are processed sequentially on the host; the selected
scheduler controls each quantum-gradient evaluation. Full cross-sample GPU
training orchestration is not claimed. Saved results contain convergence error
metrics, complete per-epoch training-loss curves and times, and final parameters.

VQE/Hamiltonian-sum benchmarking remains unimplemented in the maintained
executor, even though it is described in the expanded paper setup.

### 5. GPU traces and memory evidence

#### Automatic profiling: run, locate, and inspect

Run these commands from the repository root **on the GPU server**:

```bash
# Verify that the server has both CUDA and the profiler available.
nvidia-smi
nsys --version

# Run the benchmark; the CLI then profiles one additional QuSim-Sed step.
python -m qusimsed.server_benchmark --mode benchmark \
  --qubits 10 --layers 3 --differentiation parameter-shift \
  --streams 4 --warmup 1 --iterations 30 \
  --output results/server/run.json
```

This command uses automatic SM-demand estimation. Stream count alone does
not override memory or estimated SM admission; inspect the trace estimates
and server profiling before using a manual demand override.

The primary benchmark finishes and saves its results first. The CLI then
prints `Collecting companion QuSim-Sed profile: ...` and launches one extra
trace execution under Nsight. This extra step adds runtime but is excluded
from the reported benchmark samples. Its report captures that step, not all
30 timed iterations or the Sequential baseline.

On successful collection, the example creates:

```text
results/server/
├── run.json                         # Benchmark results and profiling links
├── run.csv                          # Timing comparison rows
└── profiling/run/
    ├── qusimsed.nsys-rep             # Open in NVIDIA Nsight Systems
    ├── qusimsed.nsight.json          # Collection status, command, stdout/stderr
    └── qusimsed-trace.json           # CDS dependencies, streams, resource data
```

Paths are relative to the directory where the command runs. Files produced
on a remote GPU server remain on that server; they are not automatically
downloaded to your local computer. Reusing the same output paths overwrites
results, so choose a distinct `--output` filename for each configuration/run.

Check the manifest before interpreting the report:

```bash
python -m json.tool results/server/profiling/run/qusimsed.nsight.json
```

`"collected": true` means the profiler/target exited successfully and a new,
nonempty report was found. If it is `false`, inspect `reason`, `stderr`, and
`returncode` when present. Missing Nsight, an explicit CPU validation run, or
a failed profiling child must not be treated as a successful GPU profile.
The main `run.json` also contains a `profiling` field with artifact paths and
status. A completed benchmark does not by itself prove collection succeeded.

To inspect the GPU timeline, open `qusimsed.nsys-rep` in **NVIDIA Nsight
Systems** on the server, or copy it to a machine with the Nsight Systems GUI.
Look for the QuSim-Sed NVTX task labels and CUDA stream/kernel intervals;
compare overlap, synchronization waits, and idle gaps. The scheduler JSON is
useful for matching tasks to dependencies, but is not a replacement for the
GPU timeline.

To choose another profiling location:

```bash
python -m qusimsed.server_benchmark --mode benchmark \
  --qubits 10 --layers 3 \
  --output results/server/run-custom.json \
  --profile-output results/nsight/run-custom/qusimsed
```

Here `--output` names the primary result JSON; `--profile-output` is a prefix
for `.nsys-rep`, `.nsight.json`, and `-trace.json`. To run without the extra
profile, append `--no-profile` to the server command. Automatic collection
is available in the server CLI, optional sweep runner, and Streamlit's
"Collect companion Nsight profile" setting. Direct Python experiment calls
save results automatically but do not launch profiling. Application datasets
currently receive a skipped-profile manifest because the replay is synthetic.

#### Manual Sequential versus QuSim-Sed comparison

Automatic profiling captures a companion QuSim-Sed run. For a separate
Sequential comparison, collect reports manually for the same workload under
both strategies. The
`0.25` SM demand here is an example estimate to be validated on the server.

```bash
for strategy in sequential qusimsed; do
  python -m qusimsed.collectors nsight --output "results/nsight/$strategy" -- \
    python -m qusimsed.server_benchmark --mode trace --strategy "$strategy" \
    --qubits 10 --layers 3 --differentiation parameter-shift \
    --streams 4 --sm-demand 0.25 \
    --output "results/server/${strategy}-trace.json" || exit 1
done
```

The collector requests CUDA, NVTX, and OS runtime tracing. Inspect its manifest:
`collected: false` means no successful profiling result, even if a manifest was
written. The collector may return normally for an unavailable profiler, so the
shell exit status alone is insufficient. Nsight Compute collection and
automatic SM-demand calibration are not provided by this wrapper.

Use the Nsight timeline to establish actual kernel overlap, stream activity,
synchronization, and idle periods. CDS timestamps include host submission and
completion-observation overhead. Stream handles or overlapping host intervals
alone do not establish overlapping GPU kernels.

| Run | Saved artifacts |
|---|---|
| Correctness | JSON with configuration, environment, errors, and traces; CSV comparison rows |
| Synthetic benchmark | JSON with timing scope, configuration, environment, final trace per method, and timing rows; CSV comparison rows |
| Trace | JSON with circuit/FX metadata, partitions, task dependencies, stream handles, event-wait count, memory reservations, observed Torch peak allocation, and estimated SM peak |
| Application | JSON and CSV with dataset/split sizes, configuration, performance, and correctness metrics |
| Nsight collector | `.nsight.json` manifest and, on successful collection, `.nsys-rep` |

The CLI records Torch/CUDA versions and selected GPU properties. Also retain
`python -m pip freeze`, driver information, profiler version, and raw reports
alongside each experiment. A CUDA field reported as `null` is unavailable,
not a zero measurement.

## Interactive UI and scheduler diagnostics

To start application training **and** Streamlit together on the GPU server,
run this from the repository root in the environment containing the GPU
dependencies and Streamlit:

```bash
python -m qusimsed.train --dataset iris --qubits 4 --layers 2 \
  --differentiation adjoint --epochs 3 --samples 50 \
  --output-dir results/training --port 8501
```

This runs both Sequential and QuSim-Sed on the default Torch CUDA backend.
No separate Streamlit launch or result-collection command is required.

| Training option | Default | Purpose |
|---|---|---|
| `--dataset` | `iris` | `iris`, `wine`, `breast-cancer`, or `mnist-pca` |
| `--qubits`, `--layers` | `4`, `2` | Circuit width and depth |
| `--parameters` | All rotations | Train the first P rotations; freeze the rest |
| `--differentiation` | `adjoint` | `adjoint` or `parameter-shift` |
| `--epochs`, `--samples` | `3`, `50` | Training epochs and training-sample cap |
| `--learning-rate`, `--seed` | `0.05`, `7` | Optimizer step size and reproducibility |
| `--backend`, `--gpu-device` | `torch-cuda`, `0` | Execution backend and GPU index |
| `--streams`, `--sm-demand` | `4`, automatic | Stream count and optional manual demand override |
| `--output-dir`, `--port` | `results/training`, `8501` | Shared result directory and server dashboard port |
| `--no-ui` | Unset | Train without starting the dashboard |
| `--exit-after-training` | Unset | Stop the dashboard when training completes |

Run `python -m qusimsed.train --help` for memory, partition, scheduling-weight
and trace options. MNIST-PCA requires dataset access or an existing cache.

The launcher starts Streamlit first, checks that it is ready, then runs training.
The dashboard uses the same output directory, so **Saved results → Refresh
saved results** shows the CLI run and its saved results. A running record is
visible during execution; curves and metrics become available after training
finishes. Dashboard controls launch new experiments; they do not change a CLI
training job already in progress. Avoid launching another GPU run while the
CLI job is using the same GPU if you need isolated benchmark timings.

On your **local computer**, keep this SSH tunnel running:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8501:127.0.0.1:8501 YOUR_USER@YOUR_GPU_SERVER
```

Open **http://localhost:8501** in your local browser. Streamlit binds only to
the server's loopback interface; no public port needs to be opened. If your
local 8501 port is occupied, use `-L 8502:127.0.0.1:8501` and browse localhost:8502.
For a different server port, change `--port` and the tunnel's destination port.

The dashboard remains running after training ends until Ctrl+C. Use
`--exit-after-training` to stop it automatically or `--no-ui` for training only.
Errors during training stop the child dashboard and remain in the automatic
result archive when execution has begun. Server startup errors are reported
before training; inspect `<output-dir>/streamlit.log`. Use a persistent server
terminal such as tmux when the launcher should survive SSH disconnection.
Streamlit must be installed in the same Python environment as the launcher.

With the example above, files are stored as follows:

```text
results/training/
├── streamlit.log
└── experiments/<timestamp>-training-<unique-id>/
    ├── result.json       # Configuration, status, metrics, final parameters
    ├── summary.csv      # Training time, accuracy, errors, speedup, time savings
    ├── errors.csv
    ├── convergence.csv  # Training loss and duration for each epoch
    └── timings.csv      # Empty for application runs; used by synthetic timing
```

If the page cannot be reached, verify the server printed `Dashboard ready`,
the SSH tunnel is still running, and both port numbers match. If the launcher
reports an occupied server port, choose another `--port`. For a missing
Streamlit module, install it using `python -m pip install streamlit` in the
same activated environment. The startup log is `results/training/streamlit.log`
unless `--output-dir` was changed.

For the dashboard alone and standalone diagnostics:

```bash
streamlit run app/streamlit_app.py

# Dependency/trace smoke demo; this is not a VQC benchmark.
python -m qusimsed.demo --output-dir results/demo --streams 4

# CDS metadata/process-memory diagnostic, not VQC GPU-memory consumption.
python -m qusimsed.collectors memory-demo --output results/memory/cds_memory.json
```

The UI exposes qubit count, circuit depth, trainable parameter count,
differentiation, backend and correctness-reference selection, GPU index,
streams, SM demand, seed, learning rate, warm-up, memory cap/safety factor,
partition size, affinity/synchronization weights, trace recording and profiling.
The benchmark tab adds measured iteration count. The training tab adds dataset
(Iris, Wine, breast cancer or MNIST-PCA), training sample count and epoch count.
Application updates are full-batch; a minibatch control is not exposed because
the current runner does not implement it. Comparisons run all supported methods
for the selected backend; unsupported Catalyst combinations are labeled clearly.

Results show the actual saved configuration, status, method tables and selectable
charts for errors, training time, accuracy, speedup and time savings. Synthetic
timings have per-iteration plots; application results have loss and duration
curves by epoch. JSON/CSV downloads, trace tables and profiling status are
available alongside the results. The **Saved results** tab reopens automatic
archives under the selected output directory, without rerunning an experiment.
Changing controls does not relabel an earlier result: its original settings
remain visible. These charts describe individual runs; launching scaling sweeps
and plotting aggregated scaling curves are not yet UI features.

The demo exports JSON/CSV traces plus SVG timeline/workflow diagrams. Its
optional `--cuda-smoke` path runs Torch matrix multiplication for executor
plumbing validation; use server trace mode for VQC evidence.

## Supported scope and limitations

The executor supports analytic unitary tapes with one expectation measurement,
complex128 states, and dense gates/observables on up to four wires. Trainable
gates must have one scalar parameter with an applicable shift recipe/frequency
rule or adjoint derivative. The repository's RX/RY/RZ + CNOT ansatz is supported.

State preparation operations, noisy/finite-shot circuits, parameter
broadcasting, multi-parameter trainable gates, large dense observables, and
Hamiltonian-sum VQE need additional adapters. The implementation uses
host-observed per-task completion events; per-gate overhead can be substantial.
Partitioning, more streams, or successful correctness checks do not guarantee
a speedup.

## Repository map

| Path | Purpose |
|---|---|
| [core/cds.py](qusimsed/core/cds.py) | Records, dependency validation, topological partitioning |
| [core/scheduler.py](qusimsed/core/scheduler.py) | Task/stream selection, memory leases, SM admission, completion traces |
| [graph_adapter.py](qusimsed/graph_adapter.py) | PennyLane tape and Torch FX graph extraction |
| [tape_executor.py](qusimsed/tape_executor.py) | CUDA streams/events and state-vector operations |
| [scheduled_vqc.py](qusimsed/scheduled_vqc.py) | Executable joint graph, parameter-shift, adjoint, MSE/SGD |
| [server_benchmark.py](qusimsed/server_benchmark.py) | GPU CLI for correctness, timing, and traces |
| [pennylane_experiments.py](qusimsed/pennylane_experiments.py) | Backend-specific experiment matrix and independent validation |
| [real_benchmarks.py](qusimsed/real_benchmarks.py) | Classification workloads and training comparisons |
| [collectors.py](qusimsed/collectors.py) | Nsight and metadata-memory entry points |
| [app/streamlit_app.py](app/streamlit_app.py) | Interactive experiment controls |
| [tests](tests) | Scheduler, numerical, application-routing, and GPU-gated checks |
| [Scheduling implementation](docs/SCHEDULING_IMPLEMENTATION.md) | Detailed methodology mapping and resource model |

## Paper context and historical scripts

The paper describes an A100 80 GB/108-SM evaluation with PennyLane, JAX, and
Lightning-GPU and reports up to 6.24x parameter-shift speedup and 83.6% time
reduction. Those are paper-reported results, not validated results of the
current Torch executor. The current backend and timing scope must be named in
any new comparison.

Historical files remain available for inspecting earlier experiments:

| File | Status |
|---|---|
| `qusimsed_merged_benchmark.py`, `qusimsed_merged_benchmark (1).py` | Full-width circuit thread-pool/`vmap` scheduling proxies with older Catalyst comparisons; do not invoke the current CDS executor |
| `qusimsed_catalyst_benchmark (1).py` | Earlier Catalyst comparison script |
| `qusimsed_four_config_benchmark (1).py` | Earlier block-decomposed circuit experiment; reduced state-space work can inflate apparent scheduling speedups |

The old HTML cost-model console is not present in this checkout. Use the
maintained Streamlit app for current experiment controls. The
[experiment-review plan](docs/EXPERIMENT_REVIEW_PLAN.md) and older portions of
the [environment guide](docs/ENVIRONMENT_AND_RUN_GUIDE.md) provide historical
context; the implementation guide and executable method matrix define current
support.

Paper title for reference:

> QuSim-Sed: Scheduling-Driven Acceleration for Hybrid Quantum-Classical Simulation on GPUs.
> Working paper, SC'26 submission draft.
