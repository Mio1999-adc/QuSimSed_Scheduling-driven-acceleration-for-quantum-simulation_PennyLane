# QuSim-Sed environment and run guide

For the current stream-controlled VQC executor, use the commands and backend
requirements in [Scheduling implementation](SCHEDULING_IMPLEMENTATION.md).
The historical Lightning/proxy instructions below do not provide explicit
CUDA-stream ownership.


This guide separates three execution modes. Never cite a CPU-thread run as
GPU-stream evidence.

| Mode | When it is selected | What it can validate |
|---|---|---|
| `cpu-threads` | No usable CUDA Python runtime | CDS dependencies, correctness logic, CPU scheduling behavior |
| `cuda-streams` | CUDA-enabled Python runtime is detected | CUDA task dispatch and GPU memory telemetry |
| `lightning.gpu` VQC | PennyLane's `lightning.gpu` initializes | Real GPU VQC correctness and timing experiments |

## Recommended research platform

Use a Linux workstation, HPC node, Docker container, or Google Colab GPU with
an NVIDIA GPU. `lightning.gpu` uses NVIDIA cuQuantum for GPU state-vector
simulation and requires CUDA-capable hardware. For the cleanest setup, use
Python 3.11 or 3.12 in a new virtual environment. This repository's current
Windows/Python 3.13 environment detects its NVIDIA driver but cannot install
the required `custatevec-cu12` package, so it runs in `cpu-threads` mode.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pennylane "pennylane-lightning[gpu]" scikit-learn pandas streamlit psutil
# Match this installation to the CUDA version supported by the host.
pip install "jax[cuda12]"
pip install pennylane-catalyst optax
```

If package installation does not install cuQuantum automatically, follow the
official Lightning-GPU source-install procedure: install `custatevec-cu12`, set
`CUQUANTUM_SDK` to the installed cuQuantum location, then install/build
Lightning-GPU. Do not mix arbitrary CUDA, driver, JAX, and cuQuantum versions.

Verify the platform before a paper run:

```bash
nvidia-smi
python -c "from qusimsed.runtime import detect_runtime; print(detect_runtime())"
python -c "import pennylane as qml; print(qml.device('lightning.gpu', wires=2))"
python -c "import jax; print(jax.devices())"
nsys --version
```

`detect_runtime()` must select `cuda-streams`, and the PennyLane command must
create `lightning.gpu`. If either fails, resolve the environment before calling
the result a GPU experiment.

## Install this repository

```bash
git clone <repository-url>
cd QuSimSed_Scheduling-driven-acceleration-for-quantum-simulation_PennyLane
pip install -r requirements-gpu.txt
pip install scikit-learn streamlit psutil pandas
```

`requirements-gpu.txt` is a minimal starting point; the recommended-platform
commands above add the application and profiling dependencies.

## Test the implementation

Run these before changing qubits, streams, or the benchmark matrix:

```bash
python -m unittest discover -s tests -v
python -m qusimsed.demo --output-dir results/demo --streams 4
python -m qusimsed.collectors memory-demo --output results/memory/cds_memory.json
```

Expected artifacts:

- `results/demo/scheduler_trace.csv` and `.json`: node, graph, dependency,
  stream, timestamps, and executor.
- `results/demo/stream_timeline.svg`: observed timeline.
- `results/demo/scheduler_workflow.svg`: Algorithm 2 workflow.
- `results/memory/cds_memory.json`: CDS metadata plus process/CUDA snapshots.

On a GPU runtime, run correctness validation first:

```bash
python -m qusimsed.gpu_correctness --qubits 4 --layers 2 \
  --output results/gpu-correctness.json
```

## Run the Streamlit control panel

```bash
streamlit run app/streamlit_app.py
```

Open the local address printed by Streamlit, usually
`http://localhost:8501`.

1. Check the top status banners. Confirm `cuda-streams` and `lightning.gpu`
   before GPU claims.
2. Set qubits, layers, seed, streams, output directory, and trace option.
3. Use **Run CDS scheduler validation** to inspect stream assignment and the
   scheduler workflow.
4. Use **Correctness validation** to collect expectation, gradient, loss, and
   parameter-update errors for Sequential, Batched Parameter-shift, Naive
   Multi-stream, and QuSim-Sed.
5. Use **Baseline comparison** to collect mean gradient time and speedup for
   the same four methods.
6. Use **Real QML benchmarks** for Iris, Wine, Breast Cancer, or PCA-MNIST.
   Keep the sample count and epochs modest for the first run. PCA-MNIST
   downloads and caches MNIST through OpenML, then uses only digits 3 and 5.
7. Use **Memory footprint** to save CDS metadata and live CUDA allocation
   fields. `null` CUDA fields mean unavailable telemetry, not zero memory.
8. Use **Nsight profiling** to run the benchmark command under Nsight Systems.
   Download the manifest and keep the raw `.nsys-rep` file.

Streamlit writes JSON and CSV under the configured `output_dir`. Archive the
configuration, result CSV/JSON, scheduling trace, memory result, and `.nsys-rep`
together for every paper figure.

## Nsight Systems collection

Install Nsight Systems on the GPU host and ensure `nsys` is on `PATH`. On
Windows, launch the terminal as Administrator. This project can invoke it from
the Streamlit tab or command line:

```bash
python -m qusimsed.collectors nsight --output results/nsight/parameter_shift -- \
  python -m qusimsed.gpu_correctness --qubits 4 --layers 2
```

The collector uses `--trace=cuda,nvtx,osrt`. Inspect the timeline for CUDA
streams, kernel intervals, synchronization, idle gaps, and actual overlap.
Compare separate reports for Sequential, Batched Parameter-shift, Naive
Multi-stream, and QuSim-Sed. A manifest with `collected: false` is an
environment failure record, not a profiling result.

## Reproducible experiment order

1. Unit tests and the 4-qubit/2-layer scheduler demo.
2. 4-qubit correctness validation for every strategy.
3. Small baseline and real-data runs with fixed seed.
4. Memory admission sweep before increasing qubits.
5. Nsight runs with a fixed short configuration.
6. Qubit/depth/parameter scaling, then the selected realistic benchmark.

Record the GPU model/VRAM, driver/CUDA versions, Python/PennyLane/JAX/Catalyst
versions, seed, qubits, layers, differentiation method, stream count, warm-up,
timed iterations, and dataset split with each result.

## Sources

Lightning-GPU uses NVIDIA cuQuantum and raises an error rather than providing
a GPU device when required libraries are unavailable. See the official
[Lightning-GPU device guide](https://docs.pennylane.ai/projects/lightning/en/latest/lightning_gpu/device.html)
and [installation guide](https://docs.pennylane.ai/projects/lightning/en/latest/lightning_gpu/installation.html).
Nsight Systems documents CLI collection and CUDA/NVTX traces in its
[User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/). The PCA-MNIST
choice follows PennyLane's [Downscaled MNIST dataset](https://pennylane.ai/datasets/downscaled-mnist),
which is built from PCA-reduced MNIST inputs.
