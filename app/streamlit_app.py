"""Lightweight experiment control panel.

Launch with `streamlit run app/streamlit_app.py` after installing Streamlit.
It never substitutes a modelled result for a missing GPU experiment.
"""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import replace
from pathlib import Path

from qusimsed.collectors import memory_demo
from qusimsed.config import ExperimentConfig
from qusimsed.demo import run_demo
from qusimsed.profiling import collect_nsight, nsight_status, profile_qusimsed
from qusimsed.result_collection import save_archive
from qusimsed.runtime import detect_runtime


def show_memory(st, result: dict) -> None:
    st.subheader("Memory footprint")
    metadata, live = result["cds_metadata"], result["live_measurement"]
    columns = st.columns(3)
    columns[0].metric("CDS metadata", f"{metadata['cds_bytes']:,} B")
    columns[1].metric("Nodes", metadata["nodes"])
    columns[2].metric("Metadata / node", f"{metadata['bytes_per_node']:,.1f} B")
    before, after = live["before"], live["after"]
    st.dataframe({"metric": ["Process RSS", "CUDA allocated", "CUDA reserved", "CUDA peak allocated"],
                  "before_bytes": [before["process_rss_bytes"], before["cuda_allocated_bytes"], before["cuda_reserved_bytes"], before["cuda_peak_allocated_bytes"]],
                  "after_bytes": [after["process_rss_bytes"], after["cuda_allocated_bytes"], after["cuda_reserved_bytes"], after["cuda_peak_allocated_bytes"]]})
    st.caption("Null CUDA values mean no usable CUDA Python runtime; they are not zero-byte measurements.")


def show_trace_artifacts(st, output_dir: str, result: dict) -> None:
    directory = Path(output_dir)
    timeline, workflow = directory / "stream_timeline.svg", directory / "scheduler_workflow.svg"
    if timeline.exists() and workflow.exists():
        left, right = st.columns(2)
        left.image(str(timeline), caption="Observed task timeline")
        right.image(str(workflow), caption="Scheduler workflow")
    st.download_button("Download scheduler metadata", json.dumps(result, indent=2), "scheduler_demo_metadata.json", "application/json")


def show_result(st, result: dict, key: str) -> None:
    """Render stored measurements, always alongside the configuration that produced them."""
    import pandas as pd
    st.caption(f"Backend: {result.get('backend', 'unavailable')}. Status: {result.get('status', 'legacy result')}.")
    with st.expander("Settings used for this result"):
        st.json(result.get("configuration", {}))
        st.write(result.get("timing_scope", result.get("validation_scope", "")))
    if result.get("reason"):
        st.warning(result["reason"])
    collection = result.get("collection", {})
    if collection:
        st.caption(f"Automatically saved: {collection['directory']}")
    rows = result.get("rows", [])
    if rows:
        frame = pd.DataFrame(rows).set_index("method")
        st.dataframe(frame, use_container_width=True)
        metrics = [name for name in ("mean_iteration_time_ms", "training_seconds", "speedup_vs_sequential",
                   "time_saving_percent", "test_accuracy", "test_loss", "gradient_max_abs",
                   "expectation_max_abs", "parameters_max_abs", "trajectory_max_abs_error") if name in frame]
        if metrics:
            metric = st.selectbox("Result metric", metrics, key=f"{key}_metric")
            st.bar_chart(frame[[metric]])
        raw = [{"iteration": index + 1, "method": row["method"], "elapsed_ms": elapsed}
               for row in rows for index, elapsed in enumerate(row.get("iteration_times_ms", []))]
        if raw:
            st.subheader("Measured iteration times (ms)")
            st.line_chart(pd.DataFrame(raw).pivot(index="iteration", columns="method", values="elapsed_ms"))
    for field, title in (("trajectories", "Training loss by epoch"), ("epoch_times_seconds", "Epoch time (seconds)")):
        if result.get(field):
            st.subheader(title)
            chart = pd.DataFrame({method: pd.Series(values, index=range(1, len(values) + 1))
                                  for method, values in result[field].items()})
            chart.index.name = "epoch"
            st.line_chart(chart)
    if result.get("traces"):
        with st.expander("Scheduling traces"):
            method = st.selectbox("Trace method", list(result["traces"]), key=f"{key}_trace")
            st.dataframe(result["traces"][method], use_container_width=True)
    if result.get("resource_estimates"):
        with st.expander("Automatic resource estimates / manual overrides"):
            st.caption("Admission model inputs and estimates; these are not measured GPU occupancy.")
            st.json(result["resource_estimates"])
    if "profiling" in result:
        with st.expander("Profiling status and artifacts"):
            st.json(result["profiling"])
    st.download_button("Download full result JSON", json.dumps(result, indent=2),
                       "experiment.json", "application/json", key=f"{key}_download")
    for name in ("summary", "errors", "timings", "convergence"):
        path = Path(collection.get("directory", "")) / f"{name}.csv"
        if collection and path.is_file():
            st.download_button(f"Download {name} CSV", path.read_bytes(), f"{name}.csv", "text/csv", key=f"{key}_{name}")


def finish_run(config, result):
    if config.enable_nsight and any(row.get("method") == "QuSim-Sed" for row in result.get("rows", [])):
        if result.get("status") == "validation_failed":
            result['profiling'] = {'collected': False, 'reason': 'Numerical validation failed'}
        else:
            try:
                if config.execution_backend == 'torch-cuda':
                    import torch
                    with torch.cuda.device(config.gpu_device):
                        torch.cuda.synchronize(config.gpu_device)
                        torch.cuda.empty_cache()
                result['profiling'] = profile_qusimsed(config, Path(result['collection']['directory']) / 'profiling' / 'qusimsed')
            except Exception as exc:
                result['profiling'] = {'collected': False, 'reason': f'{type(exc).__name__}: {exc}'}
        save_archive(result, result['collection']['directory'])
    return result


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install streamlit to use this UI: pip install streamlit") from exc
    st.set_page_config(page_title="QuSim-Sed experiments", layout="wide")
    st.title("QuSim-Sed experiment control")
    st.caption("The scheduler demo uses a fixed toy graph; qubits, depth and dataset do not change it. Use the experiment tabs for VQC measurements.")
    left, right = st.columns(2)
    with left:
        qubits = st.number_input("Qubits", 1, 40, 4)
        layers = st.number_input("Circuit depth (layers)", 1, 20, 2)
        differentiation = st.selectbox("Differentiation", ["parameter-shift", "adjoint"])
        st.caption("Correctness and timing use a synthetic VQC. Choose a dataset in the training tab.")
    with right:
        streams = st.number_input("CUDA streams / CPU validation workers", 1, 32, 2)
        execution_backend = st.selectbox("Execution backend", ["torch-cuda", "torch-cpu", "pennylane"])
        manual_sm = st.checkbox("Override automatic SM-demand estimation", value=False)
        sm_demand = st.number_input("Manual SM-demand fraction", 0.01, 1.0, 1.0, disabled=not manual_sm)
        st.caption("Automatic admission estimates per-task demand from state size, gate width and GPU SM count. This is a static model, not measured occupancy; CPU validation uses a conservative fallback.")
        seed = st.number_input("Seed", 0, 2**31 - 1, 7)
        output_dir = st.text_input("Output directory", os.environ.get("QUSIMSED_OUTPUT_DIR", "results/ui"))
        trace = st.checkbox("Record scheduling trace", value=True)
    with st.expander("Advanced experiment settings", expanded=True):
        a, b, c = st.columns(3)
        with a:
            train_all = st.checkbox("Train all rotation parameters", value=True)
            capacity = 3 * int(qubits) * int(layers)
            parameters = st.number_input("Trainable parameters", 1, capacity, capacity,
                                         disabled=train_all, key=f"parameters_{capacity}")
            learning_rate = st.number_input("Learning rate", min_value=0.000001, value=0.05, format="%.6f")
            warmup = st.number_input("Benchmark warm-up iterations", 0, 1000, 1)
        with b:
            gpu_device = st.number_input("GPU device index", 0, 128, 0)
            reference_backend = st.selectbox("Independent correctness reference", ["default.qubit", "lightning.gpu"])
            memory_cap = st.number_input("Memory budget cap (GiB; 0 = automatic)", 0.0, value=0.0)
            safety = st.number_input("Memory safety factor", 0.01, 1.0, 0.8)
        with c:
            partition = st.number_input("Maximum partition nodes", 1, 100000, 16)
            affinity = st.number_input("Subgraph affinity weight", min_value=0.0, value=1.0)
            sync = st.number_input("Synchronization cost weight", min_value=0.0, value=1.0)
            parallelism = st.number_input("Potential parallelism weight", min_value=0.0, value=1.0)
            resource = st.number_input("Resource contention weight", min_value=0.0, value=1.0)
            auto_profile = st.checkbox("Collect companion Nsight profile", value=True)
        st.caption("A smaller trainable count freezes the remaining rotations. Application profiling is currently skipped. Batch size is not configurable: application training uses full-batch updates with sequential sample processing.")
    config = ExperimentConfig(qubits=int(qubits), layers=int(layers), differentiation=differentiation,
                              streams=int(streams), seed=int(seed), output_dir=output_dir,
                              execution_backend=execution_backend, sm_demand=float(sm_demand) if manual_sm else None, enable_trace=trace,
                              trainable_parameters=None if train_all else int(parameters), learning_rate=float(learning_rate),
                              warmup=int(warmup), gpu_device=int(gpu_device), reference_backend=reference_backend,
                              memory_budget_bytes=int(memory_cap * (1 << 30)) if memory_cap else None,
                              memory_safety_factor=float(safety), partition_size=int(partition),
                              affinity_weight=float(affinity), sync_weight=float(sync), parallelism_weight=float(parallelism),
                              resource_weight=float(resource), enable_nsight=auto_profile)
    from qusimsed.pennylane_experiments import methods_for
    st.caption("Available comparison methods: " + ", ".join(methods_for(config)) + ". Catalyst combinations are not integrated.")
    with st.expander("Shared configuration preview"):
        st.json(config.metadata())
    st.caption("Results are collected automatically. Changing controls does not change an already displayed result; inspect its saved settings.")
    st.info(nsight_status().message)
    runtime = detect_runtime()
    st.info(f"Execution mode: {runtime.selected_mode}. {runtime.reason}")
    if st.button("Run CDS scheduler validation"):
        try:
            result = run_demo(config)
        except Exception as exc:
            st.error(str(exc)); return
        st.success("Scheduler validation complete")
        st.caption("CUDA tasks: %d. A value of 0 means this run is not GPU-overlap evidence." % result["scheduler"]["cuda_tasks"])
        st.json(result["scheduler"])
        show_trace_artifacts(st, output_dir, result)

    st.divider()
    correctness, baselines, applications, memory, profiling, history = st.tabs(["Correctness validation", "Baseline comparison", "Real QML benchmarks", "Memory footprint", "Nsight profiling", "Saved results"])
    with correctness:
        st.caption("Independent PennyLane reference; identical inputs and MSE/SGD update. Validates the selected differentiation method across supported execution methods.")
        if st.button("Run correctness validation for all methods"):
            try:
                from qusimsed.pennylane_experiments import correctness_experiment, save_result
                run_config = config
                result = finish_run(run_config, correctness_experiment(run_config))
                path = save_result(result, Path(output_dir) / "correctness" / "correctness.json")
                st.success(f"Saved {path}"); st.session_state["correctness"] = result
            except Exception as exc:
                st.error(str(exc))
        if "correctness" in st.session_state:
            show_result(st, st.session_state["correctness"], "correctness")
    with baselines:
        st.caption("Torch compares Sequential and QuSim-Sed using identical kernels. PennyLane batching is a separate backend. Timings include graph construction, execution, gradients, and update.")
        iterations = st.number_input("Benchmark iterations", 1, 1000, 30, key="baseline_iterations")
        if st.button("Run baseline comparison"):
            try:
                from qusimsed.pennylane_experiments import baseline_experiment, save_result
                run_config = replace(config, iterations=int(iterations))
                result = finish_run(run_config, baseline_experiment(run_config))
                path = save_result(result, Path(output_dir) / "baselines" / "baseline_comparison.json")
                st.success(f"Saved {path}"); st.session_state["baselines"] = result
            except Exception as exc:
                st.error(str(exc))
        if "baselines" in st.session_state:
            show_result(st, st.session_state["baselines"], "baselines")
    with applications:
        st.caption("Application validation is secondary to the controlled synthetic benchmark. PCA-MNIST uses digits 3 vs 5 and fits PCA after splitting train/test data.")
        selected_dataset = st.selectbox("Application dataset", ["iris", "wine", "breast-cancer", "mnist-pca"])
        application_samples = st.number_input("Training samples", 10, 2000, 50, key="application_samples")
        application_iterations = st.number_input("Training epochs", 1, 100, 3, key="application_iterations")
        if st.button("Run real QML benchmark"):
            try:
                from qusimsed.real_benchmarks import run_real_benchmark, save_real_result
                run_config = replace(config, workload=selected_dataset, samples=int(application_samples), iterations=int(application_iterations))
                result = finish_run(run_config, run_real_benchmark(run_config))
                path = save_real_result(result, Path(output_dir) / "applications" / f"{selected_dataset}.json")
                st.success(f"Saved {path}"); st.session_state["application"] = result
            except Exception as exc:
                st.error(str(exc))
        if "application" in st.session_state:
            show_result(st, st.session_state["application"], "application")
    with memory:
        st.caption("Measures CDS metadata and live process/CUDA fields. This CDS-only collector is not a PennyLane baseline.")
        if st.button("Collect memory footprint"):
            path = str(Path(output_dir) / "memory" / "cds_memory.json")
            try:
                result = memory_demo(path)
                st.success(f"Saved {path}")
                show_memory(st, result)
                st.download_button("Download memory JSON", json.dumps(result, indent=2), "cds_memory.json", "application/json")
            except Exception as exc:
                st.error(str(exc))
    with profiling:
        st.caption("Collects an `.nsys-rep` only when Nsight Systems is installed. Otherwise it saves an unavailable manifest.")
        command = st.text_input("Benchmark command", "python -m qusimsed.gpu_correctness --qubits 4 --layers 2")
        if st.button("Collect Nsight profile"):
            try:
                result = collect_nsight(shlex.split(command), Path(output_dir) / "nsight" / "profile")
                if result["collected"]:
                    st.success("Nsight profile collected")
                else:
                    st.warning(result.get("reason", result.get("stderr", "Nsight collection failed")))
                st.json(result)
                st.download_button("Download Nsight manifest", json.dumps(result, indent=2), "profile.nsight.json", "application/json")
            except ValueError as exc:
                st.error(f"Invalid command: {exc}")
            except Exception as exc:
                st.error(str(exc))

    with history:
        st.caption("Browse automatically archived runs under the current output directory without rerunning training.")
        st.button("Refresh saved results")
        st.caption("A CLI run appears as running until its final results are saved. UI controls configure new runs; they do not modify training already running in another process.")
        root = Path(output_dir) / "experiments"
        files = sorted(root.glob("*/result.json"), reverse=True) if root.exists() else []
        if not files:
            st.info("No archived runs in this output directory yet.")
        else:
            selected = st.selectbox("Saved run", files, format_func=lambda path: path.parent.name)
            try:
                show_result(st, json.loads(selected.read_text()), "history")
            except (OSError, ValueError, KeyError) as exc:
                st.error(f"Could not load saved result: {exc}")


if __name__ == "__main__": main()
