"""Lightweight experiment control panel.

Launch with `streamlit run app/streamlit_app.py` after installing Streamlit.
It never substitutes a modelled result for a missing GPU experiment.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from qusimsed.collectors import memory_demo
from qusimsed.config import ExperimentConfig
from qusimsed.demo import run_demo
from qusimsed.profiling import collect_nsight, nsight_status
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


def show_rows(st, result: dict, *, chart_column: str, title: str) -> None:
    st.subheader(title)
    st.dataframe(result["rows"], use_container_width=True)
    st.bar_chart({row["method"]: row[chart_column] for row in result["rows"]})
    st.caption(f"Backend: {result['backend']}. Mode: {result['runtime']['selected_mode']}.")


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install streamlit to use this UI: pip install streamlit") from exc
    st.set_page_config(page_title="QuSim-Sed experiments", layout="wide")
    st.title("QuSim-Sed experiment control")
    st.caption("The scheduler demo validates CDS behavior. It is not a quantum-performance result.")
    left, right = st.columns(2)
    with left:
        qubits = st.number_input("Qubits", 1, 40, 4)
        layers = st.number_input("Layers", 1, 20, 2)
        differentiation = st.selectbox("Differentiation", ["parameter-shift", "adjoint"])
        workload = st.selectbox("Workload", ["synthetic", "iris", "wine", "breast-cancer", "mnist-pca"])
    with right:
        streams = st.number_input("Logical streams", 1, 32, 2)
        seed = st.number_input("Seed", 0, 2**31 - 1, 7)
        output_dir = st.text_input("Output directory", "results/ui")
        trace = st.checkbox("Record scheduling trace", value=True)
    st.info(nsight_status().message)
    runtime = detect_runtime()
    st.info(f"Execution mode: {runtime.selected_mode}. {runtime.reason}")
    if st.button("Run CDS scheduler validation"):
        config = ExperimentConfig(qubits=int(qubits), layers=int(layers), differentiation=differentiation,
                                  workload=workload, streams=int(streams), seed=int(seed), output_dir=output_dir,
                                  enable_trace=trace)
        try:
            result = run_demo(config)
        except Exception as exc:
            st.error(str(exc)); return
        st.success("Scheduler validation complete")
        st.caption("CUDA tasks: %d. A value of 0 means this run is not GPU-overlap evidence." % result["scheduler"]["cuda_tasks"])
        st.json(result["scheduler"])
        show_trace_artifacts(st, output_dir, result)

    st.divider()
    correctness, baselines, applications, memory, profiling = st.tabs(["Correctness validation", "Baseline comparison", "Real QML benchmarks", "Memory footprint", "Nsight profiling"])
    with correctness:
        st.caption("Same parameters, circuit, inputs, shift rule, and update rule for every method. Sequential is the reference.")
        if st.button("Run correctness validation for all methods"):
            try:
                from qusimsed.pennylane_experiments import correctness_experiment, save_result
                config = ExperimentConfig(qubits=int(qubits), layers=int(layers), differentiation=differentiation,
                                          workload=workload, streams=int(streams), seed=int(seed), output_dir=output_dir,
                                          learning_rate=0.05)
                result = correctness_experiment(config)
                path = save_result(result, Path(output_dir) / "correctness" / "correctness.json")
                st.success(f"Saved {path}"); st.session_state["correctness"] = result
            except Exception as exc:
                st.error(str(exc))
        if "correctness" in st.session_state:
            show_rows(st, st.session_state["correctness"], chart_column="gradient_max_abs", title="Maximum gradient absolute error")
            st.download_button("Download correctness JSON", json.dumps(st.session_state["correctness"], indent=2), "correctness.json", "application/json")
    with baselines:
        st.caption("All baselines execute the same full-width VQC. The methods differ only in dispatch: sequential, device batch, naïve independent workers, or CDS scheduling.")
        iterations = st.number_input("Benchmark iterations", 1, 1000, 10, key="baseline_iterations")
        if st.button("Run baseline comparison"):
            try:
                from qusimsed.pennylane_experiments import baseline_experiment, save_result
                config = ExperimentConfig(qubits=int(qubits), layers=int(layers), differentiation=differentiation,
                                          workload=workload, streams=int(streams), seed=int(seed), output_dir=output_dir,
                                          iterations=int(iterations))
                result = baseline_experiment(config)
                path = save_result(result, Path(output_dir) / "baselines" / "baseline_comparison.json")
                st.success(f"Saved {path}"); st.session_state["baselines"] = result
            except Exception as exc:
                st.error(str(exc))
        if "baselines" in st.session_state:
            show_rows(st, st.session_state["baselines"], chart_column="speedup_vs_sequential", title="Speedup versus sequential")
            st.download_button("Download baseline JSON", json.dumps(st.session_state["baselines"], indent=2), "baseline_comparison.json", "application/json")
    with applications:
        st.caption("Application validation is secondary to the controlled synthetic benchmark. PCA-MNIST uses digits 3 vs 5 and fits PCA after splitting train/test data.")
        selected_dataset = st.selectbox("Application dataset", ["iris", "wine", "breast-cancer", "mnist-pca"])
        application_samples = st.number_input("Training samples", 10, 2000, 50, key="application_samples")
        application_iterations = st.number_input("Training epochs", 1, 100, 3, key="application_iterations")
        if st.button("Run real QML benchmark"):
            try:
                from qusimsed.real_benchmarks import run_real_benchmark, save_real_result
                config = ExperimentConfig(qubits=int(qubits), layers=int(layers), differentiation=differentiation,
                                          workload=selected_dataset, streams=int(streams), seed=int(seed), output_dir=output_dir,
                                          samples=int(application_samples), iterations=int(application_iterations))
                result = run_real_benchmark(config)
                path = save_real_result(result, Path(output_dir) / "applications" / f"{selected_dataset}.json")
                st.success(f"Saved {path}"); st.session_state["application"] = result
            except Exception as exc:
                st.error(str(exc))
        if "application" in st.session_state:
            result = st.session_state["application"]
            st.write(result["dataset_description"])
            st.dataframe(result["rows"], use_container_width=True)
            left, right = st.columns(2)
            left.bar_chart({row["method"]: row["test_accuracy"] for row in result["rows"]})
            right.bar_chart({row["method"]: row["speedup_vs_sequential"] for row in result["rows"]})
            st.download_button("Download application benchmark JSON", json.dumps(result, indent=2), "application_benchmark.json", "application/json")
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


if __name__ == "__main__": main()
