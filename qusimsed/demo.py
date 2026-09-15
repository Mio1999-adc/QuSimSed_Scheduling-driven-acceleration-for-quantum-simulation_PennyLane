"""Small dependency-scheduler experiment that needs no quantum framework.

It validates CDS state transitions, tracing, memory gates, and the distinction
between host concurrency and CUDA evidence. It is not a quantum performance
benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict

from .config import ExperimentConfig
from .core.cds import CDSRecord, RecordPool
from .core.scheduler import ResourceAwareScheduler
from .memory import metadata_memory
from .profiling import nsight_status, trace_summary
from .runtime import detect_runtime
from .visualization import scheduler_workflow_svg, stream_timeline_svg


def build_demo_pool() -> RecordPool:
    pool = RecordPool()
    for node_id, graph, kind, priority in [
        ("encode", "circuit", "encoding", 3), ("shift_0_plus", "circuit", "shift", 2),
        ("shift_0_minus", "circuit", "shift", 2), ("gradient_0", "autograd", "gradient", 1),
        ("update", "autograd", "optimizer", 0),
    ]:
        pool.add(CDSRecord(node_id, graph, kind, memory_bytes=1024, priority=priority))
    pool.add_edge("encode", "shift_0_plus"); pool.add_edge("encode", "shift_0_minus")
    pool.add_edge("shift_0_plus", "gradient_0", cross_graph=True); pool.add_edge("shift_0_minus", "gradient_0", cross_graph=True)
    pool.add_edge("gradient_0", "update")
    return pool


def run_demo(config: ExperimentConfig, *, cuda_smoke: bool = False) -> dict:
    config.validate(); pool = build_demo_pool()
    scheduler = ResourceAwareScheduler(pool, config.streams, memory_budget_bytes=1024 * max(2, config.streams), safety_factor=1.0)
    def task(_: int) -> None:
        time.sleep(0.003)
    runtime = detect_runtime()
    if cuda_smoke and runtime.selected_mode == "cuda-streams":
        from .execution import TorchCUDAStreamExecutor, cuda_smoke_workload
        cuda_task = cuda_smoke_workload(TorchCUDAStreamExecutor(config.streams))
        task_map = {key: cuda_task for key in pool.records}
    else:
        task_map = {key: task for key in pool.records}
    trace = [row.to_dict() for row in scheduler.run(task_map)]
    output = config.results_path; output.mkdir(parents=True, exist_ok=True)
    with (output / "scheduler_trace.json").open("w", encoding="utf-8") as handle: json.dump(trace, handle, indent=2)
    with (output / "scheduler_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace[0].keys()); writer.writeheader(); writer.writerows(trace)
    stream_timeline_svg(trace, output / "stream_timeline.svg"); scheduler_workflow_svg(output / "scheduler_workflow.svg")
    result = {"config": config.metadata(), "system": {"python": platform.python_version(), "platform": platform.platform()},
              "runtime": runtime.metadata(), "scheduler": trace_summary(trace), "memory": asdict(metadata_memory(pool, scheduler)), "nsight": asdict(nsight_status()),
              "artifacts": ["scheduler_trace.json", "scheduler_trace.csv", "stream_timeline.svg", "scheduler_workflow.svg"]}
    (output / "scheduler_demo_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="results/demo"); parser.add_argument("--streams", type=int, default=2); parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args(); result = run_demo(ExperimentConfig(output_dir=args.output_dir, streams=args.streams), cuda_smoke=args.cuda_smoke)
    if args.cuda_smoke and result["runtime"]["selected_mode"] != "cuda-streams": print("CUDA smoke request fell back to CPU worker threads: " + result["runtime"]["reason"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
