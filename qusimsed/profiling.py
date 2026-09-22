from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class NsightStatus:
    available: bool
    executable: str | None
    message: str


def nsight_status() -> NsightStatus:
    executable = shutil.which("nsys")
    if executable:
        return NsightStatus(True, executable, "Nsight Systems detected")
    return NsightStatus(False, None, "Nsight Systems (nsys) is not on PATH; no kernel-overlap claim can be made")


def nsight_command(command: Sequence[str], output: str | Path) -> list[str]:
    status = nsight_status()
    if not status.available:
        raise RuntimeError(status.message)
    return [status.executable, "profile", "--trace=cuda,nvtx,osrt", "--force-overwrite=true", "-o", str(output), *command]


def collect_nsight(command: Sequence[str], output: str | Path, *, cwd: str | Path | None = None) -> dict:
    """Collect an Nsight report or save an explicit unavailable manifest."""
    base = Path(output); base.parent.mkdir(parents=True, exist_ok=True); status = nsight_status()
    report = Path(str(base) + ".nsys-rep")
    manifest_path = Path(str(base) + ".nsight.json")
    manifest = {"command": list(command), "output_base": str(base), "nsight": asdict(status),
                "manifest": str(manifest_path), "report": str(report), "collected": False}
    if status.available:
        before = report.stat() if report.exists() else None
        environment = dict(os.environ, QUSIMSED_UNDER_NSYS="1")
        try:
            completed = subprocess.run(nsight_command(command, base), cwd=cwd, capture_output=True,
                                       text=True, env=environment)
            after = report.stat() if report.exists() else None
            fresh = after is not None and after.st_size > 0 and (
                before is None or (before.st_mtime_ns, before.st_size, before.st_ino)
                != (after.st_mtime_ns, after.st_size, after.st_ino))
            manifest.update(returncode=completed.returncode, stdout=completed.stdout,
                            stderr=completed.stderr, collected=completed.returncode == 0 and fresh)
            if not manifest["collected"]:
                manifest["reason"] = "Nsight or its target failed, or no new nonempty report was produced"
        except OSError as exc:
            manifest["reason"] = f"Could not launch Nsight: {exc}"
    else:
        manifest["reason"] = status.message
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def profile_qusimsed(config, output: str | Path) -> dict:
    """One companion VQC trace; never instrument the reported timing samples.

    Called after the primary run. Child flags and the collector environment
    both prevent recursive/nested profiling. Uses the same Python environment.
    """
    base = Path(output).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    trace = Path(str(base) + "-trace.json")
    manifest_path = Path(str(base) + ".nsight.json")
    reason = None
    if os.environ.get("QUSIMSED_UNDER_NSYS") == "1":
        reason = "Already running under the Nsight collector; nested profiling skipped"
    elif config.execution_backend != "torch-cuda":
        reason = "Automatic CUDA profiling requires the torch-cuda backend"
    elif config.workload != "synthetic":
        reason = "The companion server trace supports synthetic VQC workloads only"
    if reason:
        result = {"collected": False, "reason": reason, "manifest": str(manifest_path),
                  "configuration": config.metadata(), "kind": "companion-qusimsed-trace"}
        manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    command = [sys.executable, "-m", "qusimsed.server_benchmark", "--mode", "trace",
               "--strategy", "qusimsed", "--no-profile", "--output", str(trace)]
    for flag, value in [("qubits", config.qubits), ("layers", config.layers),
                        ("differentiation", config.differentiation), ("streams", config.streams),
                        ("gpu-device", config.gpu_device), ("sm-demand", config.sm_demand),
                        ("memory-safety-factor", config.memory_safety_factor),
                        ("partition-size", config.partition_size), ("affinity-weight", config.affinity_weight),
                        ("sync-weight", config.sync_weight), ("parallelism-weight", config.parallelism_weight),
                        ("resource-weight", config.resource_weight), ("seed", config.seed)]:
        if value is not None:
            command.extend(["--" + flag, str(value)])
    if config.memory_budget_bytes is not None:
        command.extend(["--memory-gib", str(config.memory_budget_bytes / (1 << 30))])
    if config.trainable_parameters is not None:
        command.extend(["--parameters", str(config.trainable_parameters)])
    command.extend(["--reference-backend", config.reference_backend])
    result = collect_nsight(command, base)
    result.update(configuration=config.metadata(), kind="companion-qusimsed-trace",
                  scheduler_trace=str(trace), timing_samples_profiled=False)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def trace_summary(trace: list[dict]) -> dict:
    """Host timeline summary; CUDA fields only exist when the executor supplied them."""
    cuda_tasks = [row for row in trace if row.get("cuda_stream_id") is not None]
    intervals = [(row["start_ns"], row["end_ns"]) for row in trace]
    total = sum(max(0, end - start) for start, end in intervals)
    makespan = max((end for _, end in intervals), default=0) - min((start for start, _ in intervals), default=0)
    return {"tasks": len(trace), "cuda_tasks": len(cuda_tasks), "logical_streams": len({r["stream_id"] for r in trace}),
            "cuda_streams": len({r["cuda_stream_id"] for r in cuda_tasks}), "sum_task_ns": total,
            "makespan_ns": makespan, "host_overlap_ratio": (total / makespan if makespan else 0.0),
            "gpu_evidence": bool(cuda_tasks)}
