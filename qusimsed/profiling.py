from __future__ import annotations

import json
import shutil
import subprocess
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
    manifest = {"command": list(command), "output_base": str(base), "nsight": asdict(status), "collected": False}
    if status.available:
        completed = subprocess.run(nsight_command(command, base), cwd=cwd, capture_output=True, text=True)
        manifest.update({"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
                         "report": str(base.with_suffix(".nsys-rep")), "collected": completed.returncode == 0})
    else:
        manifest["reason"] = status.message
    base.with_suffix(".nsight.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


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
