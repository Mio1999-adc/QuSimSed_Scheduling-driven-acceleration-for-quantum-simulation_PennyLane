"""CLI collectors for review-ready memory and Nsight artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import build_demo_pool
from .memory import measure_memory, metadata_memory
from .profiling import collect_nsight


def memory_demo(output: str) -> dict:
    """Records CDS metadata plus a live process/CUDA snapshot for construction."""
    pool = build_demo_pool()
    result = {"workload": "CDS construction only (not a PennyLane baseline)",
              "cds_metadata": metadata_memory(pool).__dict__, "live_measurement": measure_memory(build_demo_pool)}
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="kind", required=True)
    memory = sub.add_parser("memory-demo"); memory.add_argument("--output", default="results/memory/cds_memory.json")
    nsight = sub.add_parser("nsight"); nsight.add_argument("--output", default="results/nsight/qusimsed"); nsight.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.kind == "memory-demo": result = memory_demo(args.output)
    else:
        command = args.command[1:] if args.command and args.command[0] == "--" else args.command
        if not command: parser.error("nsight requires a command after --")
        result = collect_nsight(command, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
