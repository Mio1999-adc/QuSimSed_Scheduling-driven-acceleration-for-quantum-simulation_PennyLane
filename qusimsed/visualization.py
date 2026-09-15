from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable


def stream_timeline_svg(trace: Iterable[dict], path: str | Path) -> Path:
    """Render observed trace intervals. Logical-host traces are labelled as such."""
    rows = list(trace); output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='720' height='80'><text x='20' y='40'>No runtime trace collected.</text></svg>", encoding="utf-8")
        return output
    origin = min(row["start_ns"] for row in rows); end = max(row["end_ns"] for row in rows); scale = 600 / max(1, end-origin)
    streams = sorted({row["stream_id"] for row in rows}); height = 55 + 48 * len(streams)
    blocks = [f"<text x='12' y='20'>Observed scheduler trace ({'CUDA-backed' if any(r.get('cuda_stream_id') is not None for r in rows) else 'host-thread; not GPU-overlap evidence'})</text>"]
    colors = {"circuit":"#2864a0", "autograd":"#ba5a31"}
    for index, stream in enumerate(streams):
        y = 38 + index * 48; blocks.append(f"<text x='12' y='{y+16}'>Stream {stream}</text><line x1='100' y1='{y+10}' x2='710' y2='{y+10}' stroke='#ddd'/>")
        for row in rows:
            if row["stream_id"] != stream: continue
            x = 100 + (row["start_ns"]-origin)*scale; width = max(2, (row["end_ns"]-row["start_ns"])*scale)
            label = html.escape(row["task_id"])
            blocks.append(f"<rect x='{x:.1f}' y='{y}' width='{width:.1f}' height='20' fill='{colors.get(row['graph_type'], '#666')}'/><title>{label}</title>")
    output.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='730' height='{height}' font-family='Arial' font-size='12'>{''.join(blocks)}</svg>", encoding="utf-8")
    return output


def scheduler_workflow_svg(path: str | Path) -> Path:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Circuit + AutoGrad graphs", "CDS / RecordPool", "Priority ready queue", "memory + stream check", "asynchronous dispatch", "completion: release resources", "decrement successors / activate ready nodes"]
    parts = ["<rect width='760' height='460' fill='white'/>"]
    for i, label in enumerate(labels):
        y = 18 + i*58; parts.append(f"<rect x='230' y='{y}' width='300' height='34' rx='5' fill='#eaf1fb' stroke='#2864a0'/><text x='245' y='{y+22}' font-size='14'>{label}</text>")
        if i < len(labels)-1: parts.append(f"<path d='M380 {y+34} v20' stroke='#333' marker-end='url(#a)'/>")
    parts.append("<defs><marker id='a' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0,0 L0,6 L6,3 z'/></marker></defs>")
    output.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='760' height='460' font-family='Arial'>{''.join(parts)}</svg>", encoding="utf-8")
    return output
