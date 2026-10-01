#!/usr/bin/env python3
"""Plot benchmark timings as vector PDFs and PNG previews.

Edit INPUT_CSV and DATASETS below, then run: python3 plot_execution_times.py
Optional override: python3 plot_execution_times.py path/to/data.csv --output-dir output/pdf
Dependency: matplotlib (python3 -m pip install matplotlib).
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Configuration, following datasets-diff-plot.py.
# This CSV contains the measured timings; raw ML datasets are not needed to plot them.
INPUT_CSV = Path("/home/hpclab-lananh/AVQC/Results from qusimsed/max_config_execution_time_data.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pdf"
OUTPUT_STEM = "execution_time_comparison"
DATASETS = ("Synthetic", "Iris", "MNIST")
METHODS = ("PennyLane", "QuSim-Sed")
DIFFERENTIATIONS = ("Parameter-shift", "Adjoint")
COLORS = ("#42658A", "#D78435")


def read_data(path):
    """Validate complete pairs; do not silently aggregate duplicate results."""
    values = {}
    datasets = list(DATASETS)
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("DATASETS must contain unique, nonempty dataset names")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "differentiation", "method", "execution_time_s"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}")
        for row in reader:
            dataset, diff, method = (row[k].strip() for k in
                                     ("dataset", "differentiation", "method"))
            if diff not in DIFFERENTIATIONS or method not in METHODS:
                raise ValueError(f"Unsupported differentiation/method: {diff}/{method}")
            if dataset not in datasets:
                raise ValueError(f"Unexpected dataset: {dataset!r}; expected {datasets}")
            key = (dataset, diff, method)
            if key in values:
                raise ValueError(f"Duplicate measurement: {key}")
            value = float(row["execution_time_s"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Expected positive finite time: {key}")
            values[key] = value
    for dataset in datasets:
        for diff in DIFFERENTIATIONS:
            for method in METHODS:
                if (dataset, diff, method) not in values:
                    raise ValueError(f"Missing measurement: {dataset}/{diff}/{method}")
    return datasets, values


def draw_panel(ax, diff, datasets, values):
    width = 0.34
    maximum = max(values[d, diff, m] for d in datasets for m in METHODS)
    for j, (method, color) in enumerate(zip(METHODS, COLORS)):
        heights = [values[d, diff, method] for d in datasets]
        bars = ax.bar([i + (j - 0.5) * width for i in range(len(datasets))],
                      heights, width, color=color, label=method, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" if diff == "Parameter-shift"
                                  else f"{v:.6f}" for v in heights],
                     padding=5, fontsize=8)
    ax.set(ylabel="Execution time (s)",
           xticks=range(len(datasets)), xticklabels=datasets,
           ylim=(0, maximum * 1.30))
    ax.set_title(diff, loc="left", fontsize=14, fontweight="bold", pad=15)
    ax.yaxis.grid(True, color="#DFE4E9", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB3BD")
    ax.tick_params(length=0, pad=8)
    for i, dataset in enumerate(datasets):
        ratio = values[dataset, diff, METHODS[0]] / values[dataset, diff, METHODS[1]]
        ax.text(i, maximum * 1.19, f"{ratio:.2f}x" if diff == "Parameter-shift"
                else f"{ratio:.4f}x", ha="center", fontsize=10, fontweight="bold")


def make_figure(diffs, datasets, values, output, stem):
    fig, axes = plt.subplots(1, len(diffs), figsize=(7 * len(diffs), 5.2), squeeze=False)
    for ax, diff in zip(axes[0], diffs):
        draw_panel(ax, diff, datasets, values)
    fig.legend(handles=[Patch(facecolor=c, label=m) for c, m in zip(COLORS, METHODS)],
               loc="upper right", bbox_to_anchor=(0.98, 0.99), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.07 if len(diffs) == 2 else 0.11,
                        right=0.98, bottom=0.10, top=0.86, wspace=0.23)
    for extension in ("pdf", "png"):
        target = output / f"{stem}.{extension}"
        fig.savefig(target, dpi=180, facecolor="white")
        print(target.resolve())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=INPUT_CSV,
                        help="Timing CSV (defaults to INPUT_CSV in the configuration)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    datasets, values = read_data(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    for diffs, stem in ((DIFFERENTIATIONS, OUTPUT_STEM),
                        (("Parameter-shift",), "parameter_shift_execution_time"),
                        (("Adjoint",), "adjoint_execution_time")):
        make_figure(diffs, datasets, values, args.output_dir, stem)


if __name__ == "__main__":
    main()
