"""
plot_readout_compare.py
=======================
Side-by-side line plots of logical-CNOT LER vs readout length for the two tools
(ECCentric vs lattice-sim), one line per code distance. One PDF per noise model.

Uses the CNN (realistic) readout results:
  experiment_results/readout_compare/{eccentric,artifact}_readout_ler_cnn.csv
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTDIR = "experiment_results/readout_compare"
TOOLS = [("eccentric", "(a) ECCentric"), ("artifact", "(b) lattice-sim")]
NOISE_MODELS = ["current", "futuristic"]
DISTANCES = [5, 7, 9, 11, 13]

# Per-distance colour + marker (distinct colours, distinct markers).
COLORS = {5: "#1f77b4", 7: "#ff7f0e", 9: "#2ca02c", 11: "#d62728", 13: "#9467bd"}
MARKERS = {5: "o", 7: "s", 9: "^", 11: "D", 13: "v"}

# Figure dimensions.
PAPER_SIZE = (16, 4)

# Fonts: titles deliberately LARGER than the in-plot text.
TITLE_FS = 22
LABEL_FS = 16
TICK_FS = 14
LEGEND_FS = 13
BETTER_FS = 16


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path)):
        rows.append(r)
    return rows


DATA = {
    "eccentric": load(os.path.join(OUTDIR, "eccentric_readout_ler_cnn.csv")),
    "artifact":  load(os.path.join(OUTDIR, "artifact_readout_ler_cnn.csv")),
}


def series(tool, d, nm):
    """Return (xs, ys) sorted by readout length; LER==0 -> nan (can't log-plot)."""
    pts = []
    for r in DATA[tool]:
        if int(r["distance"]) == d and r["noise_model"] == nm:
            v = r["logical_error_rate"]
            y = float(v) if v not in ("", "None") else np.nan
            if y == 0.0:
                y = np.nan
            pts.append((int(r["readout_ns"]), y))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def make_figure(nm):
    fig, axes = plt.subplots(1, 2, figsize=PAPER_SIZE, sharey=True)
    fig.subplots_adjust(top=0.78, wspace=0.06, left=0.06, right=0.84, bottom=0.20)

    for ax, (tool, title) in zip(axes, TOOLS):
        for d in DISTANCES:
            xs, ys = series(tool, d, nm)
            ax.plot(xs, ys, marker=MARKERS[d], color=COLORS[d], markersize=8,
                    markeredgecolor="black", markeredgewidth=0.6, linewidth=2,
                    label=f"d = {d}")
        ax.set_yscale("log")
        ax.set_xlabel("Measurement duration (ns)", fontsize=LABEL_FS)
        if ax is axes[0]:                       # shared y-axis -> label once
            ax.set_ylabel("Logical error rate (log)", fontsize=LABEL_FS)
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)
        ax.set_xticks([200, 400, 600, 800, 1000])
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_axisbelow(True)
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6, which="both")

    # Single shared legend, flush to the right edge of both panels (one column).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               ncol=1, fontsize=LEGEND_FS, title="Distance", title_fontsize=LEGEND_FS)

    # Single "Lower is better" annotation, centered above both titles.
    fig.text(0.5, 0.95, "Lower is better ↓", ha="center", va="top",
             fontsize=BETTER_FS, fontweight="bold", color="blue")

    out = os.path.join(OUTDIR, f"readout_ler_cnn_{nm}.pdf")
    fig.savefig(out)
    # also a PNG for quick visual inspection
    fig.savefig(out.replace(".pdf", ".png"), dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    for nm in NOISE_MODELS:
        make_figure(nm)
