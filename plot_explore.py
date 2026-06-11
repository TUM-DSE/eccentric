"""Plot the surface-code readout-mechanism test: baseline vs backlog vs more-rounds."""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = "experiment_results/noise_models/surface_readout_mechanisms.csv"
OUT = "experiment_results/noise_models/surface_readout_mechanisms.pdf"
PANELS = [("baseline", "(a) Baseline (rounds = d)"),
          ("backlog", "(b) Backlog idle (∝ rounds×readout)"),
          ("rounds_5d", "(c) More rounds (5d)")]
DISTS = [5, 7, 9, 11, 13]
COLORS = {5: "#1f77b4", 7: "#ff7f0e", 9: "#2ca02c", 11: "#d62728", 13: "#9467bd"}
MARKERS = {5: "o", 7: "s", 9: "^", 11: "D", 13: "v"}
RO = [200, 400, 600, 800, 1000]
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, BETTER_FS = 15, 13, 11, 11, 14
ROWS = list(csv.DictReader(open(CSV)))

def series(variant, d):
    pts = []
    for r in ROWS:
        if r["variant"] == variant and int(r["distance"]) == d:
            y = float(r["logical_error_rate"]); y = np.nan if y == 0 else y
            pts.append((int(r["readout_ns"]), y))
    pts.sort(); return [p[0] for p in pts], [p[1] for p in pts]

fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
fig.subplots_adjust(top=0.82, bottom=0.20, left=0.07, right=0.88, wspace=0.10)
for ax, (variant, title) in zip(axes, PANELS):
    for d in DISTS:
        xs, ys = series(variant, d)
        ax.plot(xs, ys, marker=MARKERS[d], color=COLORS[d], markersize=7,
                markeredgecolor="black", markeredgewidth=0.5, linewidth=1.8)
    ax.set_yscale("log"); ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)
    ax.set_xticks([200, 600, 1000]); ax.tick_params(labelsize=TICK_FS)
    ax.set_xlabel("Readout length (ns)", fontsize=LABEL_FS)
    ax.set_axisbelow(True); ax.grid(color="gray", ls="--", lw=0.4, alpha=0.55, which="both")
    if ax is axes[0]:
        ax.set_ylabel("Logical error rate (log)", fontsize=LABEL_FS)
handles = [Line2D([0], [0], color=COLORS[d], marker=MARKERS[d], markersize=7,
                  markeredgecolor="black", markeredgewidth=0.5, linewidth=1.8) for d in DISTS]
fig.legend(handles, [f"d = {d}" for d in DISTS], loc="center left",
           bbox_to_anchor=(0.885, 0.5), ncol=1, fontsize=LEGEND_FS,
           title="Distance", title_fontsize=LEGEND_FS)
fig.text(0.475, 0.94, "Lower is better ↓", ha="center", va="top",
         fontsize=BETTER_FS, fontweight="bold", color="blue")
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=130)
print("saved", OUT)
