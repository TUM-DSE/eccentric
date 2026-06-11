"""
plot_noise_models.py
====================
LER vs readout length for the six QEC codes, one PDF per noise model.
Each PDF is (16,4) with a 1x6 row of panels (a)-(f), one per code; lines = code
distance. Shared log y-axis, single right legend, single centered "Lower is better".
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = "experiment_results/noise_models/ler_noise_models.csv"
OUTDIR = "experiment_results/noise_models"

CODES = [("surface", "(a) Surface"), ("bacon", "(b) Bacon-shor"),
         ("color", "(c) Color"), ("hh", "(d) Heavy-hex"),
         ("gross", "(e) Gross"), ("steane", "(f) Steane")]
MODELS = [
    ("heron_herqules",      "ibm_boston gates + HERQULES readout + decoherence"),
    ("readout_decoherence", "HERQULES readout + decoherence (no gate errors)"),
    ("decoherence_only",    "decoherence only (no readout flip)"),
]
DISTS = [5, 7, 9, 11, 12, 13]
COLORS = {5: "#1f77b4", 7: "#ff7f0e", 9: "#2ca02c", 11: "#d62728", 12: "#8c564b", 13: "#9467bd"}
MARKERS = {5: "o", 7: "s", 9: "^", 11: "D", 12: "P", 13: "v"}
RO = [200, 400, 600, 800, 1000]

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, BETTER_FS = 15, 13, 10, 11, 14

ROWS = list(csv.DictReader(open(CSV)))

def series(code, d, model):
    pts = []
    for r in ROWS:
        if r["code"] == code and int(r["distance"]) == d and r["noise_model"] == model:
            v = r["logical_error_rate"]
            y = float(v) if v not in ("", "None") else np.nan
            if y == 0.0:
                y = np.nan          # can't log-plot 0
            pts.append((int(r["readout_ns"]), y))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]

def code_dists(code):
    return sorted({int(r["distance"]) for r in ROWS if r["code"] == code})


def make_figure(model, subtitle):
    fig, axes = plt.subplots(1, 6, figsize=(16, 4), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.055, right=0.925, wspace=0.12)

    for ax, (code, title) in zip(axes, CODES):
        for d in code_dists(code):
            xs, ys = series(code, d, model)
            ax.plot(xs, ys, marker=MARKERS[d], color=COLORS[d], markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5, linewidth=1.6)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=6)
        ax.set_xticks([200, 600, 1000])
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_axisbelow(True)
        ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.55, which="both")
        if ax is axes[0]:
            ax.set_ylabel("Logical error rate (log)", fontsize=LABEL_FS)

    # shared right legend (distances), flush right
    handles = [Line2D([0], [0], color=COLORS[d], marker=MARKERS[d], markersize=6,
                      markeredgecolor="black", markeredgewidth=0.5, linewidth=1.6,
                      label=f"d = {d}") for d in DISTS]
    fig.legend(handles, [f"d = {d}" for d in DISTS], loc="center left",
               bbox_to_anchor=(0.928, 0.5), ncol=1, fontsize=LEGEND_FS,
               title="Distance", title_fontsize=LEGEND_FS)

    # single centered "Lower is better" above the titles, and shared x-label
    fig.text(0.49, 0.88, "Lower is better ↓", ha="center", va="top",
             fontsize=BETTER_FS, fontweight="bold", color="blue")
    fig.text((0.055 + 0.925) / 2, 0.04, "Readout length (ns)", ha="center",
             va="bottom", fontsize=LABEL_FS)

    out = os.path.join(OUTDIR, f"noise_models_ler_{model}.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    for model, sub in MODELS:
        make_figure(model, sub)
