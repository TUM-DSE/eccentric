"""
plot_discriminators.py
======================
How do different readout DISCRIMINATORS affect the LER?

Four six-panel figures (one per requested distance d=5, 7, 9, 12), 2x3 grid, one
panel per QEC code. Lines are DISCRIMINATORS (not distances). x = readout length,
y = logical error rate (log).

Data:
  - 6 discriminators (Linear Threshold, Baseline FNN, HERQULES, QubiCML, KLiNQ,
    MCMit-T) from experiment_results/discriminators/ler_discriminators_d{D}.csv.
  - MCMit-CNN ("CNN") line:
      * d=5/7/9: reused from the existing `heron_cnn` model in
        experiment_results/noise_models/ler_noise_models.csv (same noise model, the
        0.910-gmean5 CNN readout table) -- NOT re-run.
      * d=12: heron_cnn has no d=12 data, so CNN was re-run at d=12 with the SAME
        0.910 table (LER_INCLUDE_CNN=1) and lives in ler_discriminators_d12.csv.

Distance handling per figure: surface/bacon/color/hh use the requested D; gross is
the fixed [[144,12,12]] code (always d=12); steane is pinned to d=9 (concat-Steane
supports only d in {3,9,27}). So the gross/steane panels are identical across all
four figures. color has no even-distance code -> its d=12 panel is left blank.
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

DISC_CSV = "experiment_results/discriminators/ler_discriminators_d{D}.csv"
CNN_CSV = "experiment_results/noise_models/ler_noise_models.csv"
OUTDIR = "experiment_results/discriminators"

CODES = [("surface", "(a) Surface"), ("bacon", "(b) Bacon-shor"),
         ("color", "(c) Color"), ("hh", "(d) Heavy-hex"),
         ("gross", "(e) Gross"), ("steane", "(f) Steane")]

# gross/steane have fixed distances; everything else uses the requested D.
def code_dist(code, D):
    return 12 if code == "gross" else 9 if code == "steane" else D

# Line order and styling (discriminators). CNN is the merged-in existing result.
DISCRIMINATORS = ["Linear Threshold", "Baseline FNN", "HERQULES",
                  "QubiCML", "KLiNQ", "MCMit-T", "MCMit-CNN"]
COLORS = {
    "Linear Threshold": "#1f77b4", "Baseline FNN": "#ff7f0e", "HERQULES": "#2ca02c",
    "QubiCML": "#d62728", "KLiNQ": "#9467bd", "MCMit-T": "#8c564b",
    "MCMit-CNN": "#000000",
}
MARKERS = {
    "Linear Threshold": "o", "Baseline FNN": "s", "HERQULES": "^",
    "QubiCML": "D", "KLiNQ": "v", "MCMit-T": "P", "MCMit-CNN": "*",
}
RO = [200, 400, 600, 800, 1000]

FIG_SIZE = (10, 4)
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, BETTER_FS = 15, 13, 10, 10.5, 14


def load(path):
    return list(csv.DictReader(open(path)))


def series(rows, code, d, disc, disc_key, val_key):
    """(xs, ys) for one (code, distance, discriminator) line; 0 -> nan for log."""
    pts = []
    for r in rows:
        if r["code"] == code and int(r["distance"]) == d and r[disc_key] == disc:
            v = r["logical_error_rate"]
            y = float(v) if v not in ("", "None") else np.nan
            if y == 0.0:
                y = np.nan
            pts.append((int(r["readout_ns"]), y))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def cnn_series(code, d, disc_rows, cnn_rows):
    """CNN line: prefer the existing heron_cnn data (d=5/7/9, gross d=12, steane d=9);
    fall back to a re-run MCMit-CNN row in the discriminator CSV (the d=12 case)."""
    xs, ys = series(cnn_rows, code, d, "heron_cnn", "noise_model", "logical_error_rate")
    if xs:
        return xs, ys
    return series(disc_rows, code, d, "MCMit-CNN", "discriminator", "logical_error_rate")


# Side-by-side 1x4 panel sets.
#   d=5 : the four tunable codes (gross/steane dropped -- larger distance anyway).
#   d=12: codes genuinely at d=12 -- surface/bacon/hh + gross ([[144,12,12]]);
#         color omitted (no even-distance run) and steane omitted (d=9).
CODES_4 = [("surface", "(a) Surface"), ("bacon", "(b) Bacon-shor"),
           ("color", "(c) Color"), ("hh", "(d) Heavy-hex")]
CODES_4_D12 = [("surface", "(a) Surface"), ("bacon", "(b) Bacon-shor"),
               ("hh", "(c) Heavy-hex"), ("gross", "(d) Gross")]


def make_figure_4panel(D, disc_rows, cnn_rows, codes=CODES_4, out_name=None):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.06, right=0.845, wspace=0.12)

    for ax, (code, title) in zip(axes.flat, codes):
        d = code_dist(code, D)
        for disc in DISCRIMINATORS:
            if disc == "MCMit-CNN":
                xs, ys = cnn_series(code, d, disc_rows, cnn_rows)
            else:
                xs, ys = series(disc_rows, code, d, disc, "discriminator", "logical_error_rate")
            if not xs:
                continue
            ax.plot(xs, ys, marker=MARKERS[disc], color=COLORS[disc], markersize=5.5,
                    markeredgecolor="black", markeredgewidth=0.5, linewidth=1.5)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=6)
        ax.set_xticks([200, 600, 1000])
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_axisbelow(True)
        ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.55, which="both")

    fig.supylabel("Logical error rate (log)", fontsize=LABEL_FS, x=0.01)

    handles = [Line2D([0], [0], color=COLORS[s], marker=MARKERS[s], markersize=6,
                      markeredgecolor="black", markeredgewidth=0.5, linewidth=1.6,
                      label=s) for s in DISCRIMINATORS]
    fig.legend(handles, DISCRIMINATORS, loc="center left",
               bbox_to_anchor=(0.85, 0.5), ncol=1, fontsize=LEGEND_FS,
               title="Discriminator", title_fontsize=LEGEND_FS)

    fig.text((0.06 + 0.845) / 2, 0.965, "Lower is better ↓", ha="center", va="top",
             fontsize=BETTER_FS, fontweight="bold", color="blue")
    fig.text((0.06 + 0.845) / 2, 0.02, "Measurement duration (ns)", ha="center",
             va="bottom", fontsize=LABEL_FS)

    out = os.path.join(OUTDIR, f"{out_name or f'ler_discriminators_d{D}'}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} (4-panel)")


def make_figure(D, disc_rows, cnn_rows):
    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE, sharex=True, sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.085, right=0.79,
                        wspace=0.10, hspace=0.38)

    for ax, (code, title) in zip(axes.flat, CODES):
        d = code_dist(code, D)
        for disc in DISCRIMINATORS:
            if disc == "MCMit-CNN":
                xs, ys = cnn_series(code, d, disc_rows, cnn_rows)
            else:
                xs, ys = series(disc_rows, code, d, disc, "discriminator", "logical_error_rate")
            if not xs:
                continue
            ax.plot(xs, ys, marker=MARKERS[disc], color=COLORS[disc], markersize=5.5,
                    markeredgecolor="black", markeredgewidth=0.5, linewidth=1.5)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())   # drop cluttered minor labels
        ttl = title + (f"  (d={d})" if code in ("gross", "steane") else "")
        ax.set_title(ttl, fontsize=TITLE_FS - (2 if code in ("gross", "steane") else 0),
                     fontweight="bold", pad=6)
        ax.set_xticks([200, 600, 1000])
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_axisbelow(True)
        ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.55, which="both")

    fig.supylabel("Logical error rate (log)", fontsize=LABEL_FS, x=0.015)

    handles = [Line2D([0], [0], color=COLORS[s], marker=MARKERS[s], markersize=6,
                      markeredgecolor="black", markeredgewidth=0.5, linewidth=1.6,
                      label=s) for s in DISCRIMINATORS]
    fig.legend(handles, DISCRIMINATORS, loc="center left",
               bbox_to_anchor=(0.795, 0.5), ncol=1, fontsize=LEGEND_FS,
               title="Discriminator", title_fontsize=LEGEND_FS)

    fig.text(0.435, 0.965, f"Lower is better ↓   (tunable codes d={D})",
             ha="center", va="top", fontsize=BETTER_FS, fontweight="bold", color="blue")
    fig.text((0.085 + 0.79) / 2, 0.02, "Measurement duration (ns)", ha="center",
             va="bottom", fontsize=LABEL_FS)

    out = os.path.join(OUTDIR, f"ler_discriminators_d{D}.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    cnn_rows = load(CNN_CSV)
    # gross (d=12) and steane (d=9) were run once (in the d=5 sweep) and reused in
    # every figure -- the d=9/d=12 CSVs omit them.
    reuse = [r for r in load(DISC_CSV.format(D=5))
             if r["code"] in ("gross", "steane")]
    for D in (5, 7, 9, 12):
        own = load(DISC_CSV.format(D=D))
        have = {r["code"] for r in own}
        disc_rows = own + [r for r in reuse if r["code"] not in have]
        if D == 5:
            # d=5: 1x4 side-by-side, tunable codes only (gross/steane dropped).
            make_figure_4panel(D, disc_rows, cnn_rows, CODES_4)
        elif D == 12:
            # d=12: 1x4 side-by-side, surface/bacon/hh + gross; color & steane dropped.
            make_figure_4panel(D, disc_rows, cnn_rows, CODES_4_D12)
        else:
            make_figure(D, disc_rows, cnn_rows)

    # Futuristic d=12 (errors/10, T1/T2 x3): self-contained CSV with re-run CNN for
    # all four codes (surface/bacon/hh/gross). Pass cnn_rows=[] so the CNN line comes
    # from THIS CSV, not the current-noise heron_cnn (which would mis-plot gross=1.0).
    fut = DISC_CSV.format(D=12).replace(".csv", "_futuristic.csv")
    if os.path.exists(fut):
        make_figure_4panel(12, load(fut), [], CODES_4_D12,
                           out_name="ler_discriminators_d12_futuristic")
