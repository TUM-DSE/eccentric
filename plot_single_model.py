"""Plot single-model LER CSVs (futuristic / highfid / boston / fixedreadout)
as the 1x6 (16,4) row layout."""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CODES = [("surface", "(a) Surface"), ("bacon", "(b) Bacon-shor"),
         ("color", "(c) Color"), ("hh", "(d) Heavy-hex"),
         ("gross", "(e) Gross"), ("steane", "(f) Steane")]
DISTS = [5, 7, 9, 11, 12, 13]
COLORS = {5: "#1f77b4", 7: "#ff7f0e", 9: "#2ca02c", 11: "#d62728", 12: "#8c564b", 13: "#9467bd"}
MARKERS = {5: "o", 7: "s", 9: "^", 11: "D", 12: "P", 13: "v"}
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, BETTER_FS = 15, 13, 10, 11, 14

PAPER_SIZE = (16, 4)

# (csv path, output pdf path, suptitle)
JOBS = [
    ("experiment_results/noise_models/ler_noise_models_futuristic.csv",
     "experiment_results/noise_models/noise_models_ler_futuristic.pdf",
     "Futuristic (all errors /10, T1/T2 x3) - Model 1"),
    ("experiment_results/noise_models/ler_noise_models_highfid.csv",
     "experiment_results/noise_models/noise_models_ler_highfid.pdf",
     "High-fidelity readout (readout error /10 only) - Model 1"),
    ("experiment_results/noise_models/ler_noise_models_boston_lowt_backlog.csv",
     "experiment_results/noise_models/noise_models_ler_boston_lowt_backlog.pdf",
     "ibm_boston gates, fixed readout 3.54e-3, backlog ON"),
    ("experiment_results/noise_models/ler_noise_models_futuristic_fixedreadout.csv",
     "experiment_results/noise_models/noise_models_ler_futuristic_fixedreadout.pdf",
     "Futuristic gates (/10) + T1/T2 x3 + fixed readout 3.54e-4 (boston/10)"),
]


def plot(csv_path, out, suptitle):
    rows = list(csv.DictReader(open(csv_path)))

    def series(code, d):
        pts = []
        for r in rows:
            if r["code"] == code and int(r["distance"]) == d:
                y = r["logical_error_rate"]
                y = float(y) if y not in ("", "None") else np.nan
                if y == 0.0:
                    y = np.nan
                pts.append((int(r["readout_ns"]), y))
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    def cdists(code):
        return sorted({int(r["distance"]) for r in rows if r["code"] == code})

    fig, axes = plt.subplots(1, 6, figsize=PAPER_SIZE, sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.055, right=0.875, wspace=0.18)
    for ax, (code, title) in zip(axes.flat, CODES):
        for d in cdists(code):
            xs, ys = series(code, d)
            ax.plot(xs, ys, marker=MARKERS[d], color=COLORS[d], markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5, linewidth=1.6)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=6)
        ax.set_xticks([200, 600, 1000])
        ax.tick_params(labelsize=TICK_FS)
        ax.set_axisbelow(True)
        ax.grid(color="gray", ls="--", lw=0.4, alpha=0.55, which="both")
    fig.supylabel("Logical error rate (log)", fontsize=LABEL_FS, x=0.012)
    handles = [Line2D([0], [0], color=COLORS[d], marker=MARKERS[d], markersize=6,
                      markeredgecolor="black", markeredgewidth=0.5, linewidth=1.6) for d in DISTS]
    fig.legend(handles, [f"d = {d}" for d in DISTS], loc="center left",
               bbox_to_anchor=(0.882, 0.5), ncol=1, fontsize=LEGEND_FS,
               title="Distance", title_fontsize=LEGEND_FS)
    fig.text(0.465, 0.965, "Lower is better ↓", ha="center", va="top",
             fontsize=BETTER_FS, fontweight="bold", color="blue")
    fig.text((0.055 + 0.875) / 2, 0.03, "Measurement duration (ns)", ha="center", va="bottom", fontsize=LABEL_FS)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    for c, o, s in JOBS:
        if os.path.exists(c):
            plot(c, o, s)
        else:
            print("missing", c)
