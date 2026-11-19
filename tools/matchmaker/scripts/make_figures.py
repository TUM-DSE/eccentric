"""The README's figures, in a light and a dark version each (docs/figures/<name>-{light,dark}.png).

python scripts/make_figures.py [--data data] [--out docs/figures] [--jobs 4]

Held-out predictions come from qec_matchmaker.evaluate.cross_val_predictions (10 folds over codes, intervals
conformalized as in reports/REPORT.md) and are cached in experiments/cache/ under the dataset's hash.
"""
import argparse
import hashlib
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from qec_matchmaker.dataset import load_dataset
from qec_matchmaker.evaluate import COMPLETE, cross_val_predictions
from qec_matchmaker.models import LER_FLOOR, circuit_labels
from qec_matchmaker.recommend import recommend

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"mwpm": "MWPM", "mwpf": "MWPF", "bposd": "BP-OSD", "bposd_tuned": "BP-OSD (tuned)", "bplsd": "BP-LSD",
         "belief_find": "BeliefFind", "relay_bp": "Relay-BP", "tesseract": "Tesseract", "chromobius": "Chromobius"}
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  accent="#2a78d6", neutral="#898781", ramp=BLUE[:]),
    "dark": dict(surface="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 accent="#3987e5", neutral="#898781", ramp=BLUE[::-1][1:] + ["#e6f0fd"]),
}


def style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.8, "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": t["ink"],
        "axes.titlelocation": "left", "axes.titlepad": 10, "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "mathtext.fontset": "custom",
        "mathtext.rm": "sans", "mathtext.it": "sans", "mathtext.default": "regular",
    })
    return t


def predictions(df, jobs):
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]
    path = os.path.join(HERE, "experiments", "cache", f"cv_code_{digest}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    pred = cross_val_predictions(df, folds=10, calibrate=True, n_jobs=jobs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pred.to_pickle(path)
    return pred


def decades(ax, axis="both"):
    from matplotlib.ticker import FuncFormatter

    def f(v, _):
        e = int(round(v))
        return {0: "1", 1: "10", 2: "100", 3: "1000"}.get(e, f"$10^{{{e}}}$")
    if axis in ("both", "x"):
        ax.xaxis.set_major_formatter(FuncFormatter(f))
    if axis in ("both", "y"):
        ax.yaxis.set_major_formatter(FuncFormatter(f))


def parity(ax, x, y, t, lo, hi, title, xlabel, ylabel, note):
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    ax.hexbin(x, y, gridsize=55, extent=(lo, hi, lo, hi), bins="log", mincnt=1, cmap=cmap, linewidths=0.2,
              edgecolors=t["surface"])
    ax.plot([lo, hi], [lo, hi], color=t["ink2"], lw=1, zorder=3)
    for k in (-1, 1):
        ax.plot([lo, hi], [lo + k * np.log10(3), hi + k * np.log10(3)], color=t["muted"], lw=0.8, ls=(0, (3, 3)),
                zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    decades(ax)
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=t["ink2"],
            linespacing=1.5)


def fig_accuracy(df, pred, theme, out):
    t = style(theme)
    labels = circuit_labels(df)
    labels = labels[labels.best_status == "measured"]
    first = pred.groupby("circuit_key").first().reindex(labels.index)
    y_ler = np.log10(np.clip(labels.best_ler.to_numpy(), LER_FLOOR, 1))
    q = np.log10(first[["ler_q10", "ler_q50", "ler_q90"]].to_numpy(float))
    ler_err = np.median(np.abs(q[:, 1] - y_ler))
    ler_cov = np.mean((q[:, 0] <= y_ler) & (y_ler <= q[:, 2]))

    tm = df[df.ms_per_shot > 0][["circuit_key", "decoder", "ms_per_shot"]].merge(
        pred[["circuit_key", "decoder", "ms_q50"]], on=["circuit_key", "decoder"]).dropna()
    y_t, p_t = np.log10(tm.ms_per_shot.to_numpy()), np.log10(tm.ms_q50.to_numpy())
    within = np.mean(np.abs(p_t - y_t) <= np.log10(3))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    parity(axes[0], y_ler, q[:, 1], t, -6.3, 0.2, "Logical error rate of the best decoder",
           "measured", "predicted",
           f"{len(y_ler):,} circuits of held-out codes\nmedian error {10 ** ler_err:.2f}×\n"
           f"{100 * ler_cov:.0f}% inside the 80% interval")
    lo, hi = np.floor(min(y_t.min(), p_t.min())), np.ceil(max(y_t.max(), p_t.max()))
    parity(axes[1], y_t, p_t, t, lo, hi, "Decoding time per shot (ms)", "measured", "predicted",
           f"{len(y_t):,} decodes, 9 decoders\n{100 * within:.0f}% within 3× (dashed)")
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(out, f"accuracy-{theme}.png"), dpi=200)
    plt.close(fig)


def outcomes(df, pred):
    """Per circuit on which all of COMPLETE were measured: each policy's decoder good enough?, its time over the
    cheapest good-enough decoder's"""
    m = df[(df.status == "measured") & (df.ms_per_shot > 0)]
    m = m[m.circuit_key.isin(set(pred.circuit_key))]
    predicted = dict(tuple(pred.groupby("circuit_key", sort=False)))
    rows = []
    for key, s in m.groupby("circuit_key", sort=False):
        ler, ms = dict(zip(s.decoder, s.ler)), dict(zip(s.decoder, s.ms_per_shot))
        if not COMPLETE <= set(ler):
            continue
        best = s.loc[s.ler.idxmin()]
        good = set(s[~(s.ler_low > best.ler_high)].decoder)
        cheapest = s[s.decoder.isin(good)].ms_per_shot.min()
        for name, d in {"Matchmaker, accurate": recommend(predicted[key], "accurate").decoder,
                        "Matchmaker, cheapest": recommend(predicted[key], "cheapest").decoder,
                        "Always Tesseract": "tesseract", "Always BP-OSD": "bposd"}.items():
            used = d if d in ler else "bposd"
            rows.append({"policy": name, "good": used in good, "time": ms[used] / cheapest})
    return pd.DataFrame(rows)


def fig_choice(df, pred, theme, out):
    t = style(theme)
    o = outcomes(df, pred)
    order = ["Matchmaker, accurate", "Matchmaker, cheapest", "Always Tesseract", "Always BP-OSD"]
    good = o.groupby("policy").good.mean().reindex(order) * 100
    time = o.groupby("policy").time.median().reindex(order)
    colors = [t["accent"], t["accent"], t["neutral"], t["neutral"]]
    n = o.groupby("policy").size().iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.9), sharey=True)
    ypos = np.arange(len(order))[::-1]
    axes[0].barh(ypos, good, color=colors, height=0.56)
    for yv, v in zip(ypos, good):
        axes[0].text(v + 1.5, yv, f"{v:.0f}%", va="center", fontsize=9.5, color=t["ink"])
    axes[0].set_xlim(0, 112)
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_title("Decoder good enough", loc="left")
    axes[0].set_xlabel("% of circuits, not significantly worse than the best")
    base = np.log10(0.8)
    axes[1].barh(ypos, np.log10(time) - base, left=base, color=colors, height=0.56)
    for yv, v in zip(ypos, time):
        axes[1].text(np.log10(v) + 0.03, yv, f"{v:.1f}×", va="center", fontsize=9.5, color=t["ink"])
    axes[1].set_xlim(base, 1.35)
    axes[1].set_xticks([0, 0.5, 1])
    axes[1].set_xticklabels(["1×", "3×", "10×"])
    axes[1].set_title("Time paid for it", loc="left")
    axes[1].set_xlabel("median time, × the cheapest good-enough decoder")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(order)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(f"Choosing a decoder on {n:,} held-out circuits", x=0.012, ha="left", fontsize=12.5,
                 fontweight="semibold", color=t["ink"])
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(out, f"choice-{theme}.png"), dpi=200)
    plt.close(fig)


def fig_decoders(df, theme, out):
    """The trade-off in the data: how often each decoder is good enough, and what it costs"""
    t = style(theme)
    m = df[(df.status == "measured") & (df.ms_per_shot > 0)]
    m = m[m.groupby("circuit_key").decoder.transform("size") >= 2]
    best = m.loc[m.groupby("circuit_key").ler.idxmin()].set_index("circuit_key").ler_high
    m = m.assign(good=~(m.ler_low > m.circuit_key.map(best)))
    stats = m.groupby("decoder").agg(good=("good", "mean"), n=("good", "size"))
    stats = stats[stats.n >= 50]
    times = {d: np.log10(m[m.decoder == d].ms_per_shot.to_numpy()) for d in stats.index}
    order = sorted(stats.index, key=lambda d: np.median(times[d]))
    ypos = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True, gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].barh(ypos, stats.good.reindex(order) * 100, color=t["accent"], height=0.56)
    for yv, d in zip(ypos, order):
        axes[0].text(stats.good[d] * 100 + 1.5, yv, f"{100 * stats.good[d]:.0f}%", va="center", fontsize=9.5,
                     color=t["ink"])
    axes[0].set_xlim(0, 115)
    axes[0].set_xticks([0, 50, 100])
    axes[0].set_title("Good enough on", loc="left")
    axes[0].set_xlabel("% of the circuits it decoded")
    bp = axes[1].boxplot([times[d] for d in order], positions=ypos, vert=False, widths=0.5, whis=(5, 95),
                         showfliers=False, patch_artist=True, medianprops=dict(color=t["surface"], lw=1.6),
                         boxprops=dict(facecolor=t["accent"], edgecolor=t["accent"]),
                         whiskerprops=dict(color=t["accent"], lw=1.2), capprops=dict(color=t["accent"], lw=1.2))
    axes[1].set_title("Time per shot", loc="left")
    axes[1].set_xlabel("ms, single core (box: middle half, whiskers: 5-95%)")
    decades(axes[1], "x")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([NAMES.get(d, d) for d in order])
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("No decoder is both accurate and cheap everywhere", x=0.012, ha="left", fontsize=12.5,
                 fontweight="semibold", color=t["ink"])
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(out, f"decoders-{theme}.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "data"))
    p.add_argument("--out", default=os.path.join(HERE, "docs", "figures"))
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, _ = load_dataset(a.data)
    pred = predictions(df, a.jobs)
    for theme in THEMES:
        fig_accuracy(df, pred, theme, a.out)
        fig_choice(df, pred, theme, a.out)
        fig_decoders(df, theme, a.out)
    print("figures in", a.out)


if __name__ == "__main__":
    main()
