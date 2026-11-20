"""Page-1 figure for the Matchmaker paper: decoder-choice policies on held-out codes.

Circuits: the ones on which BP-OSD, BP-LSD, BeliefFind, Relay-BP and Tesseract were all measured (metrics.json's
"complete" set), scored as in qec_matchmaker.evaluate: a decoder is good enough when its ler_low <= the best
measured decoder's ler_high; time is relative to the cheapest good-enough decoder's ms per shot; a choice not run
on a circuit falls back to BP-OSD. Fixed policies are computed here from data/dataset.csv; Matchmaker's points
come from reports/metrics.json (10-fold GroupKFold over codes, so every circuit's code is unseen).

    python experiments/paper/fig1_motivation.py      # writes fig1_motivation.{pdf,png} and .json next to this file
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COMPLETE = {"bposd", "bplsd", "belief_find", "relay_bp", "tesseract"}


def table():
    d = pd.read_csv(ROOT / "data/dataset.csv", low_memory=False)
    m = d[(d.status == "measured") & (d.ms_per_shot > 0)]
    m = m[m.groupby("circuit_key").decoder.transform("size") >= 2].copy()
    has = m.groupby("circuit_key").decoder.agg(set)
    m = m[m.circuit_key.isin(has[has.apply(lambda s: COMPLETE <= s)].index)]
    best_high = m.loc[m.groupby("circuit_key").ler.idxmin()].set_index("circuit_key").ler_high
    m["good"] = m.ler_low <= m.circuit_key.map(best_high)
    cheapest = m[m.good].groupby("circuit_key").ms_per_shot.min()
    m["time"] = m.ms_per_shot / m.circuit_key.map(cheapest)
    return m, has


def score(m, choice):
    rows = m.set_index(["circuit_key", "decoder"])
    out = []
    for key, dec in choice.items():
        r = rows.loc[(key, dec if (key, dec) in rows.index else "bposd")]
        out.append((r.good, r.time))
    g, t = zip(*out)
    return float(np.mean(g)), float(np.median(t))


def family_table(m):
    """For each code, the decoder that did best on the OTHER codes of its family (cheapest with >= 90% good enough,
    else the most often good enough): what an expert would pick for a new code without benchmarking it"""
    choice = {}
    for code, cs in m.groupby("code"):
        other = m[(m.code != code) & (m.family == cs.family.iloc[0])]
        if other.empty:
            other = m[m.code != code]
        s = other.groupby("decoder").agg(g=("good", "mean"), t=("time", "median"), n=("good", "size"))
        s = s[s.n >= 0.5 * other.circuit_key.nunique()]
        ok = s[s.g >= 0.9]
        dec = ok.t.idxmin() if len(ok) else s.sort_values(["g", "t"], ascending=[False, True]).index[0]
        choice.update({k: dec for k in cs.circuit_key.unique()})
    return choice


def points():
    m, has = table()
    keys = m.circuit_key.unique()
    fixed = {
        "Always BP-OSD": "bposd",
        "Always tuned BP-OSD": "bposd_tuned",
        "Always Relay-BP": "relay_bp",
        "Always Tesseract": "tesseract",
    }
    pts = {name: score(m, {k: dec for k in keys}) for name, dec in fixed.items()}
    pts["Rule of thumb"] = score(m, {k: "mwpm_corr" if "mwpm_corr" in has[k] else "bposd_tuned" for k in keys})
    pts["Family's best"] = score(m, family_table(m))
    rec = json.loads((ROOT / "reports/metrics.json").read_text())["recommendation"]["complete"]
    assert rec["circuits"] == len(keys), (rec["circuits"], len(keys))
    for obj in ("accurate", "cheapest"):
        p = rec["policies"][f"recommend, {obj}"]
        pts[f"Matchmaker ({obj})"] = (p["good_enough"], p["median_time_vs_cheapest_good"])
    return pts, len(keys), m.code.nunique()


def draw(pts, n, codes):
    ink, ink2, muted, grid, accent = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    ax.grid(True, color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ylo, yhi = 45, 108
    # shaded: at least 90% accurate within 2.5x of the cheapest accurate decoder
    ax.axvspan(0.85, 2.5, ymin=(90 - ylo) / (yhi - ylo), ymax=1, color=accent, alpha=0.07, lw=0)
    # label offsets (points) chosen to avoid collisions
    place = {"Always BP-OSD": (5, 0, "left"), "Always tuned BP-OSD": (5, 5, "left"),
             "Always Relay-BP": (5, -4, "left"), "Always Tesseract": (0, -8, "center"),
             "Rule of thumb": (5, -6, "left"), "Family's best": (5, -1, "left"),
             "Matchmaker (accurate)": (4, 8, "left"), "Matchmaker (cheapest)": (5, -7, "left")}
    for name, (g, t) in pts.items():
        mm = name.startswith("Matchmaker")
        ax.scatter(t, 100 * g, s=34 if mm else 22, color=accent if mm else muted, edgecolor="white", lw=1, zorder=3)
        dx, dy, ha = place[name]
        ax.annotate(name, (t, 100 * g), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=7, color=ink if mm else ink2, fontweight="semibold" if mm else "normal")
    ax.scatter(1, 100, marker="*", s=60, color=ink, zorder=3)
    ax.annotate("ideal", (1, 100), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=7, color=ink)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 120)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(range(50, 101, 10))
    ax.set_xticks([1, 3, 10, 30, 100], ["1×", "3×", "10×", "30×", "100×"])
    ax.set_xlabel("Median decode time vs. cheapest accurate decoder")
    ax.set_ylabel("Circuits decoded accurately (%)")
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_motivation.{ext}", dpi=300)
    (OUT / "fig1_motivation.json").write_text(json.dumps(
        {"circuits": n, "codes": codes, "points": {k: {"good_enough": g, "median_time_vs_cheapest_good": t}
                                                   for k, (g, t) in pts.items()}}, indent=1))


if __name__ == "__main__":
    pts, n, codes = points()
    for k, (g, t) in pts.items():
        print(f"{k:24s} {100 * g:5.1f}%  {t:6.2f}x")
    draw(pts, n, codes)
