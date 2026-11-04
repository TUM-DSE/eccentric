"""Tables for experiments/RESULTS.md from the cached per-unit results of model_quality.py.

    python experiments/report.py hp | mono | drop | curve | importance | final
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_quality as mqx  # noqa: E402
import mq_core as mq  # noqa: E402

KEY = ["regime_acc", "ler_med", "ler_p90", "ler_spearman", "time_med", "time_p90", "time_3x", "time_3x_relay_bp",
       "regret_med", "regret_p90", "rec_acc_good", "rec_cheap_good", "ler_cov", "time_cov"]


def curve_table(metrics=KEY, prefix="frac", fractions=(0.25, 0.5, 0.75)):
    """Mean over 3 subsamples (min-max in brackets) per share of training codes (prefix frac) or of each code's
    circuits (prefix cfrac); 100% is the baseline"""
    rows = {}
    for f in fractions:
        ss = [mq.summarize(pd.read_pickle(os.path.join(mqx.CACHE, f"{prefix}{f}_s{s}.pkl"))) for s in (0, 1, 2)]
        rows[f] = {m: (np.mean([s[m] for s in ss]), min(s[m] for s in ss), max(s[m] for s in ss)) for m in metrics}
    base = mq.summarize(pd.read_pickle(os.path.join(mqx.CACHE, "baseline.pkl")))
    rows[1.0] = {m: (base[m], base[m], base[m]) for m in metrics}
    what = "training codes per fold" if prefix == "frac" else "circuits per training code (all codes)"
    lines = [f"| {what} | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 1)]
    for f, r in rows.items():
        label = f"{round(100 * f)}% (~{round(f * 48.6)} codes)" if prefix == "frac" else f"{round(100 * f)}%"
        cells = [f"{r[m][0]:.3f}" + (f" ({r[m][1]:.3f}-{r[m][2]:.3f})" if f < 1 else "") for m in metrics]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def width_at_coverage(frame, coverage=0.8):
    """Oracle conformal widening: the symmetric widening (decades) that makes [q10, q90] hold `coverage` of
    the held-out truths, and the median interval width after it; comparable across configs only"""
    lo, hi, y = frame.lo.to_numpy(), frame.hi.to_numpy(), frame.y.to_numpy()
    widen = max(float(np.quantile(np.maximum(lo - y, y - hi), coverage)), 0.0)
    return widen, float(np.median(hi - lo + 2 * widen)), float(np.median(hi - lo))


def interval_table(names):
    lines = ["| config | LER raw coverage | LER raw median width | LER width at 80% | time raw coverage | "
             "time raw median width | time width at 80% |", "|---|---|---|---|---|---|---|"]
    for name in names:
        u = pd.read_pickle(os.path.join(mqx.CACHE, f"{name}.pkl"))
        cells = []
        for k in ("ler", "time"):
            if "lo" not in u[k]:
                cells += ["n/a"] * 3
                continue
            _, w80, raw = width_at_coverage(u[k])
            cells += [f"{u[k].inside.mean():.0%}", f"{raw:.2f}", f"{w80:.2f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    what = sys.argv[1]
    if what in ("hp", "hp_b", "mono", "drop", "final"):
        names = None
        if what == "hp":
            names = [c.name for c in mqx.configs("hp")] + [c.name for c in mqx.configs("hp_b")][1:]
        print(mqx.table("hp" if what == "hp" else what, names=names, metrics=KEY))
    elif what == "curve":
        print(curve_table())
    elif what == "curve_circuits":
        print(curve_table(prefix="cfrac", fractions=(0.25, 0.5)))
    elif what == "finals":
        names = ["base_iv", "cap200_ler_time", "cap_ler_time", "cap63_ler_time", "cap_drop_em", "cap_drop_em_det",
                 "drop_em_iv"]
        print(mqx.table("final", names=names, base="base_iv", metrics=KEY))
        print()
        print(interval_table(names))
    elif what == "importance":
        print(mqx.importance_table())
        print()
        print(mqx.probe_table())
        for name in ("mono_all",):
            if os.path.exists(os.path.join(mqx.RESULTS, f"probe_{name}.csv")):
                print()
                print(mqx.probe_table(name))
