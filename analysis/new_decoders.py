"""Correlated matching (mwpm_corr) and level-by-level BP-OSD (hier_bposd, hier_bposd_hard) against the
decoders already in the sweep. No plots.
python analysis/new_decoders.py [<results.csv>]

A configuration is one circuit (sweep.collect CONFIG without the decoder). Two decoders are compared on the
configurations where both have a logical error rate above 0 and the reference decoder is below threshold
(rate < 0.5); ratio = new / reference, so below 1 is better. "Significant" means the 95% Wilson intervals
of the two rates do not overlap.
"""
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sweep.collect import CONFIG  # noqa: E402

BP_TYPE = ["bposd", "bposd_tuned", "bplsd", "belief_find", "relay_bp", "gari", "gari_pp"]


def wilson(errors, shots, z=1.96):
    p = errors / shots
    centre = (p + z * z / (2 * shots)) / (1 + z * z / shots)
    half = z * np.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / (1 + z * z / shots)
    return centre - half, centre + half


def load(path):
    t = pd.read_csv(path, low_memory=False)
    t["operation"] = t["operation"].fillna("memory")
    columns = [c for c in CONFIG if c in t.columns and c != "decoder"]
    t["config"] = t[columns].astype(str).agg("|".join, axis=1)
    t = t.drop_duplicates(["config", "decoder"])
    t["low"], t["high"] = wilson(t.errors.fillna(0), t.shots.where(t.shots > 0))
    return t


def pair(t, new, reference):
    a = t[t.decoder == new].set_index("config")
    b = t[t.decoder == reference].set_index("config")
    j = a.join(b[["logical_error_rate", "low", "high", "status"]], rsuffix="_ref", how="inner")
    j = j[(j.logical_error_rate > 0) & (j.logical_error_rate_ref > 0) & (j.logical_error_rate_ref < 0.5)].copy()
    j["ratio"] = j.logical_error_rate / j.logical_error_rate_ref
    j["better"] = j.high < j.low_ref
    j["worse"] = j.low > j.high_ref
    return j


def summary(g):
    return pd.Series({"configurations": len(g), "median": g.ratio.median(), "q10": g.ratio.quantile(0.1),
                      "q90": g.ratio.quantile(0.9), "better (sig.)": int(g.better.sum()), "worse (sig.)": int(g.worse.sum())})


def correlated_matching(t):
    print("# Correlated matching (mwpm_corr) against plain matching (mwpm)\n")
    c = t[t.decoder == "mwpm_corr"]
    print("rows by status:", c.status.value_counts().to_dict())
    j = pair(t, "mwpm_corr", "mwpm")
    print("\nall:", summary(j).round(3).to_dict())
    print("\nby operation:")
    print(j.groupby("operation").apply(summary).round(3).sort_values("median").to_string())
    print("\nby code (5 or more configurations):")
    by_code = j.groupby("code").apply(summary)
    print(by_code[by_code.configurations >= 5].round(3).sort_values("median").to_string())
    m = t[t.decoder == "mwpm"].set_index("config")
    rated = c[(c.logical_error_rate.notna()) & c.status.isin(["measured", "low statistics", "upper bound"])].set_index("config")
    newly = rated[~rated.index.isin(m[m.logical_error_rate.notna()].index)]
    print("\ndecoded by mwpm_corr, not by mwpm:", len(newly), newly.groupby(["code", "operation"]).size().to_dict())
    best = t[t.logical_error_rate > 0].groupby("config").logical_error_rate.min()
    measured = c[c.logical_error_rate > 0].set_index("config")
    print("configurations where mwpm_corr has the lowest rate of all decoders:",
          int((measured.logical_error_rate <= best.reindex(measured.index)).sum()))


def level_by_level(t):
    print("\n# Level-by-level BP-OSD (hier_bposd: soft information; hier_bposd_hard: hard decisions)\n")
    h = t[t.decoder.str.startswith("hier")]
    print(h.groupby(["decoder", "code", "distance"]).status.value_counts().unstack(fill_value=0).to_string())
    rows = []
    for reference in ["hier_bposd_hard", "bposd", "bposd_tuned", "relay_bp", "gari_pp", "tesseract"]:
        j = pair(t, "hier_bposd", reference)
        for (code, distance), g in j.groupby(["code", "distance"]):
            rows.append({"code": f"{code} d={distance}", "against": reference, **summary(g).to_dict()})
    print()
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # Against the best BP-type decoder of each configuration
    soft = h[(h.decoder == "hier_bposd") & (h.logical_error_rate > 0) & (h.logical_error_rate < 0.5)].set_index("config")
    bp = t[t.decoder.isin(BP_TYPE) & (t.logical_error_rate > 0)]
    best_bp = bp.loc[bp.groupby("config").logical_error_rate.idxmin()].set_index("config")
    best_all = t[(t.logical_error_rate > 0) & (t.decoder != "hier_bposd")].groupby("config").logical_error_rate.min()
    soft = soft.join(best_bp[["logical_error_rate", "low", "decoder"]], rsuffix="_bp")
    soft["best_all"] = best_all.reindex(soft.index)
    print("\nagainst the best BP-type decoder of each configuration (rate below 0.5):")
    for (code, distance), g in soft.groupby(["code", "distance"]):
        wins = g[g.logical_error_rate < g.logical_error_rate_bp]
        print(f"  {code} d={distance}: {len(g)} configurations, lower than every BP-type decoder on {len(wins)} "
              f"({int((g['high'] < g['low_bp']).sum())} significantly), "
              f"lowest of all decoders on {int((g.logical_error_rate < g.best_all).sum())}")
        for r in wins.sort_values("logical_error_rate").itertuples():
            print(f"    {r.backend:22s} {r.error_type:18s} {r.logical_error_rate:.4g} vs {r.decoder_bp} "
                  f"{r.logical_error_rate_bp:.4g} (x{r.logical_error_rate / r.logical_error_rate_bp:.2f}); best of all {r.best_all:.4g}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiment_results", "Sweep", "results.csv")
    t = load(path)
    correlated_matching(t)
    level_by_level(t)


if __name__ == "__main__":
    main()
