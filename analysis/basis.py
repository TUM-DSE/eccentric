"""X-basis against Z-basis memory (slice basis_x). No plots.
python analysis/basis.py [<results.csv>]

Each X-basis configuration of basis_x is paired with the Z-basis results of the same configuration
(code, schedule, distance, rounds, device and its size, noise, compiler settings) in any slice. A configuration's rate in a basis is its
best decoder's per-round logical error rate among decodes with at least MIN_ERRORS errors, with its 95%
Wilson interval, the best taken over the decoders rated in both bases; ratio = X / Z, significant when
the intervals do not overlap.
"""
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_ERRORS = 10
PAIR = ["code", "schedule", "distance", "cycles", "backend", "backend_size", "error_type", "error_probability",
        "layout_method", "routing_method", "translating_method"]


def per_round(p, rounds):
    if p is None or not np.isfinite(p):
        return np.nan
    return 0.5 if p >= 0.5 else (1 - (1 - 2 * p) ** (1 / rounds)) / 2


def rated(df):
    r = df[df.status.isin(["measured", "low statistics"]) & (df.errors >= MIN_ERRORS)].copy()
    rounds = r.rounds.fillna(r.cycles).astype(float)
    for col, src in (("pr", "logical_error_rate"), ("lo", "ler_low"), ("hi", "ler_high")):
        r[col] = [per_round(p, n) for p, n in zip(r[src], rounds)]
    for c in PAIR:
        r[c] = r[c].map(str)
    return r[PAIR + ["decoder", "pr", "lo", "hi"]]


def best_common(x, z):
    """Per configuration, each basis's best decoder among the decoders rated in both bases (the Z side often
    has more decoders, from the decoder fill-in)"""
    both = x.merge(z[PAIR + ["decoder"]], on=PAIR + ["decoder"])
    zc = z.merge(x[PAIR + ["decoder"]], on=PAIR + ["decoder"])
    bx = both.loc[both.groupby(PAIR).pr.idxmin()]
    bz = zc.loc[zc.groupby(PAIR).pr.idxmin()]
    return bx.merge(bz, on=PAIR, suffixes=("_x", "_z"))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiment_results", "Sweep", "results.csv")
    df = pd.read_csv(path, low_memory=False)
    df["basis"] = df.basis.fillna("Z")
    x = rated(df[(df.slice == "basis_x") & (df.basis == "X")])
    z = rated(df[df.basis == "Z"])
    p = best_common(x, z)
    p["ratio"] = p.pr_x / p.pr_z
    p["significant"] = (p.lo_x > p.hi_z) | (p.hi_x < p.lo_z)
    p["device"] = p.backend + " / " + p.error_type
    print(f"# X against Z basis: {len(p)} paired configurations (best of the decoders rated in both bases)\n")
    print("## Per code: median X/Z ratio of the per-round rate, and configurations significantly worse / better in X\n")
    g = p.groupby("code").agg(configs=("ratio", "size"), median_ratio=("ratio", "median"),
                              x_worse=("ratio", lambda r: int(((r > 1) & p.loc[r.index, "significant"]).sum())),
                              x_better=("ratio", lambda r: int(((r < 1) & p.loc[r.index, "significant"]).sum())))
    print(g.round(2).to_string(), "\n")
    print("## Per device: median X/Z ratio\n")
    print(p.groupby("device").ratio.agg(["size", "median", "min", "max"]).round(2).to_string(), "\n")
    print("## Every pair\n")
    cols = ["code", "distance", "cycles", "device", "error_probability", "decoder_z", "pr_z", "decoder_x", "pr_x", "ratio", "significant"]
    print(p.sort_values(["code", "device", "distance", "cycles"])[cols].to_string(index=False,
          formatters={"pr_z": "{:.2e}".format, "pr_x": "{:.2e}".format, "ratio": "{:.2f}".format}))


if __name__ == "__main__":
    main()
