"""The numbers behind the noise-scaling sweep (slice `noise_scaling`: each device's own noise model with
every error probability multiplied by 0.1, 0.25, 0.5 and 1). No plots.
python analysis/noise_scaling.py [<results.csv>] [--with-relay]

Reads experiment_results/Sweep/results.csv. A configuration's rate is its best decoder's: the lowest
logical error rate among decodes with at least MIN_ERRORS logical errors, per round (rounds = the
circuit's syndrome rounds), with its 95% Wilson interval; a configuration whose decodes all saw no
logical error gets its lowest 95% upper bound, marked "<". Relay-BP decodes made before 2026-11-05
17:30 UTC used padded syndromes (README, Decoders) and are left out unless --with-relay (or until
the rerun is merged).

Tables:
1. coverage: configurations with a rate, per device and noise scale;
2. per device and matchable code (rounds = d), the per-round LER at each distance and scale, and the
   suppression factor Lambda = LER(d) / LER(d + 2) between successive distances, marked significant
   when the Wilson intervals do not overlap;
3. the threshold scale per device and code: where Lambda(3 -> 5) crosses 1, interpolated in log scale
   between the measured scales (">1" if already below threshold at the device's own noise, "<0.1" if
   still above at a tenth of it);
4. per device and qLDPC code (rounds = d), the slope s of log LER against log scale between successive
   scales; below threshold LER ~ p^((d_eff + 1) / 2), so d_eff = 2 s - 1 is an effective distance,
   given from the lowest pair of scales with both per-round rates below DEFF_BELOW;
5. which decoder is best how often, per scale.
"""
import math
import os
import re
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 40)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_ERRORS = 10
# Effective distances only from scales whose per-round rates are both below this
DEFF_BELOW = 0.01
SCALES = (0.1, 0.25, 0.5, 1.0)
MATCHABLE = ("surface", "color", "hh", "planar_honeycomb")
SCALED = re.compile(r"(.+)_x(\d+(?:\.\d+)?)$")


def per_round(p, rounds):
    """Per-round error rate; 0.5 (saturated) for a rate of 0.5 or more"""
    if p is None or not np.isfinite(p):
        return np.nan
    if p >= 0.5:
        return 0.5
    return (1 - (1 - 2 * p) ** (1 / rounds)) / 2


def load(path, with_relay=False):
    df = pd.read_csv(path, low_memory=False)
    df = df[df.slice == "noise_scaling"].copy()
    parsed = df.error_type.str.extract(SCALED)
    df["noise"] = parsed[0].fillna(df.error_type)
    df["scale"] = parsed[1].astype(float).fillna(1.0)
    df["device"] = df.backend + " / " + df.noise
    if not with_relay:
        df = df[df.decoder != "relay_bp"]
    return df


def best(df):
    """One row per configuration: its best decoder's per-round LER (or upper bound)"""
    keys = ["device", "code", "distance", "cycles", "scale"]
    rated = df[df.status.isin(["measured", "low statistics", "upper bound"])].copy()
    rounds = rated.rounds.fillna(rated.cycles).astype(float)
    rated["pr"] = [per_round(p, r) for p, r in zip(rated.logical_error_rate, rounds)]
    rated["pr_low"] = [per_round(p, r) for p, r in zip(rated.ler_low, rounds)]
    rated["pr_high"] = [per_round(p, r) for p, r in zip(rated.ler_high, rounds)]
    measured = rated[(rated.errors >= MIN_ERRORS) & rated.pr.notna()]
    out = measured.loc[measured.groupby(keys).pr.idxmin()][keys + ["decoder", "pr", "pr_low", "pr_high", "errors"]]
    out["bound"] = False
    bounds = rated[rated.errors == 0]
    bounds = bounds[~bounds.set_index(keys).index.isin(out.set_index(keys).index)]
    if len(bounds):
        b = bounds.loc[bounds.groupby(keys).pr_high.idxmin()][keys + ["decoder", "pr_high"]]
        out = pd.concat([out, b.assign(pr=b.pr_high, pr_low=0.0, bound=True, errors=0)], ignore_index=True)
    return out


def fmt(row):
    if row is None or (isinstance(row, float) and math.isnan(row)):
        return ""
    return ("<" if row["bound"] else "") + f"{row['pr']:.2g}"


def coverage(df):
    rated = df[df.status.isin(["measured", "low statistics", "upper bound"])]
    total = df[df.status != "does not fit"].groupby(["device", "scale"]).apply(
        lambda g: g[["code", "distance", "cycles"]].drop_duplicates().shape[0], include_groups=False)
    done = rated.groupby(["device", "scale"]).apply(
        lambda g: g[["code", "distance", "cycles"]].drop_duplicates().shape[0], include_groups=False)
    t = pd.DataFrame({"done": done, "configurations": total}).fillna(0).astype(int)
    return t.unstack("scale")


def suppression(b):
    """Per device, matchable code and scale: per-round LER by distance and Lambda between successive ones"""
    rows = []
    m = b[b.code.isin(MATCHABLE) & (b.cycles == b.distance)]
    for (device, code, scale), g in m.groupby(["device", "code", "scale"]):
        g = g.set_index("distance").sort_index()
        row = {"device": device, "code": code, "scale": scale}
        for d, r in g.iterrows():
            row[f"d{d}"] = fmt(r)
        ds = list(g.index)
        for d1, d2 in zip(ds, ds[1:]):
            a, c = g.loc[d1], g.loc[d2]
            if a["bound"]:
                continue
            lam = a["pr"] / c["pr"]
            significant = (a["pr_low"] > c["pr_high"]) or (a["pr_high"] < c["pr_low"] if not c["bound"] else False)
            row[f"L{d1}-{d2}"] = (">" if c["bound"] else "") + f"{lam:.2f}" + ("*" if significant else "")
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_scales(b):
    """Per device and matchable code, the scale where Lambda(3 -> 5) crosses 1"""
    rows = []
    m = b[b.code.isin(MATCHABLE) & (b.cycles == b.distance) & b.distance.isin([3, 5]) & ~b.bound]
    for (device, code), g in m.groupby(["device", "code"]):
        p = g.pivot_table(index="scale", columns="distance", values="pr")
        if 3 not in p or 5 not in p:
            continue
        lam = np.log(p[3] / p[5]).dropna()
        if lam.empty:
            continue
        scales = list(lam.index)
        if (lam > 0).all():
            est = f">{max(scales):g}" if max(scales) == 1.0 else f">={max(scales):g}"
        elif (lam < 0).all():
            est = f"<{min(scales):g}"
        else:
            est = ""
            for s1, s2 in zip(scales, scales[1:]):
                if lam[s1] > 0 >= lam[s2]:
                    x = math.log(s1) + (math.log(s2) - math.log(s1)) * lam[s1] / (lam[s1] - lam[s2])
                    est = f"{math.exp(x):.2g}"
                    break
            est = est or "not monotone"
        rows.append({"device": device, "code": code, "threshold_scale": est,
                     **{f"Lambda at x{s:g}": round(float(np.exp(lam[s])), 2) for s in scales}})
    return pd.DataFrame(rows)


def effective_distance(b):
    """Per device and code that is not matchable: the slope of log LER against log scale between successive
    scales whose rates are not saturated, and d_eff = 2 s - 1 from the lowest pair with both per-round
    rates below DEFF_BELOW (closer to threshold the slope is steeper than the distance alone gives)"""
    rows = []
    q = b[~b.code.isin(MATCHABLE) & ~b.bound & (b.pr < 0.5)]
    for (device, code, d, cycles), g in q.groupby(["device", "code", "distance", "cycles"]):
        g = g.set_index("scale").sort_index()
        row = {"device": device, "code": code, "d": d}
        deff = None
        for s1, s2 in zip(g.index, g.index[1:]):
            slope = math.log(g.loc[s2, "pr"] / g.loc[s1, "pr"]) / math.log(s2 / s1)
            row[f"slope {s1:g}-{s2:g}"] = round(slope, 2)
            if deff is None and max(g.loc[s1, "pr"], g.loc[s2, "pr"]) < DEFF_BELOW:
                deff = round(2 * slope - 1, 1)
        row["d_eff"] = deff
        rows.append(row)
    t = pd.DataFrame(rows)
    slopes = sorted([c for c in t.columns if c.startswith("slope")], key=lambda c: float(c.split()[1].split("-")[0]))
    return t[["device", "code", "d", *slopes, "d_eff"]] if len(t) else t


def decoder_wins(b):
    return b[~b.bound].groupby(["scale", "decoder"]).size().unstack(fill_value=0)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else os.path.join(ROOT, "experiment_results", "Sweep", "results.csv")
    df = load(path, with_relay="--with-relay" in sys.argv)
    b = best(df)
    print(f"# Noise scaling: {b[['device', 'code', 'distance', 'cycles', 'scale']].drop_duplicates().shape[0]} "
          f"configurations with a rate, {df[df.status == 'pending'][['device', 'code', 'distance', 'cycles', 'scale']].drop_duplicates().shape[0]} pending\n")
    print("## 1. Coverage (configurations with a rate / all that fit)\n")
    print(coverage(df).to_string(), "\n")
    print("## 2. Per-round LER by distance, and Lambda = LER(d) / LER(d + 2) (* significant)\n")
    s = suppression(b)
    print(s.to_string(index=False) if len(s) else "(none yet)", "\n")
    print("## 3. Noise scale at which Lambda(3 -> 5) = 1\n")
    t = threshold_scales(b)
    print(t.to_string(index=False) if len(t) else "(none yet)", "\n")
    print("## 4. qLDPC and other codes: slope of log LER against log scale, effective distance 2 s - 1\n")
    e = effective_distance(b)
    print(e.to_string(index=False) if len(e) else "(none yet)", "\n")
    print("## 5. Best decoder per scale (configurations)\n")
    print(decoder_wins(b).to_string())


if __name__ == "__main__":
    main()
