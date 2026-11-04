"""Monotonicity probe along realistic directions, on held-out codes: scale all noise-rate features together
(as a higher physical error rate would) and all size features together (as a bigger circuit would), and count
the predictions that move the wrong way. Compares the baseline with the monotone-constrained models.

    python experiments/probe_joint.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_quality as mqx  # noqa: E402
import mq_core as mq  # noqa: E402

NOISE = ["det_fire_mean", "det_fire_median", "det_fire_max", "error_mass_per_detector", "error_mass"]
SIZE = ["detectors", "mechanisms", "qubits", "measurements", "two_qubit_gates", "error_mass", "ticks"]
FACTORS = (1.1, 1.5)


def _hook(models, meta, test, decoders, fold):
    base = mq.predict(models, meta, test, decoders)
    rows = []
    for direction, cols, col in (("noise up", NOISE, "ler_q50"), ("noise up", NOISE, "p_above"),
                                 ("size up", SIZE, "ms_q50")):
        for f in FACTORS:
            t = test.copy()
            for c in cols:
                t[c] = t[c] * f
            p = mq.predict(models, meta, t, decoders)
            a, b = base[col].to_numpy(float), p[col].to_numpy(float)
            if col != "ms_q50":
                a, b = a[::len(decoders)], b[::len(decoders)]
            delta = (np.log10(b) - np.log10(a)) if col != "p_above" else (b - a)
            delta = delta[~np.isnan(delta)]
            rows.append({"fold": fold, "direction": direction, "output": col, "factor": f, "n": len(delta),
                         "decreases": int((delta < -1e-9).sum()), "mean": float(delta.mean()), "min": float(delta.min())})
    return rows


def main():
    df, _ = mq.load()
    out = []
    for cfg in [c for c in mqx.configs("mono") if c.name in ("baseline", "mono_all", "mono_full")]:
        _, hooks = mq.cross_val(df, cfg, n_jobs=mqx.N_JOBS, permute=_hook)
        out += [{**r, "config": cfg.name} for h in hooks for r in h]
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(mqx.RESULTS, "probe_joint.csv"), index=False)
    g = res.groupby(["config", "direction", "output", "factor"], sort=False).agg(
        n=("n", "sum"), decreases=("decreases", "sum"), mean=("mean", "mean"), worst=("min", "min"))
    print("| config | direction | output | factor | share moving the wrong way | mean change | worst |")
    print("|---|---|---|---|---|---|---|")
    for (c, d, o, f), r in g.iterrows():
        unit = "" if o == "p_above" else " dec"
        print(f"| {c} | {d} | {o} | x{f} | {r.decreases / r.n:.1%} | {r['mean']:+.3f}{unit} | {r.worst:+.3f}{unit} |")


if __name__ == "__main__":
    main()
