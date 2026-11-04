"""Can a configuration's LER be predicted without decoding it? python analysis/predict_ler.py [<dem_features.json>]

Per circuit, the best measured LER of the sweep table is the target. Predictors, all tested
leave-one-code-out (trained on the other codes): a linear model of log10 LER, and a classification tree
for "saturated" (LER > 0.5), on features of the table (size, distance, rounds, routing, a nominal
two-qubit error of the noise model) and on properties of the circuit's detector error model
(analysis/dem_features.py, saved in experiment_results/Sweep/dem_features.json), which take seconds to
compute and need no decoding. Last, the decoder choice of analysis/decoding.py with the mean detector
firing probability in place of the pilot run's LER.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoding  # noqa: E402
from decoding import grow, load, predict  # noqa: E402

# Nominal two-qubit gate error of each device noise model, for the table-only features
P2 = {"real_flamingo": 2.6e-3, "real_flamingo_downscaled": 2.6e-4, "real_flamingo_m700": 2.6e-3,
      "real_flamingo_downscaled_m700": 2.6e-4, "real_heron": 3e-3, "real_apollo": 1e-3, "real_infleqtion": 5e-3,
      "real_harvard": 5e-3, "real_willow": 3.3e-3, "real_willow_spec": 3.3e-3}
TABLE = ["p2", "distance", "observables", "rounds", "detectors", "two_qubit_gates", "swaps", "circuit_qubits"]
DEM = ["det_mean", "det_median", "det_max", "error_mass", "mechanisms", "mechanism_weight"]


def logs(frame, cols):
    return np.column_stack([np.log10(np.maximum(frame[c].astype(float).values, 1e-9)) for c in cols])


def leave_one_code_out(c, X):
    y, sat, codes = c.y.values, c.sat.values, c.code.values
    pred, cls = np.zeros(len(c)), np.zeros(len(c), dtype=int)
    for code in np.unique(codes):
        tr = codes != code
        w, *_ = np.linalg.lstsq(np.column_stack([X[tr], np.ones(tr.sum())]), y[tr], rcond=None)
        pred[~tr] = np.column_stack([X[~tr], np.ones((~tr).sum())]) @ w
        tree = grow(X[tr], sat[tr], depth=4, min_leaf=15)
        cls[~tr] = [predict(tree, x) for x in X[~tr]]
    err = np.abs(pred - y)
    tp = int(((cls == 1) & (sat == 1)).sum())
    return {"median |log10 error|": np.median(err), "p90 |log10 error|": np.quantile(err, 0.9),
            "rank correlation": pd.Series(pred).corr(pd.Series(y), method="spearman"),
            "saturated: accuracy": np.mean(cls == sat), "recall": tp / sat.sum(), "precision": tp / max(cls.sum(), 1)}


def main():
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_results", "Sweep", "dem_features.json")
    dem = json.load(open(sys.argv[1] if len(sys.argv) > 1 else default))
    df = load()
    rated = df[(df.status == "measured") & df.two_qubit_gates.notna()].copy()
    c = rated.loc[rated.groupby("cid").logical_error_rate.idxmin()].copy()
    c["key"] = c.run + "/" + c.job_id.str.replace(r"^decode\.", "circuit.", regex=True).str.replace(r"\.decoder=.*$", "", regex=True)
    c = c.join(pd.DataFrame(dem).T, on="key")
    c["p2"] = c.error_type.map(P2).fillna(pd.to_numeric(c.error_probability, errors="coerce"))
    c["swaps"] = c.swaps + 1
    c["y"] = np.log10(c.logical_error_rate.clip(1e-7, 0.999))
    c["sat"] = (c.logical_error_rate > 0.5).astype(int)
    c = c.dropna(subset=TABLE + DEM)
    print(f"circuits: {len(c)}, saturated: {c.sat.mean():.0%} (always 'not saturated' is right for {1 - c.sat.mean():.0%})\n")

    results = {"table features": leave_one_code_out(c, logs(c, TABLE)),
               "error-model features": leave_one_code_out(c, logs(c, DEM)),
               "both": leave_one_code_out(c, logs(c, TABLE + DEM))}
    # The simplest physical criterion: saturated above a mean detector firing probability, threshold
    # chosen on the other codes
    codes, hit = c.code.values, np.zeros(len(c), dtype=int)
    for code in np.unique(codes):
        tr = codes != code
        grid = np.quantile(c.det_mean[tr], np.linspace(0.01, 0.99, 99))
        t = max(grid, key=lambda g: np.mean((c.det_mean[tr] > g) == c.sat[tr]))
        hit[~tr] = c.det_mean[~tr] > t
    tp = int(((hit == 1) & (c.sat.values == 1)).sum())
    results["threshold on mean detector firing"] = {"saturated: accuracy": np.mean(hit == c.sat.values),
                                                     "recall": tp / c.sat.sum(), "precision": tp / max(hit.sum(), 1)}
    print(pd.DataFrame(results).T.round(2).to_string())

    print("\nLER against mean detector firing probability (circuits per bin, median LER, share saturated):")
    c["bin"] = pd.cut(c.det_mean, [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    print(c.groupby("bin", observed=True).agg(n=("y", "size"), median_ler=("logical_error_rate", "median"),
                                              saturated=("sat", "mean")).round(4).to_string())
    print("\nfor a full threshold sweep: mean detector firing at the threshold of the saturation tree, all circuits:")
    tree = grow(logs(c, ["det_mean"]), c.sat.values, depth=1, min_leaf=15)
    print(f"  saturated above det_mean = {10 ** tree[1]:.3f}" if isinstance(tree, tuple) else tree)

    print("\nDecoder choice (analysis/decoding.py) with the pilot LER against the mean detector firing:")
    m = df[(df.status == "measured") & df.ms.notna()].copy()
    circ = decoding.circuits(m)
    key = m.groupby("cid").apply(lambda s: s.run.iloc[0] + "/" + s.job_id.iloc[0].replace("decode.", "circuit.", 1).split(".decoder=")[0])
    circ["det_mean"] = circ.cid.map(key).map(pd.DataFrame(dem).T.det_mean).astype(float)
    circ = circ.dropna(subset=["det_mean"])
    rows = {}
    for label, regime in (("pilot LER", "pilot"), ("mean detector firing", "det_mean")):
        X = np.array([[r.group == "matchable", r.group == "color", r.detectors, r.k, r.rounds, r.routed, r.device_noise, r[regime]]
                      for _, r in circ.iterrows()], dtype=float)
        for target in ("cheapest_good", "best"):
            preds = {}
            for code in circ.code.unique():
                tr = (circ.code != code).values
                tree = grow(X[tr], circ[target].values[tr], depth=4, min_leaf=15)
                for i in np.where(~tr)[0]:
                    preds[circ.cid.iloc[i]] = predict(tree, X[i])
            rows[f"{label}, tree for the {target.replace('_', ' ')}"] = decoding.evaluate(circ, lambda r, p=preds: p[r.cid])
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
