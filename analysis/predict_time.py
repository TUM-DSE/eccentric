"""Can a decoder's time per shot be predicted before decoding? python analysis/predict_time.py [<dem_features.json>]

Per decoder, a linear model of log10 time per shot (one core, decoder construction included) on log
features of the circuit's detector error model (analysis/dem_features.py: mechanisms, their mean
detector count, the mean detector firing probability, the error mass) and of its size (detectors,
rounds), tested leave-one-code-out on the measured decodes of the sweep table.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoding import load  # noqa: E402

FEATURES = ["detectors", "rounds", "mechanisms", "mechanism_weight", "det_mean", "error_mass"]


def main():
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_results", "Sweep", "dem_features.json")
    dem = pd.DataFrame(json.load(open(sys.argv[1] if len(sys.argv) > 1 else default))).T
    df = load()
    m = df[(df.status == "measured") & (df.ms > 0)].copy()
    m["key"] = m.run + "/" + m.job_id.str.replace(r"^decode\.", "circuit.", regex=True).str.replace(r"\.decoder=.*$", "", regex=True)
    m = m.join(dem[["mechanisms", "mechanism_weight", "det_mean", "error_mass"]].astype(float), on="key").dropna(subset=FEATURES)
    rows = {}
    for decoder, s in m.groupby("decoder"):
        if s.code.nunique() < 5:
            continue
        X = np.column_stack([np.log10(s[f].astype(float).clip(lower=1e-9)) for f in FEATURES] + [np.ones(len(s))])
        y = np.log10(s.ms.values)
        pred = np.zeros(len(s))
        for code in s.code.unique():
            tr = (s.code != code).values
            w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            pred[~tr] = X[~tr] @ w
        err = np.abs(pred - y)
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows[decoder] = {"decodes": len(s), "codes": s.code.nunique(), "median |log10 error|": np.median(err),
                         "p90 |log10 error|": np.quantile(err, 0.9), "within 3x": np.mean(err < np.log10(3)),
                         "rank correlation": pd.Series(pred).corr(pd.Series(y), method="spearman"),
                         **{f"exp {f}": w[i] for i, f in enumerate(FEATURES)}}
    print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
