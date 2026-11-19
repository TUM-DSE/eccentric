"""A slice of the sweep for the live run of the pilot runtime (experiments/live/run_live.py):
python experiments/live/prepare.py <out.json> [--circuits 300] [--seed 0]

Samples circuits from those of the end-to-end replay (all five BP-type decoders and Tesseract run) and writes,
per circuit, its key and configuration, its held-out predictions (the model never saw its code) and what the
replay simulates for it, so that the live run can be compared with the simulation circuit by circuit.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import end_to_end as E
from qec_matchmaker.dataset import load_dataset
from qec_matchmaker.models import circuit_labels
from qec_matchmaker.runtime import execute

COLUMNS = ["decoder", "p_above", "ler_q50", "regret_q50", "regret_q70", "regret_q80", "regret_q90", "ler_decoder_q50",
           "ms_q10", "ms_q50", "ms_q90", "applicable"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--circuits", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objective", default="cheapest")
    a = p.parse_args()
    df, _ = load_dataset(os.path.join(E.HERE, "data"))
    pred = E.predictions(df, 4)
    labels = circuit_labels(df)
    run = df[df.status.isin(E.RUN)].copy()
    run["seconds"] = np.where(run.status == "timeout", run.timeout_s, run.ms_per_shot * run.shots / 1000)
    predicted = dict(tuple(pred.groupby("circuit_key", sort=False)))
    keys = [k for k, s in run.groupby("circuit_key", sort=False)
            if k in labels.index and k in predicted and E.COMPLETE <= set(s.decoder) and not s.seconds.isna().any()]
    rng = np.random.default_rng(a.seed)
    chosen = sorted(rng.choice(keys, size=min(a.circuits, len(keys)), replace=False))
    out, total = [], 0.0
    for key in chosen:
        s = run[run.circuit_key == key]
        first = s.iloc[0]
        sim = execute(predicted[key], E.simulated(s, np.random.default_rng(a.seed)), a.objective)
        m = s[(s.status == "measured") & (s.ler > 0)]
        best = m.loc[m.ler.idxmin()] if len(m) else None
        total += sim.seconds
        out.append({
            "key": key, "code": first.code, "backend": first.backend, "error_type": first.error_type,
            "above": bool(labels.above[key]), "best_ler": float(labels.best_ler[key]),
            "best_measured": None if best is None else {"decoder": best.decoder, "ler": float(best.ler),
                                                        "ler_low": float(best.ler_low), "ler_high": float(best.ler_high)},
            "sweep": {r.decoder: {"status": r.status, "seconds": float(r.seconds), "ler": None if np.isnan(r.ler) else float(r.ler)}
                      for r in s.itertuples()},
            "simulated": {"status": sim.status, "decoder": sim.decoder, "seconds": sim.seconds,
                          "trace": [[step, d, sec] for step, d, sec in sim.trace]},
            "predictions": json.loads(predicted[key][COLUMNS].to_json(orient="records")),
        })
    with open(a.out, "w") as f:
        json.dump({"objective": a.objective, "seed": a.seed, "circuits": out}, f)
    above = sum(c["above"] for c in out)
    print(f"{len(out)} circuits ({above} above threshold), simulated {total / 3600:.0f} CPU-h with the {a.objective} objective; "
          f"longest {max(c['simulated']['seconds'] for c in out) / 3600:.1f} h")


if __name__ == "__main__":
    main()
