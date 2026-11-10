"""End to end: what a benchmark campaign costs when Matchmaker makes its decisions.

python experiments/end_to_end.py [--data data] [--jobs 4] [--skip-above 0.5]

Replays ECCentric's sweep on its own measurements. Every circuit is predicted by a model that never saw its
code (10 folds over codes, as in reports/REPORT.md). A policy decides, per circuit, whether to decode it at
all and with which decoder; it is charged the CPU time that decode took in the sweep (shots x time per shot
to its error target, or its timeout), and its outcome is what the sweep measured:

- on circuits below threshold: is the result good enough (the decoder's rate not significantly worse than
  the best measured one; a timeout or a skipped circuit is not)?
- on circuits above threshold (best measured logical error rate above 0.5, models.ABOVE): CPU time spent
  to learn that nothing works.

Circuits: those on which all of bposd, bplsd, belief_find, relay_bp and tesseract were run (measured or
timed out), so that every policy's choice has a measured cost and outcome. Policies:
  everything        every decoder that was run (the sweep as it was)
  always <decoder>  one decoder everywhere
  matchmaker        skip the circuit if P(above threshold) > --skip-above, else the recommended decoder
                    ("accurate", or "cheapest" within 1.25x of the most accurate)
  oracle            skip above threshold, else the cheapest good-enough decoder (hindsight)
Writes experiments/END_TO_END.md and experiments/results/end_to_end.json.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from qec_matchmaker.dataset import load_dataset
from qec_matchmaker.evaluate import COMPLETE, cross_val_predictions
from qec_matchmaker.models import circuit_labels
from qec_matchmaker.recommend import recommend

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = ("measured", "low statistics", "upper bound", "timeout")


def predictions(df, jobs):
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]
    path = os.path.join(HERE, "experiments", "cache", f"cv_code_{digest}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    pred = cross_val_predictions(df, folds=10, calibrate=True, n_jobs=jobs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pred.to_pickle(path)
    return pred


def replay(df, pred, skip_above):
    """One row per circuit and policy: CPU seconds, whether it was skipped, and whether its result is good enough"""
    labels = circuit_labels(df)
    run = df[df.status.isin(RUN)].copy()
    run["seconds"] = np.where(run.status == "timeout", run.timeout_s, run.ms_per_shot * run.shots / 1000)
    predicted = dict(tuple(pred.groupby("circuit_key", sort=False)))
    rows = []
    for key, s in run.groupby("circuit_key", sort=False):
        if key not in labels.index or key not in predicted or not COMPLETE <= set(s.decoder) or s.seconds.isna().any():
            continue
        above = bool(labels.above[key])
        cost = dict(zip(s.decoder, s.seconds))
        m = s[(s.status == "measured") & (s.ler > 0)]
        good = set()
        if not above and len(m):
            best = m.loc[m.ler.idxmin()]
            good = set(m[~(m.ler_low > best.ler_high)].decoder)
            # A decode that saw fewer errors than its target but bounds the rate below the best is as good
            good |= set(s[(s.status != "timeout") & (s.ler_high <= best.ler_high)].decoder)
        p = predicted[key]
        choices = {"everything": None, **{f"always {d}": d for d in sorted(COMPLETE)}}
        for objective in ("accurate", "cheapest"):
            skip = float(p.p_above.iloc[0]) > skip_above
            choices[f"matchmaker, {objective}"] = "skip" if skip else recommend(p, objective).decoder
        choices["oracle"] = "skip" if above else (min(good, key=cost.get) if good else min(cost, key=cost.get))
        for policy, d in choices.items():
            if d is None:
                seconds, ok, skipped = sum(cost.values()), bool(good), False
            elif d == "skip":
                seconds, ok, skipped = 0.0, False, True
            else:
                used = d if d in cost else "bposd"  # a choice the sweep did not run on this circuit
                seconds, ok, skipped = cost[used], used in good, False
            rows.append({"circuit_key": key, "above": above, "policy": policy, "seconds": seconds, "good": ok,
                         "skipped": skipped, "scored": bool(good)})
    return pd.DataFrame(rows)


def summarize(r):
    out = {}
    order = list(dict.fromkeys(r.policy))
    for policy in order:
        g = r[r.policy == policy]
        below, above = g[~g.above], g[g.above]
        out[policy] = {
            "cpu_hours": g.seconds.sum() / 3600, "cpu_hours_below": below.seconds.sum() / 3600,
            "cpu_hours_above": above.seconds.sum() / 3600, "good_below": float(below[below.scored].good.mean()),
            "scored_below": int(below.scored.sum()),
            "skipped_below": float(below.skipped.mean()), "skipped_above": float(above.skipped.mean()),
        }
    return out


def markdown(summary, n_below, n_above, skip_above):
    base = summary["everything"]["cpu_hours"]
    lines = [
        "# End to end: a benchmark campaign with Matchmaker deciding", "",
        f"ECCentric's sweep replayed on its own measurements (`experiments/end_to_end.py`): {n_below + n_above:,} circuits "
        f"on which BP-OSD, BP-LSD, BeliefFind, Relay-BP and Tesseract were all run, {n_below:,} below threshold and "
        f"{n_above:,} above. Each circuit is predicted by a model that never saw its code. Matchmaker skips a circuit "
        f"when it predicts P(above threshold) > {skip_above:g}; a policy is charged the CPU time its decodes took in the "
        "sweep. Good enough: the chosen decoder's rate is not significantly worse than the best measured one, over the "
        f"{summary['everything']['scored_below']:,} circuits below threshold with a decode that reached its error target.", "",
        "| policy | CPU-hours | of the sweep | below threshold: good enough | wrongly skipped | above threshold: "
        "skipped | CPU-hours spent above |", "|---|---|---|---|---|---|---|"]
    for policy, s in summary.items():
        lines.append(f"| {policy} | {s['cpu_hours']:,.0f} | {100 * s['cpu_hours'] / base:.1f}% | {100 * s['good_below']:.1f}% "
                     f"| {100 * s['skipped_below']:.1f}% | {100 * s['skipped_above']:.1f}% | {s['cpu_hours_above']:,.0f} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "data"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--skip-above", type=float, default=0.5)
    a = p.parse_args()
    df, _ = load_dataset(a.data)
    r = replay(df, predictions(df, a.jobs), a.skip_above)
    summary = summarize(r)
    circuits = r.drop_duplicates("circuit_key")
    n_below, n_above = int((~circuits.above).sum()), int(circuits.above.sum())
    text = markdown(summary, n_below, n_above, a.skip_above)
    os.makedirs(os.path.join(HERE, "experiments", "results"), exist_ok=True)
    with open(os.path.join(HERE, "experiments", "results", "end_to_end.json"), "w") as f:
        json.dump({"circuits_below": n_below, "circuits_above": n_above, "skip_above": a.skip_above,
                   "policies": summary}, f, indent=1)
    with open(os.path.join(HERE, "experiments", "END_TO_END.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
