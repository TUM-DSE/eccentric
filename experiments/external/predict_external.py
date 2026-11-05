"""Matchmaker's predictions on circuits from outside ECCentric (make_external.py), against what ECCentric's
decoders measured on them. The packaged model never saw these circuits or their generators.

python experiments/external/predict_external.py <run_dir> [--model <dir>]
<run_dir> holds circuits.json, circuits/ and results/ (copied from the cluster). Writes
experiments/results/external.json and experiments/EXTERNAL.md. Every circuit's code distance (in its name)
is given to the model, as a user who knows their code would.

Results are turned into rows as ECCentric's sweep.collect does (measured: at least the target errors; low
statistics: fewer; upper bound: none, with the 95% Wilson bound; timeout), and scored with the same
functions as the held-out evaluation (qec_matchmaker.evaluate).
"""
import argparse
import json
import math
import os
import re

import numpy as np
import pandas as pd

from qec_matchmaker.dataset import family
from qec_matchmaker.evaluate import _ler, _recommendation, _regime, _regret, _time, _f, _pct
from qec_matchmaker.features import features_from_circuit
from qec_matchmaker.models import ModelBundle, circuit_features, circuit_labels
from qec_matchmaker.api import MODEL_DIR
from qec_matchmaker.recommend import recommend

Z95 = 1.959963984540054
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(errors, shots):
    p, denom = errors / shots, 1 + Z95 ** 2 / shots
    centre = (p + Z95 ** 2 / (2 * shots)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / shots + Z95 ** 2 / (4 * shots ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rows(run_dir):
    index = json.load(open(os.path.join(run_dir, "circuits.json")))
    features = {name: features_from_circuit(__import__("stim").Circuit.from_file(os.path.join(run_dir, "circuits", f"{name}.stim")))
                for name in index}
    out = []
    for f in sorted(os.listdir(os.path.join(run_dir, "results"))):
        m = re.fullmatch(r"decode\.external\.(.+)\.decoder=(\w+)\.json", f)
        if not m:
            continue
        name, decoder = m.groups()
        job = json.load(open(os.path.join(run_dir, "jobs", f)))
        r = json.load(open(os.path.join(run_dir, "results", f)))
        row = dict(circuit_key=f"external/{name}", source="decoder-bench" if name.startswith("decoder_bench") else "stim",
                   code=index[name]["code"], family=family(index[name]["code"]), decoder=decoder,
                   code_distance=float(re.search(r"_d(\d+)_", name)[1]), **features[name])
        if r["status"] == "failed" and str(r.get("error", "")).startswith("timeout after"):
            row.update(status="timeout", timeout_s=float(re.search(r"(\d+)", r["error"])[1]))
        elif r["status"] != "done" or not r.get("shots"):
            continue  # the decoder could not decode the circuit
        else:
            shots, errors = r["shots"], r["errors"]
            low, high = wilson(errors, shots)
            status = "upper bound" if errors == 0 else "measured" if errors >= job["target_errors"] else "low statistics"
            row.update(status=status, shots=shots, errors=errors, ler=errors / shots if errors else math.nan,
                       ler_low=low, ler_high=high, ms_per_shot=1000 * r["seconds"] / shots)
        out.append(row)
    columns = ["status", "shots", "errors", "ler", "ler_low", "ler_high", "ms_per_shot", "timeout_s"]
    return pd.DataFrame(out).reindex(columns=list(dict.fromkeys([*out[0], *columns])))


def per_circuit(df, pred):
    labels = circuit_labels(df)
    predicted = dict(tuple(pred.groupby("circuit_key", sort=False)))
    out = []
    for key, s in df.groupby("circuit_key", sort=False):
        p = predicted[key]
        rated = s[s.status.isin(["measured", "low statistics", "upper bound"])]
        best = rated.loc[rated.assign(r=np.where(rated.errors > 0, rated.ler, rated.ler_high)).r.idxmin()] if len(rated) else None
        out.append({"circuit": key.split("/", 1)[1], "detectors": int(s.detectors.iloc[0]),
                    "best_measured": None if best is None else f"{best.decoder} {best.ler if best.errors else best.ler_high:.2g}"
                    + ("" if best is None or best.errors else " (bound)"),
                    "predicted_best": f"{p.ler_q50.iloc[0]:.2g} [{p.ler_q10.iloc[0]:.2g}, {p.ler_q90.iloc[0]:.2g}]",
                    "p_above": float(p.p_above.iloc[0]), "above": bool(labels.above.get(key, False)),
                    "accurate": recommend(p, "accurate").decoder, "cheapest": recommend(p, "cheapest").decoder,
                    "outside_training": bool(p.outside_training.iloc[0]) if "outside_training" in p else None})
    return out


def score(df, pred):
    return {"circuits": int(df.circuit_key.nunique()), "decodes": len(df), "regime": _regime(df, pred),
            "ler": _ler(df, pred), "regret": _regret(df, pred), "time": _time(df, pred),
            "recommendation": _recommendation(df, pred)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--model", default=MODEL_DIR)
    args = parser.parse_args()
    df = rows(args.run_dir)
    bundle = ModelBundle.load(args.model)
    circuits = circuit_features(df)
    pred = bundle.predict(circuits.reset_index(), sorted(df.decoder.unique()))
    result = {"all": score(df, pred)}
    for source, s in df.groupby("source"):
        result[source] = score(s, pred[pred.circuit_key.isin(s.circuit_key)])
    result["circuits"] = per_circuit(df, pred)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "external.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    with open(os.path.join(HERE, "EXTERNAL.md"), "w") as f:
        f.write(report(result, bundle))
    print(report(result, bundle))


def report(result, bundle):
    lines = ["# Circuits from outside ECCentric", "",
             "Matchmaker's packaged model (trained on ECCentric only) on Stim's generated memory circuits with uniform "
             "circuit noise and on decoder-bench's circuits with SI1000 noise, decoded and timed by ECCentric's sweep "
             "worker like the training data (target 100 errors, at most 200,000 shots or 10 minutes). The model is given "
             "each circuit's code distance. Errors are in decades (|log10 error|); intervals are the model's "
             "conformalized [q10, q90]. Trees alone, without the scaling laws: see EXTRAPOLATION.md.", "",
             "| set | circuits | regime correct | best-LER median error | p90 | in [q10, q90] | time within 3x | "
             "choice good enough (accurate / cheapest) | always BP-OSD | always Tesseract |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("all", "stim", "decoder-bench"):
        if name not in result:
            continue
        r = result[name]
        pol = r["recommendation"]["policies"]
        g = lambda k: _pct(pol[k]["good_enough"]) if k in pol else "n/a"
        lines.append(f"| {name} | {r['circuits']} | {_pct(r['regime']['accuracy'])} | {_f(r['ler']['median_abs_log10_error'])} | "
                     f"{_f(r['ler']['p90_abs_log10_error'])} | {_pct(r['ler']['coverage_q10_q90'])} | "
                     f"{_pct(r['time']['decoders']['all']['within_3x'])} | {g('recommend, accurate')} / {g('recommend, cheapest')} | "
                     f"{g('always bposd')} | {g('always tesseract')} |")
    lines += ["", "Time per shot within 3x, per decoder (all circuits): " + ", ".join(
        f"{d} {_pct(m['within_3x'])} ({m['decodes']})" for d, m in result["all"]["time"]["decoders"].items() if d != "all") + ".", "",
        "| circuit | detectors | best measured | predicted best LER [q10, q90] | P(above) | recommended (accurate / cheapest) | outside training |",
        "|---|---|---|---|---|---|---|"]
    for c in result["circuits"]:
        lines.append(f"| {c['circuit']} | {c['detectors']} | {c['best_measured']} | {c['predicted_best']} | {c['p_above']:.2f} | "
                     f"{c['accurate']} / {c['cheapest']} | {'yes' if c['outside_training'] else ''} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
