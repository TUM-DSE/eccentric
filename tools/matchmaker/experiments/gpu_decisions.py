"""What the GPU option changes: for every circuit of the training table, the recommended decoder and its
predicted time per shot without and with a GPU (packaged model), for the accurate and cheapest objectives.

    python experiments/gpu_decisions.py > experiments/results/gpu_decisions.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qec_matchmaker.dataset import load_dataset  # noqa: E402
from qec_matchmaker.models import ModelBundle, circuit_features  # noqa: E402
from qec_matchmaker.recommend import recommend  # noqa: E402

df, _ = load_dataset(os.path.join(ROOT, "data"))
bundle = ModelBundle.load(os.path.join(ROOT, "qec_matchmaker", "model"))
feats = circuit_features(df).reset_index()
pred = {gpu: bundle.predict(feats, gpu=gpu) for gpu in (False, True)}
out = {}
for objective in ("accurate", "cheapest"):
    rows = []
    for key in feats.circuit_key:
        r = {}
        for gpu in (False, True):
            p = pred[gpu][pred[gpu].circuit_key == key]
            choice = recommend(p, objective=objective)
            if choice.decoder is None:
                break
            row = p[p.decoder == choice.decoder].iloc[0]
            r[gpu] = (choice.decoder, float(row.ms_q50), row.device)
        if len(r) == 2:
            rows.append(r)
    base = lambda name: name.removesuffix("_gpu")  # noqa: E731
    to_gpu = [r for r in rows if r[True][2] == "gpu"]
    other_algorithm = [r for r in rows if base(r[True][0]) != r[False][0]]
    ratio = np.array([r[False][1] / r[True][1] for r in to_gpu])
    out[objective] = {
        "circuits": len(rows),
        "recommends_a_gpu_decoder": len(to_gpu),
        "same_algorithm_on_the_gpu": sum(1 for r in to_gpu if base(r[True][0]) == r[False][0]),
        "different_algorithm": len(other_algorithm),
        "different_algorithm_examples": sorted({f"{r[False][0]} -> {r[True][0]}" for r in other_algorithm})[:12],
        "speedup_where_gpu_chosen_median": float(np.median(ratio)) if len(ratio) else None,
        "speedup_where_gpu_chosen_p10_p90": [float(np.quantile(ratio, 0.1)), float(np.quantile(ratio, 0.9))] if len(ratio) else None,
        "gpu_choices": {d: sum(1 for r in to_gpu if r[True][0] == d) for d in sorted({r[True][0] for r in to_gpu})},
        "cpu_choices_kept": {d: sum(1 for r in rows if r[True][2] != "gpu" and r[True][0] == d)
                             for d in sorted({r[True][0] for r in rows if r[True][2] != "gpu"})},
    }
print(json.dumps(out, indent=1))
