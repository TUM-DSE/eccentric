"""How well Matchmaker predicts the GPU-only decoder's accuracy on codes it has not seen: the training table plus
tcbp_jobs_sc_gpu's slice decodes (results_tcbp_jobs_sc_gpu), 10 folds over codes (evaluate.cross_val_predictions);
for each held-out slice circuit, the predicted LER of tcbp_jobs_sc_gpu (best-LER model x its regret) against its
measured LER, and the predicted regret against the measured one; next to the same for the CPU decoders on the same
circuits. python heldout.py [jobs] -> heldout.json"""
import json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from qec_matchmaker.dataset import load_dataset
from qec_matchmaker.evaluate import cross_val_predictions
from qec_matchmaker.gpu import load_gpu_only_rows
from qec_matchmaker.models import regret_rows

table, _ = load_dataset(os.path.join(ROOT, "data"))
extra = load_gpu_only_rows(HERE, table)
df = pd.concat([table, extra], ignore_index=True)
pred = cross_val_predictions(df, folds=10, n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
slice_keys = set(extra.circuit_key)
measured = df[(df.status == "measured") & (df.ler > 0) & df.circuit_key.isin(slice_keys)]
m = measured.merge(pred[["circuit_key", "decoder", "ler_decoder_q50", "regret_q50"]], on=["circuit_key", "decoder"])
reg = regret_rows(df).merge(pred[["circuit_key", "decoder", "regret_q50"]], on=["circuit_key", "decoder"])
out = {"slice_circuits": len(slice_keys), "folds": 10, "held_out": "code"}
for name, rows, rr in (("tcbp_jobs_sc_gpu", m[m.decoder == "tcbp_jobs_sc_gpu"], reg[reg.decoder == "tcbp_jobs_sc_gpu"]),
                       ("cpu decoders, same circuits", m[m.decoder != "tcbp_jobs_sc_gpu"],
                        reg[(reg.decoder != "tcbp_jobs_sc_gpu") & reg.circuit_key.isin(slice_keys)])):
    e = np.abs(np.log10(rows.ler_decoder_q50.to_numpy(float)) - np.log10(rows.ler.to_numpy(float)))
    r = np.abs(np.log10(rr.regret_q50.to_numpy(float)) - rr.regret.to_numpy(float))
    out[name] = {"decodes": int(len(rows)),
                 "ler_median_abs_log10": float(np.nanmedian(e)), "ler_p90_abs_log10": float(np.nanquantile(e, 0.9)),
                 "ler_within_3x": float(np.nanmean(e <= math.log10(3))),
                 "regret_median_abs_log10": float(np.nanmedian(r)), "regret_p90_abs_log10": float(np.nanquantile(r, 0.9))}
print(json.dumps(out, indent=1))
with open(os.path.join(HERE, "heldout.json"), "w") as f:
    json.dump(out, f, indent=1)
