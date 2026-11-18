"""GPU decoder on a circuit's saved shots: predictions and timing (decoding only; the decoder is built and warmed
up on 1,000 shots first)"""
import json, os, sys, time
import numpy as np
from common import DATA, load_shots, gpu_predictor
tag, decoder = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else None
dets, obs = load_shots(tag, n)
t0 = time.time(); g = gpu_predictor(tag, decoder); build = time.time() - t0
g.predict(dets[:1000])
for key in ("stats",):
    if hasattr(g, key): getattr(g, key).update(shots=0, postprocessed=0)
batch = 1 << 17
preds = []
t = time.time()
for a in range(0, len(dets), batch):
    preds.append(g.predict(dets[a:a + batch]))
dt = time.time() - t
pred = np.concatenate(preds).astype(bool)
np.save(os.path.join(DATA, f"{tag}.{decoder}.gpu.npy"), np.packbits(pred, axis=1, bitorder="little"))
info = dict(shots=len(dets), seconds=dt, build_seconds=build, ler=float(np.any(pred != obs, axis=1).mean()))
if hasattr(g, "stats"): info["postprocessed_on_cpu"] = int(g.stats["postprocessed"])
if hasattr(g, "stuck"): info["uf_stuck"] = int(g.stuck)
json.dump(info, open(os.path.join(DATA, f"{tag}.{decoder}.gpu.json"), "w"))
print(tag, decoder, info, flush=True)
