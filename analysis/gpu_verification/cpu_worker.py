"""CPU decoder on shots [start, end) of a circuit's saved shots"""
import json, os, sys, time
import numpy as np
from common import DATA, load_shots, cpu_batch_predictor
tag, decoder, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out = os.path.join(DATA, "parts", f"{tag}.{decoder}.cpu.{start:07d}.{end:07d}")
if os.path.exists(out + ".json"):
    sys.exit(0)
dets, obs = load_shots(tag, end, start)
f = cpu_batch_predictor(tag, decoder)
t = time.time(); pred = f(dets).astype(bool); dt = time.time() - t
np.save(out + ".npy", np.packbits(pred, axis=1, bitorder="little"))
json.dump(dict(start=start, end=end, seconds=dt), open(out + ".json", "w"))
