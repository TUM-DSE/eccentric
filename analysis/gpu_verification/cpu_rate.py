import sys, time
import numpy as np
from common import CIRCUITS, load_shots, cpu_batch_predictor
decoders = sys.argv[1].split(",")
n = int(sys.argv[2])
tags = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CIRCUITS)
for tag in tags:
    dets, obs = load_shots(tag, n)
    for dec in decoders:
        if dec.startswith("hier") and not tag.startswith(("steane", "mhc")):
            continue
        try:
            f = cpu_batch_predictor(tag, dec)
            f(dets[:5])
            t = time.time(); p = f(dets); dt = time.time() - t
            print(f"{tag:22s} {dec:16s} {n/dt:9.1f} shots/s  LER {np.any(p != obs, axis=1).mean():.4f}", flush=True)
        except Exception as e:
            print(f"{tag:22s} {dec:16s} error {e}", flush=True)
