import os, sys
import numpy as np
from common import CIRCUITS, DATA, circuit
os.makedirs(DATA, exist_ok=True)
shots = int(sys.argv[1])
for tag in CIRCUITS:
    path = os.path.join(DATA, f"{tag}.shots.npz")
    if os.path.exists(path):
        continue
    c = circuit(tag)
    dets, obs = c.compile_detector_sampler(seed=20261117).sample(shots, separate_observables=True)
    np.savez(path, dets=np.packbits(dets, axis=1, bitorder="little"), obs=np.packbits(obs, axis=1, bitorder="little"),
             num_dets=dets.shape[1], num_obs=obs.shape[1], shots=shots)
    print(tag, dets.shape, obs.shape, "raw obs flip rate", obs.any(axis=1).mean(), flush=True)
