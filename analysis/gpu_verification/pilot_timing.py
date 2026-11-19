"""How well does a short GPU pilot predict a long run's throughput? Per (circuit, decoder): decoder build time,
stim sampling time, decode time of pilots (10^3, 10^4, 10^5 shots at three offsets) and of a reference run
(repeated twice), all on the saved shots, in production-sized batches"""
import json, os, sys, time
import numpy as np
from common import DATA, circuit, load_shots, gpu_predictor

tag, decoder, ref = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = os.path.join(DATA, "pilot", f"{tag}.{decoder}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
BATCH = 1 << 16 if decoder in ("gari", "union_find") else 1 << 18   # decode.py's batch sizes
dets, _ = load_shots(tag)
res = dict(circuit=tag, decoder=decoder, batch=BATCH, reference_shots=ref)
t = time.time(); g = gpu_predictor(tag, decoder); res["build_seconds"] = time.time() - t
t = time.time(); g.predict(dets[:1000]); res["first_call_seconds"] = time.time() - t
c = circuit(tag)
sampler = c.compile_detector_sampler()
t = time.time(); sampler.sample(ref, separate_observables=True); res["sampling_seconds_reference"] = time.time() - t

def timed(a, n):
    t = time.time()
    for s in range(a, a + n, BATCH):
        g.predict(dets[s:min(a + n, s + BATCH)])
    return time.time() - t

res["reference_seconds"] = [timed(0, ref), timed(0, ref)]
res["pilots"] = {}
for n in (1000, 10000, 100000):
    if n >= ref:
        continue
    offsets = [0, (len(dets) - n) // 2, len(dets) - n]
    res["pilots"][n] = [dict(offset=a, seconds=timed(a, n)) for a in offsets]
json.dump(res, open(out, "w"), indent=1)
ref_rate = ref / np.mean(res["reference_seconds"])
print(tag, decoder, f"ref {ref_rate:,.0f}/s (runs {res['reference_seconds'][0]:.1f}s, {res['reference_seconds'][1]:.1f}s); "
      + "; ".join(f"pilot {n}: " + ",".join(f"{(n / p['seconds']) / ref_rate:.2f}" for p in ps) for n, ps in res["pilots"].items())
      + f"; build {res['build_seconds']:.1f}s, sampling {ref / res['sampling_seconds_reference']:,.0f}/s", flush=True)
