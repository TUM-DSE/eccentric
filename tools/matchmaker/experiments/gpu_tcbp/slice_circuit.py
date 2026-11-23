"""tcbp_jobs_gpu (or argv[2], e.g. tcbp_jobs_sc_gpu) on one circuit of the GPU slice (../gpu/slice.json):
python slice_circuit.py <i> [<decoder>]

1. Logical error rate: ECCentric's adaptive decode with the settings and seeds of the sweep's CPU decode jobs of
   the same circuit (target errors, max shots, time budget, batches), so the decoder sees the same shots as the
   CPU decoders did; the code family's TCBP model (eccentric/decoders/gpu/tcbp_models).
2. GPU time, as the GPU pilots (../gpu/pilot_circuit.py): build (timed), warm up on 64 shots, then batches
   (4,096 shots first, then up to 32,768) until 10^5 shots or 120 s.
Results: results/<i>.json for tcbp_jobs_gpu, results_<decoder>/<i>.json otherwise (written once both parts finish; a part that fails records its error)."""
import glob, json, os, sys, time
import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/manosgior/eccentric-validation"
SLICE = os.path.join(HERE, "slice.json")  # a copy of ../gpu/slice.json
i = int(sys.argv[1])
DECODER = sys.argv[2] if len(sys.argv) > 2 else "tcbp_jobs_gpu"
PRESET = {"tcbp_jobs_gpu": "tcbp-jobs", "tcbp_jobs_sc_gpu": "tcbp-jobs-sc"}[DECODER]
RESULTS = os.path.join(HERE, "results" if DECODER == "tcbp_jobs_gpu" else f"results_{DECODER}")
o = json.load(open(SLICE))[i]
out = {"index": i, "key": o["key"], "code": o["code"], "family": o["family"], "decoder": DECODER}

# The CPU decode jobs of this circuit: decode.<circuit id without "circuit">.decoder=<d>.json
run, _, cid = o["key"].partition("/")
pattern = os.path.join(ROOT, run, "jobs", "decode" + cid[len("circuit"):] + ".decoder=*.json")
jobs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
SETTINGS = ("target_errors", "max_shots", "time_budget", "first_batch", "max_batch", "seed_offset")
if not jobs:
    out["error"] = f"no CPU decode job matches {pattern}"
else:
    cpu = jobs[0]
    settings = {k: cpu[k] for k in SETTINGS if k in cpu}
    differ = sorted({k for j in jobs for k in SETTINGS if j.get(k) != cpu.get(k)})
    out.update(settings=settings, cpu_job=cpu["id"], settings_differ_across_cpu_jobs=differ)

from eccentric.decoders import decode
from validation.run_job import decode_adaptively
circuit = stim.Circuit.from_file(o["file"])
if "settings" in out:
    job = {**cpu, "decoder": DECODER}
    start = time.time()
    try:
        r = decode_adaptively(job, lambda n: decode(job["code"], circuit, n, DECODER, job["backend"],
                                                    job["error_type"], distance=job.get("distance")))
        out["ler"] = {**r, "wall_seconds": time.time() - start}
    except Exception as e:
        out["ler"] = {"error": repr(e)[:500]}

# GPU time (the pilots' protocol)
from eccentric.decoders.gpu import nvqldpc
from eccentric.decoders.gpu.tcbp_models import load as _load_model  # noqa: F401 (fails early without a model)
backend, error_type = o["backend"], o["error_type"]
ade = backend.split("_")[0] in ("variance", "real", "native") and (error_type == "variance" or error_type.split("_")[0] == "real")
try:
    import cupy as cp
    sampler = circuit.compile_detector_sampler(seed=i)
    t = time.time(); g = nvqldpc.tcbp_jobs_decoder(circuit, ade, o["code"], PRESET); build_s = time.time() - t
    warm = sampler.sample(64, separate_observables=True)[0]
    g.predict(warm)
    batches, total, elapsed, n = [], 0, 0.0, 4096
    while total < 100_000 and (elapsed < 120.0 or total < 4096):
        dets = sampler.sample(n, separate_observables=True)[0]
        t = time.time(); g.predict(dets); dt = time.time() - t
        batches.append([n, dt]); total += n; elapsed += dt
        n = min(32768, 100_000 - total)
        if n <= 0:
            break
    out["timing"] = dict(build_seconds=build_s, shots=total, seconds=elapsed, batches=batches,
                         gpu_shots_per_second=total / elapsed, gpu_ms_per_shot=1000 * elapsed / total,
                         bp_converged_fraction=g.converged / max(g.shots, 1))
    del g
    cp.get_default_memory_pool().free_all_blocks()
except Exception as e:
    out["timing"] = {"error": repr(e)[:500]}

os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, f"{i}.json.tmp"), "w") as f:
    json.dump(out, f, indent=1)
os.replace(os.path.join(RESULTS, f"{i}.json.tmp"), os.path.join(RESULTS, f"{i}.json"))
