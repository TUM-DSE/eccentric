"""GPU pilots of one prepared circuit: per decoder, build it (timed), warm up on 64 shots, then decode batches
(4,096 shots first, then up to 32,768) until 10^5 shots or 120 s; results appended per decoder to
results/<i>.jsonl (a decoder with a 'start' line and no result line was cut off by the circuit's timeout)"""
import json, math, os, pickle, sys, time
import numpy as np

HERE = "/home/manosgior/eccentric-validation/gpu_dev/mm_pilot"
i = int(sys.argv[1])
CAP_SECONDS, MAX_SHOTS, MIN_SHOTS = 120.0, 100_000, 4096
SKIP_PREDICTED = 1500.0   # seconds for 4,096 shots at the lowest speedup measured: recorded as censored, not run
SPEEDUP_LOW = {"relay_bp": 86, "gari": 21, "bposd_tuned": 27, "bplsd": 36, "belief_find": 36, "bposd": 12}
ORDER = ["relay_bp", "gari", "bposd_tuned", "bplsd", "belief_find", "bposd"]
p = pickle.load(open(f"{HERE}/prep/{i}.pkl", "rb"))
o = p["slice"]
out_path = f"{HERE}/results/{i}.jsonl"


def write(rec):
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush(); os.fsync(f.fileno())


write(dict(kind="circuit", index=i, key=o["key"], family=o["family"], prep_seconds=p["prep_seconds"],
           errors=p["errors"], fatal=p.get("fatal")))
if p.get("fatal"):
    sys.exit(0)
import stim, cupy as cp
from eccentric.decoders.gpu import runtime
from eccentric.decoders.bp_osd import BP
sampler = stim.Circuit.from_file(o["file"]).compile_detector_sampler(seed=i)


def build(dec):
    if dec in ("bposd", "bposd_tuned"):
        from eccentric.decoders.gpu.nvqldpc import NvQldpc
        chk_variant = o["code"] in ("steane", "bacon")
        chk, obs, priors = p["sum"] if chk_variant else p["xor"]
        bp = BP["tuned" if dec == "bposd_tuned" else "paper"]
        order = 7 if chk_variant else min(7, chk.shape[1] - chk.shape[0])
        return NvQldpc(chk, obs, priors, bp_method=1, scale_factor=float(bp["ms_scaling_factor"]), max_iterations=int(bp["max_iter"]),
                       use_osd=True, osd_method=3, osd_order=int(order), osd_batch_size=1024)
    if dec in ("bplsd", "belief_find"):
        from eccentric.decoders.gpu.ldpc_gpu import GpuBpPost
        chk, obs, priors = p["xor"]
        if dec == "bplsd":
            from ldpc.lsd_decoder import LsdDecoder
            lsd = LsdDecoder(chk, bits_per_step=1, lsd_method="lsd_cs", lsd_order=min(7, chk.shape[1] - chk.shape[0]))
            return GpuBpPost(chk, obs, priors, lambda s, llr: lsd.decode(s, llr), max_iter=10000, bp_method="minimum_sum", ms_scaling_factor=1.0)
        from ldpc import UnionFindDecoder
        seen = np.asarray(chk.sum(axis=0)).ravel() > 0
        chk, obs, priors = chk[:, seen], obs[:, seen], np.asarray(priors)[seen]
        uf = UnionFindDecoder(chk, uf_method="inversion")
        return GpuBpPost(chk, obs, priors, lambda s, llr: uf.decode(s, llrs=llr, bits_per_step=0), max_iter=10000, bp_method="minimum_sum", ms_scaling_factor=1.0)
    if dec == "relay_bp":
        from eccentric.decoders.gpu.relay_gpu import RelayGpu
        return RelayGpu(*p["relay"])
    if dec == "gari":
        from eccentric.decoders.gpu.gari_gpu import GariGpu
        return GariGpu(p["gari"])


for dec in ORDER:
    if dec not in o["decoders"]:
        continue
    base = dict(kind="pilot", index=i, key=o["key"], decoder=dec, cpu_ms_per_shot=o["decoders"][dec])
    if dec in p["errors"]:
        write(dict(base, status="not applicable", error=p["errors"][dec])); continue
    predicted = MIN_SHOTS * o["decoders"][dec] / 1000 / SPEEDUP_LOW[dec]
    if predicted > SKIP_PREDICTED:
        write(dict(base, status="skipped: predicted too slow", predicted_seconds_4096=predicted)); continue
    write(dict(base, kind="start", time=time.time()))
    try:
        t = time.time(); g = build(dec); build_s = time.time() - t
        warm = sampler.sample(64, separate_observables=True)[0]
        t = time.time(); g.predict(warm); warm_s = time.time() - t
        if hasattr(g, "stats"):
            g.stats.update({k: 0 for k in g.stats})
        batches, total, elapsed, sample_s = [], 0, 0.0, 0.0
        n = MIN_SHOTS
        while total < MAX_SHOTS and (elapsed < CAP_SECONDS or total < MIN_SHOTS):
            t = time.time(); dets = sampler.sample(n, separate_observables=True)[0]; sample_s += time.time() - t
            t = time.time(); g.predict(dets); dt = time.time() - t
            batches.append([n, dt]); total += n; elapsed += dt
            n = min(32768, MAX_SHOTS - total)
            if n <= 0:
                break
        stats = getattr(g, "stats", {})
        write(dict(base, status="measured", build_seconds=build_s, warmup_seconds=warm_s, sampling_seconds=sample_s,
                   shots=total, seconds=elapsed, batches=batches, gpu_shots_per_second=total / elapsed,
                   shared_memory=bool(getattr(getattr(g, "bp", g), "use_shared", False)),
                   postprocessed=stats.get("postprocessed"), stats=stats))
    except Exception as e:
        write(dict(base, status="error", error=repr(e)[:500]))
    finally:
        try:
            del g
        except NameError:
            pass
        cp.get_default_memory_pool().free_all_blocks()
