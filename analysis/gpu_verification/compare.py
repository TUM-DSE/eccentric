"""GPU against CPU on the same shots: identical predictions, logical error rates, paired test, speed"""
import glob, json, math, os, sys
import numpy as np
from common import CIRCUITS, DATA, load_shots
rows = []
for tag in CIRCUITS:
    dets, obs = load_shots(tag)
    for gj in sorted(glob.glob(os.path.join(DATA, f"{tag}.*.gpu.json"))):
        decoder = os.path.basename(gj)[len(tag) + 1:-len(".gpu.json")]
        g = json.load(open(gj))
        gp = np.unpackbits(np.load(gj[:-5] + ".npy"), axis=1, count=obs.shape[1], bitorder="little").astype(bool)
        cpu_decoder = decoder.replace("_nvq", "")   # the CPU decoder a GPU variant is compared with
        parts = sorted(glob.glob(os.path.join(DATA, "parts", f"{tag}.{cpu_decoder}.cpu.*.json")))
        if not parts:
            continue
        idx, cp_, secs = [], [], 0.0
        for pj in parts:
            info = json.load(open(pj)); secs += info["seconds"]
            idx.append(np.arange(info["start"], info["end"]))
            cp_.append(np.unpackbits(np.load(pj[:-5] + ".npy"), axis=1, count=obs.shape[1], bitorder="little").astype(bool))
        idx = np.concatenate(idx); cpred = np.concatenate(cp_)
        n = len(idx)
        eg_all = np.any(gp != obs[:len(gp)], axis=1)
        keep = idx < len(gp)
        idx, cpred = idx[keep], cpred[keep]
        n = len(idx)
        eg, ec = eg_all[idx], np.any(cpred != obs[idx], axis=1)
        same = np.all(gp[idx] == cpred, axis=1).mean()
        b, c = int(np.sum(eg & ~ec)), int(np.sum(ec & ~eg))
        z = (b - c) / math.sqrt(b + c) if b + c else 0.0
        se = lambda p, k: math.sqrt(max(p * (1 - p), 1e-300) / k)
        rows.append(dict(circuit=tag, decoder=decoder, gpu_shots=g["shots"], cpu_shots=n, identical=same,
                         ler_gpu_all=eg_all.mean(), se_gpu_all=se(eg_all.mean(), g["shots"]),
                         ler_gpu=eg.mean(), ler_cpu=ec.mean(), se_cpu=se(ec.mean(), n), only_gpu_wrong=b, only_cpu_wrong=c,
                         paired_z=z, gpu_rate=g["shots"] / g["seconds"], cpu_rate=n / secs,
                         speedup=(g["shots"] / g["seconds"]) / (n / secs), cpu_post=g.get("postprocessed_on_cpu")))
json.dump(rows, open(os.path.join(DATA, "comparison.json"), "w"), indent=1)
print(f"{'circuit':22s} {'decoder':16s} {'GPU shots':>9s} {'CPU shots':>9s} {'identical':>9s} {'LER GPU (same shots)':>21s} {'LER CPU':>18s} {'b/c':>11s} {'z':>6s} {'GPU/s':>9s} {'CPU/s':>8s} {'x':>7s}")
for r in rows:
    print(f"{r['circuit']:22s} {r['decoder']:16s} {r['gpu_shots']:9d} {r['cpu_shots']:9d} {r['identical']:9.5f} "
          f"{r['ler_gpu']:.3e}            {r['ler_cpu']:.3e}±{r['se_cpu']:.1e} {r['only_gpu_wrong']:5d}/{r['only_cpu_wrong']:<5d} {r['paired_z']:6.2f} "
          f"{r['gpu_rate']:9.0f} {r['cpu_rate']:8.1f} {r['speedup']:7.0f}")
