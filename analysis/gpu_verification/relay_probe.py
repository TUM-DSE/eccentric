"""relay_bp_gpu on hgp_hamming d3: which nv-qldpc setting explains the gap to relay-bp (first 2x10^5 saved shots)"""
import glob, json, os
import numpy as np, scipy.sparse as sp
from common import DATA, circuit, load_shots
from eccentric.decoders.gpu import nvqldpc

tag = "hgp_hamming_d3_p001"
N = 200000
dets, obs = load_shots(tag, N)
cp_ = []
for pj in sorted(glob.glob(os.path.join(DATA, "parts", f"{tag}.relay_bp.cpu.*.json"))):
    info = json.load(open(pj))
    if info["start"] >= N: continue
    cp_.append(np.unpackbits(np.load(pj[:-5] + ".npy"), axis=1, count=obs.shape[1], bitorder="little").astype(bool))
cpu = np.concatenate(cp_)[:N]
ec = np.any(cpu != obs, axis=1)
print(f"CPU relay-bp LER {ec.mean():.5f} on {N} shots", flush=True)

from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP
m = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=circuit(tag).detector_error_model()).check_matrices
H, O = sp.csr_matrix(m.check_matrix), sp.csr_matrix(m.observables_matrix)
priors = np.asarray(m.error_priors, dtype=np.float64)
base = dict(nvqldpc.RELAY)
variants = {
    "current (repeatable, clip 200)": {},
    "repeatable, clip 1e4": dict(clip_value=1e4),
    "not repeatable, no clip": dict(repeatable=False, clip_value=None),
    "current, bp_seed 7": dict(bp_seed=7),
    "fp32 NVIDIA preset clip": dict(proc_float="fp32"),
}
for label, change in variants.items():
    opts = dict(base, **change)
    opts = {k: v for k, v in opts.items() if v is not None}
    if "proc_float" in opts:
        pf = opts.pop("proc_float")
    else:
        pf = None
    try:
        d = nvqldpc.NvQldpc(H, O, priors, m.syndrome_bias, m.observables_bias, **opts)
        if pf:
            import cudaq_qec as qec
            d.decoder = qec.get_decoder("nv-qldpc-decoder", np.ascontiguousarray(H.toarray().astype(np.uint8)),
                                        error_rate_vec=[float(p) for p in priors], use_sparsity=True, proc_float=pf,
                                        bp_batch_size=d.chunk, **opts)
    except Exception as e:
        print(label, "error:", e, flush=True); continue
    # per-shot convergence
    syn = dets.astype(np.uint8) ^ (m.syndrome_bias.astype(np.uint8) if m.syndrome_bias is not None else 0)
    pred, conv = [], []
    for a in range(0, N, d.chunk):
        res = d.decoder.decode_batch(np.ascontiguousarray(syn[a:a + d.chunk]))
        E = np.array([r.result for r in res], dtype=np.float32) > 0.5
        conv.append([bool(r.converged) for r in res])
        p = (O @ E.T.astype(np.uint8)).T % 2
        if m.observables_bias is not None: p ^= m.observables_bias.astype(np.uint8)
        pred.append(p.astype(bool))
    pred, conv = np.concatenate(pred), np.concatenate(conv)
    eg = np.any(pred != obs, axis=1)
    b, c = int(np.sum(eg & ~ec)), int(np.sum(ec & ~eg))
    print(f"{label:32s} LER {eg.mean():.5f}; z {(b-c)/np.sqrt(b+c):6.2f}; not converged {100*(~conv).mean():.2f}% "
          f"(LER there GPU {eg[~conv].mean():.3f} CPU {ec[~conv].mean():.3f}; on converged GPU {eg[conv].mean():.5f} CPU {ec[conv].mean():.5f})", flush=True)
