"""relay_bp_gpu's kernel against relay-bp with the same explicit memory strengths: identical error vectors?"""
import sys, time
import numpy as np, scipy.sparse as sp
import relay_bp
from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP
from common import circuit, load_shots
from eccentric.decoders.gpu.relay_gpu import RelayGpu, RELAY

n_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
for tag in ("surface_d5_p002", "bb72_d6_p002", "hgp_hamming_d3_p001", "color_d5_p002"):
    dets, obs = load_shots(tag, n_shots)
    m = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=circuit(tag).detector_error_model()).check_matrices
    n = m.check_matrix.shape[1]
    table = np.random.default_rng(1).uniform(*RELAY["gamma_dist_interval"], size=(RELAY["num_sets"], n))
    cpu = relay_bp.RelayDecoderF64(m.check_matrix, error_priors=m.error_priors, alpha=None, gamma0=RELAY["gamma0"],
                                   pre_iter=RELAY["pre_iter"], num_sets=RELAY["num_sets"], set_max_iter=RELAY["set_max_iter"],
                                   gamma_dist_interval=RELAY["gamma_dist_interval"], explicit_gammas=table,
                                   stop_nconv=RELAY["stop_nconv"], stopping_criterion="nconv")
    syn = dets.astype(np.uint8)
    if m.syndrome_bias is not None: syn = syn ^ m.syndrome_bias.astype(np.uint8)
    t = time.time(); ec = np.asarray(cpu.decode_batch(syn)); tc = time.time() - t
    g = RelayGpu(m.check_matrix, m.error_priors, m.observables_matrix, m.syndrome_bias, m.observables_bias, explicit_gammas=table)
    # the kernel's chosen decodings, before mapping to observables
    import cupy as cp
    from eccentric.decoders.gpu import runtime
    best = cp.empty((len(syn), n), dtype=cp.uint8); conv = cp.empty(len(syn), dtype=cp.uint8); legs = cp.empty(len(syn), dtype=cp.int32)
    g.counter.fill(0); st = g.settings; null = np.uint64(0)
    t = time.time()
    g.kernel((int(min(g.blocks, len(syn))),), (g.threads,),
             (np.int32(g.m), np.int32(g.n), g.row_ptr, g.row_col, g.col_ptr, g.col_edge, g.lpr, g.quality_lpr, cp.asarray(syn),
              np.int32(len(syn)), np.float64(st["gamma0"]), np.int32(st["pre_iter"]), np.int32(st["num_sets"]),
              np.int32(st["set_max_iter"]), np.int32(st["stop_nconv"]), g.table, np.int32(g.table_rows),
              np.float64(-0.24), np.float64(0.66), np.uint64(0), np.int64(0), g.workspace if g.workspace is not None else null,
              np.int32(int(g.use_shared)), best, conv, legs, g.counter), shared_mem=g.state_bytes if g.use_shared else 0)
    eg = best.get(); tg = time.time() - t
    same = np.all(eg == ec, axis=1)
    print(f"{tag:22s} identical error vectors {same.sum()}/{len(syn)}; GPU converged {conv.get().mean():.4f}, mean legs {legs.get().mean():.2f}; CPU {len(syn)/tc:,.0f}/s", flush=True)
    if not same.all():
        k = int(np.flatnonzero(~same)[0])
        print("   first mismatch shot", k, "cpu weight", ec[k].sum(), "gpu weight", eg[k].sum(), flush=True)
