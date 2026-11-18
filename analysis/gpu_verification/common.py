"""Verification of ECCentric's GPU decoders against the CPU ones on the same shots"""
import os, time
import numpy as np, stim

DATA = "/home/manosgior/eccentric-validation/gpu_dev/verify"
S1 = "/home/manosgior/eccentric-validation/sweep1/circuits/circuit.reference.code={}.distance={}.cycles={}.backend=custom_full.backend_size={}.error_type=sd6.error_probability={}.stim"
S2 = "/home/manosgior/eccentric-validation/sweep_newcodes/circuits/circuit.reference.code={}.distance={}.cycles={}.backend=custom_full.backend_size={}.error_type=sd6.error_probability={}.stim"
CIRCUITS = {
    "surface_d5_p002": (S1.format("surface", 5, 5, 64, 0.002), "surface", 5),
    "color_d5_p002": (S1.format("color", 5, 5, 49, 0.002), "color", 5),
    "bb72_d6_p002": (S1.format("bb72", 6, 6, 144, 0.002), "bb72", 6),
    "gross_d12_p001": (S1.format("gross", 12, 12, 288, 0.001), "gross", 12),
    "hgp_hamming_d3_p001": (S1.format("hgp_hamming", 3, 3, 100, 0.001), "hgp_hamming", 3),
    "steane_d9_p001": (S1.format("steane", 9, 9, 97, 0.001), "steane", 9),
    "mhc36_d4_p001": (S2.format("mhc36", 4, 4, 56, 0.001), "mhc36", 4),
}


def circuit(tag):
    return stim.Circuit.from_file(CIRCUITS[tag][0])


def load_shots(tag, n=None, start=0):
    """Shots [start, n) of the circuit's saved shots (detection events, observable flips)"""
    z = np.load(os.path.join(DATA, f"{tag}.shots.npz"))
    num_dets, num_obs, total = int(z["num_dets"]), int(z["num_obs"]), int(z["shots"])
    n = total if n is None else min(n, total)
    dets = np.unpackbits(z["dets"][start:n], axis=1, count=num_dets, bitorder="little").astype(bool)
    obs = np.unpackbits(z["obs"][start:n], axis=1, count=num_obs, bitorder="little").astype(bool)
    return dets, obs


def cpu_predictor(tag, decoder):
    """Per-shot observable predictions of ECCentric's CPU decoder (same construction as decoders/*.py)"""
    circ = circuit(tag)
    code, dist = CIRCUITS[tag][1], CIRCUITS[tag][2]
    if decoder in ("bposd", "bposd_tuned"):
        from eccentric.decoders.bp_osd import _bposd_decoder
        bp = "tuned" if decoder == "bposd_tuned" else "paper"
        chk_variant = code in ("steane", "bacon")
        _, chk, obs, bpd = _bposd_decoder(circ, False, xor_priors=not chk_variant, osd_order=7 if chk_variant else None, bp=bp)
        return lambda d: (obs @ bpd.decode(d)) % 2
    if decoder in ("bplsd", "belief_find", "union_find"):
        from ldpc import BpLsdDecoder, BeliefFindDecoder, UnionFindDecoder
        from eccentric.decoders.ldpc_decoders import _check_matrices
        chk, obs, priors = _check_matrices(circ, False)
        if decoder == "bplsd":
            d = BpLsdDecoder(chk, error_channel=list(priors), max_iter=10000, bp_method="minimum_sum", ms_scaling_factor=1.0,
                             lsd_method="lsd_cs", lsd_order=min(7, chk.shape[1] - chk.shape[0]))
            return lambda s: (obs @ d.decode(s)) % 2
        if decoder == "union_find":
            d = UnionFindDecoder(chk, uf_method="inversion")
            pr = np.clip(priors, 1e-12, 1 - 1e-12); llrs = np.log((1 - pr) / pr)
            return lambda s: (obs @ d.decode(s, llrs=llrs)) % 2
        seen = np.asarray(chk.sum(axis=0)).ravel() > 0
        chk, obs, priors = chk[:, seen], obs[:, seen], np.asarray(priors)[seen]
        d = BeliefFindDecoder(chk, error_channel=list(priors), max_iter=10000, bp_method="minimum_sum", ms_scaling_factor=1.0, uf_method="inversion")
        return lambda s: (obs @ d.decode(s)) % 2
    if decoder in ("hier_bposd", "hier_bposd_hard"):
        from eccentric.decoders.hierarchical import Hierarchical, detector_levels
        h = Hierarchical(circ.detector_error_model(), detector_levels(circ, code, dist), soft=decoder == "hier_bposd")
        return lambda s: (h.obs @ h.decode(s)) % 2
    raise ValueError(decoder)


def cpu_batch_predictor(tag, decoder):
    """Decoders whose CPU version decodes a batch at once"""
    circ = circuit(tag)
    if decoder == "relay_bp":
        from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP
        compiled = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=circ.detector_error_model())
        m = compiled.check_matrices
        def run(dets):
            syn = dets.astype(np.uint8)
            if m.syndrome_bias is not None: syn = (syn + m.syndrome_bias) % 2
            p = compiled.observable_decoder.decode_observables_batch(syn, parallel=False)
            if m.observables_bias is not None: p = (p + m.observables_bias) % 2
            return p.astype(bool)
        return run
    if decoder == "gari":
        from eccentric.decoders.gari.gari import gari_encoding, decode_observables
        enc = gari_encoding(circ)
        return lambda dets: decode_observables(enc, dets)
    single = cpu_predictor(tag, decoder)
    return lambda dets: np.array([single(d) for d in dets.astype(np.uint8)]).astype(bool)


def gpu_predictor(tag, decoder):
    circ = circuit(tag)
    code, dist = CIRCUITS[tag][1], CIRCUITS[tag][2]
    if decoder in ("bposd", "bposd_tuned"):
        from eccentric.decoders.gpu.nvqldpc import bposd_decoder
        return bposd_decoder(circ, False, code in ("steane", "bacon"), "tuned" if decoder == "bposd_tuned" else "paper")
    if decoder == "relay_bp":
        from eccentric.decoders.gpu.nvqldpc import relay_decoder
        return relay_decoder(circ, False)
    if decoder == "bplsd":
        from eccentric.decoders.gpu.ldpc_gpu import bplsd_decoder
        return bplsd_decoder(circ, False)
    if decoder == "belief_find":
        from eccentric.decoders.gpu.ldpc_gpu import belief_find_decoder
        return belief_find_decoder(circ, False)
    if decoder == "union_find":
        from eccentric.decoders.gpu.union_find_gpu import union_find_decoder
        return union_find_decoder(circ, False)
    if decoder == "gari":
        from eccentric.decoders.gari.gari import gari_encoding
        from eccentric.decoders.gpu.gari_gpu import GariGpu
        return GariGpu(gari_encoding(circ))
    if decoder in ("hier_bposd", "hier_bposd_hard"):
        from eccentric.decoders.hierarchical import detector_levels
        from eccentric.decoders.gpu.ldpc_gpu import HierarchicalGpu
        return HierarchicalGpu(circ.detector_error_model(), detector_levels(circ, code, dist), soft=decoder == "hier_bposd")
    raise ValueError(decoder)
