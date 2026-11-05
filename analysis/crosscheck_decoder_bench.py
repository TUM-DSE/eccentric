"""Decode decoder-bench's circuits (Maurya et al., IISWC 2025) with ECCentric's decoders and with
decoder-bench's decoder settings, on the same shots:
python analysis/crosscheck_decoder_bench.py <decoder-bench checkout> <processes> <out.csv>

The circuits come from decoder-bench's own generators with its SI1000 noise (surface and color code
d = 7, bivariate bicycle d = 6 and 12, d rounds, Z basis). The shots are sampled once per circuit; every
decoder decodes the same ones. decoder-bench's decoders are rebuilt here with its settings
(decoder_bench/decoders.py), so its h5py/tqdm sampler is not needed. Besides the LER over all shots,
the output gives the LER over the first shot of each distinct syndrome, which is what a trace of
unique syndromes measures.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import stim

CIRCUITS = [("surface", 7, 0.003), ("surface", 7, 0.005), ("color", 7, 0.003), ("color", 7, 0.005),
            ("bb", 6, 0.003), ("bb", 6, 0.005), ("bb", 12, 0.003)]
SHOTS = {6: 20_000, 7: 20_000, 12: 4_000}
CHUNK = {6: 1_000, 7: 1_000, 12: 200}
OURS = {"surface": ["mwpm", "bposd", "relay_bp", "belief_find", "bplsd", "tesseract"],
        "color": ["bposd", "relay_bp", "belief_find", "bplsd", "tesseract"],
        "bb": ["bposd", "relay_bp", "belief_find", "bplsd", "tesseract"]}
THEIRS = {"surface": ["pymatching", "belieffind", "bplsd", "relay-bp"],
          "color": ["belieffind", "bplsd", "relay-bp"], "bb": ["belieffind", "bplsd", "relay-bp"]}
STATE = {}


def circuit_of(checkout, code, d, p):
    sys.path[:0] = [os.path.join(checkout, "decoder_bench"), os.path.join(checkout, "decoder_bench", "common")]
    from common.noise import NoiseModel
    if code == "surface":
        circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=d, distance=d)
    elif code == "color":
        from common.color_code_gen import ColorCode
        circuit = ColorCode(d=d, rounds=d, p_bitflip=p).circuit.without_noise()
    else:
        from common.build_circuit import build_circuit
        from common.codes_q import create_bivariate_bicycle_codes
        l = {6: 6, 12: 12}[d]
        bb, a, b = create_bivariate_bicycle_codes(l, 6, [3], [1, 2], [1, 2], [3])
        circuit = build_circuit(bb, a, b, p=0, num_repeat=d, use_both=False, z_basis=True).without_noise()
    return NoiseModel.SI1000(p).noisy_circuit(circuit)


class Replay:
    """Stands in for stim's detector sampler: hands out the pre-sampled shots"""

    def __init__(self, dets, obs):
        self.dets, self.obs = dets, obs

    def sample(self, shots, separate_observables=True, bit_packed=False, **kwargs):
        d, o = self.dets[:shots], self.obs[:shots]
        if bit_packed:
            return np.packbits(d, axis=1, bitorder="little"), np.packbits(o, axis=1, bitorder="little")
        return d, o


def theirs(name, circuit, dets):
    """decoder-bench's settings (decoder_bench/decoders.py): predicted observables per shot"""
    from scipy.sparse import csr_matrix
    from common.build_circuit import dem_to_check_matrices
    if name == "pymatching":
        import pymatching
        return pymatching.Matching.from_stim_circuit(circuit=circuit).decode_batch(dets), {}
    chk, obs, priors = dem_to_check_matrices(circuit.detector_error_model())
    info = {}
    if name == "belieffind":
        from ldpc.belief_find_decoder import BeliefFindDecoder
        dec = BeliefFindDecoder(chk, error_channel=list(priors), max_iter=100, bp_method="minimum_sum",
                                omp_thread_count=1, uf_method="inversion")
    elif name == "bplsd":
        from ldpc.bplsd_decoder import BpLsdDecoder
        dec = BpLsdDecoder(chk, error_channel=list(priors), max_iter=100, bp_method="minimum_sum",
                           osd_method="osd_cs", osd_order=10, omp_thread_count=1)
        info = {"lsd_method": str(getattr(dec, "lsd_method", "?")), "lsd_order": getattr(dec, "lsd_order", "?")}
    else:
        import relay_bp
        dec = relay_bp.RelayDecoderF32(csr_matrix(chk), error_priors=priors, gamma0=0.1, pre_iter=80, num_sets=60,
                                       set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stop_nconv=3)
        return np.array([(obs @ dec.decode(s.astype(np.uint8))) % 2 for s in dets]), info
    return np.array([(obs @ dec.decode(s)) % 2 for s in dets]), info


def task(args):
    key, side, decoder, start, stop = args
    circuit, dets, obs = STATE[key]
    d, o = dets[start:stop], obs[start:stop]
    t0, c0 = time.perf_counter(), time.process_time()
    if side == "ours":
        from eccentric.decoders import decode
        original = stim.Circuit.compile_detector_sampler
        stim.Circuit.compile_detector_sampler = lambda self, *a, **k: Replay(d, o)
        try:
            ler = decode(key[0], circuit, len(d), decoder, "custom_full", "sd6")
        finally:
            stim.Circuit.compile_detector_sampler = original
        wrong, info = None, {}
        errors = None if ler is None else int(round(ler * len(d)))
    else:
        pred, info = theirs(decoder, circuit, d)
        wrong = np.any(pred.astype(np.uint8) != o, axis=1)
        errors = int(wrong.sum())
    return dict(code=key[0], d=key[1], p=key[2], side=side, decoder=decoder, start=start, shots=len(d), errors=errors,
                wall=time.perf_counter() - t0, cpu=time.process_time() - c0, wrong=wrong, **info)


def init(checkout):
    sys.path[:0] = [os.path.join(checkout, "decoder_bench"), os.path.join(checkout, "decoder_bench", "common")]
    for code, d, p in CIRCUITS:
        circuit = circuit_of(checkout, code, d, p)
        dets, obs = circuit.compile_detector_sampler(seed=2025 + d).sample(SHOTS[d], separate_observables=True)
        STATE[(code, d, p)] = (circuit, dets, obs)


def main():
    checkout, processes, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tasks = []
    for code, d, p in CIRCUITS:
        for side, names in (("ours", OURS[code]), ("theirs", THEIRS[code])):
            for name in names:
                if name == "tesseract" and d == 12:
                    continue
                tasks += [((code, d, p), side, name, s, min(s + CHUNK[d], SHOTS[d])) for s in range(0, SHOTS[d], CHUNK[d])]
    # Long chunks first, so the pool does not wait on a straggler at the end
    tasks.sort(key=lambda t: (t[0][1] != 12, t[1] == "theirs"))
    with Pool(processes, initializer=init, initargs=(checkout,)) as pool:
        rows = pool.map(task, tasks, chunksize=1)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "wrong"} for r in rows])
    df.to_csv(out.replace(".csv", "_chunks.csv"), index=False)
    agg = df.groupby(["code", "d", "p", "side", "decoder"]).agg(shots=("shots", "sum"), errors=("errors", "sum"),
                                                                 cpu=("cpu", "sum")).reset_index()
    agg["ler"] = agg.errors / agg.shots
    agg["ms_per_shot"] = 1000 * agg.cpu / agg.shots
    info = [c for c in ("lsd_method", "lsd_order") if c in df]
    if info:
        agg = agg.merge(df.groupby(["code", "d", "p", "side", "decoder"])[info].first().reset_index(),
                        on=["code", "d", "p", "side", "decoder"], how="left")
    # LER over unique syndromes (decoder-bench's traces) against all shots, for decoder-bench's decoders
    init(checkout)
    unique = []
    for r in rows:
        if r["wrong"] is None:
            continue
        unique.append(r)
    for (code, d, p, decoder), group in pd.DataFrame(unique).groupby(["code", "d", "p", "decoder"]):
        wrong = np.concatenate([w for w in group.sort_values("start").wrong])
        dets = STATE[(code, d, p)][1][:len(wrong)]
        _, first = np.unique(np.packbits(dets, axis=1), axis=0, return_index=True)
        agg.loc[(agg.code == code) & (agg.d == d) & (agg.p == p) & (agg.side == "theirs") & (agg.decoder == decoder),
                ["unique_syndromes", "ler_unique"]] = [len(first), wrong[first].mean()]
    agg.to_csv(out, index=False)
    print(agg.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
