"""CPU worker: for each circuit of the slice (in order, claimed with mkdir), build the decoders' inputs (error
model, check matrices, relay-bp's matrices, GARI's encoding) and pickle them for the GPU runner; stays at most
AHEAD circuits ahead of the runner"""
import json, os, pickle, sys, time, traceback
import numpy as np, scipy.sparse as sp, stim

HERE = "/home/manosgior/eccentric-validation/gpu_dev/mm_pilot"
AHEAD = 10
slice_ = json.load(open(f"{HERE}/slice.json"))
os.makedirs(f"{HERE}/prep", exist_ok=True); os.makedirs(f"{HERE}/claims", exist_ok=True)


def approx(o):
    # decode.decode's rule
    return o["backend"].split("_")[0] in ("variance", "real", "native") and (o["error_type"] == "variance" or o["error_type"].split("_")[0] == "real")


for i, o in enumerate(slice_):
    try:
        os.mkdir(f"{HERE}/claims/{i}")
    except FileExistsError:
        continue
    while True:  # wait until the runner is close enough
        done = len(os.listdir(f"{HERE}/results")) if os.path.isdir(f"{HERE}/results") else 0
        if i < done + AHEAD:
            break
        time.sleep(20)
    out = {"index": i, "slice": o, "approx": approx(o), "prep_seconds": {}, "errors": {}}
    t0 = time.time()
    try:
        circ = stim.Circuit.from_file(o["file"])
        from eccentric.decoders.bp_osd import dem_to_check_matrices
        t = time.time()
        dem = circ.detector_error_model(approximate_disjoint_errors=out["approx"])
        out["prep_seconds"]["dem"] = time.time() - t
        decs = o["decoders"]
        if {"bposd", "bposd_tuned", "bplsd", "belief_find"} & set(decs):
            t = time.time()
            out["xor"] = dem_to_check_matrices(dem, xor_priors=True)
            if o["code"] in ("steane", "bacon") and {"bposd", "bposd_tuned"} & set(decs):
                out["sum"] = dem_to_check_matrices(dem, xor_priors=False)
            out["prep_seconds"]["check_matrices"] = time.time() - t
        if "relay_bp" in decs:
            t = time.time()
            try:
                from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP
                m = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=dem).check_matrices
                out["relay"] = (sp.csr_matrix(m.check_matrix), np.asarray(m.error_priors), sp.csr_matrix(m.observables_matrix),
                                None if m.syndrome_bias is None else np.asarray(m.syndrome_bias),
                                None if m.observables_bias is None else np.asarray(m.observables_bias))
            except Exception as e:
                out["errors"]["relay_bp"] = repr(e)
            out["prep_seconds"]["relay"] = time.time() - t
        if "gari" in decs:
            t = time.time()
            try:
                from eccentric.decoders.gari.gari import gari_encoding
                out["gari"] = gari_encoding(circ, out["approx"])
            except Exception as e:
                out["errors"]["gari"] = repr(e)
            out["prep_seconds"]["gari"] = time.time() - t
    except Exception:
        out["fatal"] = traceback.format_exc()
    out["prep_seconds"]["total"] = time.time() - t0
    tmp = f"{HERE}/prep/{i}.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, f"{HERE}/prep/{i}.pkl")
    print(i, o["key"], f"{out['prep_seconds']['total']:.1f}s", "FATAL" if "fatal" in out else "", flush=True)
