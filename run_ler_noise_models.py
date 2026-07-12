"""
run_ler_noise_models.py
=======================
Logical error rate (LER) of six QEC codes under three Heron-based noise models,
swept over CNN readout lengths and code distances.

Codes        : surface, bacon (Bacon-Shor), color, hh (heavy-hex), gross, steane
Distances    : surface/bacon/color/hh -> 5,7,9,11 ; gross -> 12 (fixed
               [[144,12,12]] code) ; steane -> 9 (only d in {3,9,27} supported)
Noise models :
  1) "heron_cnn" -- SOTA ibm_boston Heron r3 medians (sq=1.637e-4,
     tq=1.191e-3) with the CNN readout error as the measurement-flip prob.
  2) "readout_decoherence" -- gate errors zeroed (sq=tq=reset=0); only CNN
     readout flip error + T1/T2 decoherence remain (readout still decoheres via
     its M-gate duration).
  3) "decoherence_only" -- everything zeroed except T1/T2 decoherence; readout
     adds no flip error but its M-gate duration still drives decoherence.
Readout      : CNN per-qubit accuracies (Q1,Q3,Q4,Q5; Q2 excluded),
               measure_error = 1 - geomean(.), swept over 200/400/600/800/1000 ns.
               The readout length sets BOTH the flip error (models 1,2) and the
               M-gate duration that drives readout decoherence (all models).

Backend      : FakeIBMHeron topology with SOTA ibm_boston coherence
               (T1=284.95us, T2=322.68us uniform), idle_multiplier=1.0.
Gate times   : SQ=32ns, TQ=68ns, R=0 (noiseless reset), M=readout_ns -- ibm_boston.
Gate errors  : ibm_boston (Heron r3) medians -- SX=1.637e-4, CZ=1.191e-3.
Decoders     : surface, hh -> MWPM ; bacon, color, steane, gross -> BP+OSD
               (osd_0, max_iter=100). gross is not matchable so MWPM is N/A.

Output       : experiment_results/noise_models/ler_noise_models.csv
"""

import sys, os, csv, time, itertools
import multiprocessing as mp

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))
sys.path.append(os.getcwd())

import numpy as np
import stim
import pymatching

from codes import get_code
from backends import get_backend, QubitTracking
from noise import NoiseModel
from ldpc import bposd_decoder
from decoders.bp_osd import dem_to_check_matrices

# ---------------------------------------------------------------------------
# CNN readout (per-qubit accuracies Q1,Q3,Q4,Q5; Q2 excluded)
# measure_error = 1 - geomean(Q1,Q3,Q4,Q5)            -- from evaluate_mcm_latency.py
# ---------------------------------------------------------------------------
CNN_ACC = {
    200:  (0.782, 0.864, 0.774, 0.770),   # CNN per-qubit accuracies (Q2 excluded)
    400:  (0.907, 0.926, 0.901, 0.951),
    600:  (0.953, 0.938, 0.937, 0.970),
    800:  (0.965, 0.941, 0.944, 0.970),
    1000: (0.970, 0.942, 0.947, 0.970),
}
READOUT_LENGTHS_NS = [200, 400, 600, 800, 1000]

def cnn_measure_error(readout_ns: int) -> float:
    q1, q3, q4, q5 = CNN_ACC[readout_ns]
    return 1.0 - (q1 * q3 * q4 * q5) ** 0.25

# ---------------------------------------------------------------------------
# ibm_boston (Heron r3) noise -- EXACTLY the ibm_boston values, EXCEPT the
# readout error/length which come from CNN (above). Sources:
#   - SX error, CZ error, T2  : ibm_boston dashboard image.
#   - T1, gate latencies, reset: documented ibm_boston in evaluate_mcm_latency.py
#     (the dashboard image does not list these): T1=284.95us, SX=32ns, CZ=68ns,
#     reset = noiseless & instantaneous (R=0).
# idle_multiplier=1.0 per request (the artifact-only x3 inflation is dropped).
# ---------------------------------------------------------------------------
HERON_SQ = 1.637e-4     # single-qubit (SX) error median, ibm_boston (image)
HERON_TQ = 1.191e-3     # two-qubit  (CZ) error median, ibm_boston (image)
SQ_NS, TQ_NS = 32, 68   # ibm_boston SX/CZ gate latencies (evaluate_mcm_latency.py)
R_S = 0.0               # ibm_boston reset: noiseless & instantaneous
T1_S, T2_S = 284.95e-6, 322.68e-6   # ibm_boston coherence (T2 from image, T1 from ref)
IDLE_MULTIPLIER = 1.0               # no artifact x3 inflation

NOISE_MODELS = ["heron_cnn", "readout_decoherence", "decoherence_only"]

# Which run to do (each writes its own CSV, leaving the others untouched):
#   "current"    -> the 3 standard models.
#   "futuristic" -> Model 1 with all errors /10 and T1/T2 x3.
#   "highfid"    -> Model 1 with ONLY the readout flip error /10 (gates & T1/T2 current).
#   "boston_lowt_backlog" -> ibm_boston gates, but T1/T2 = default-Heron 190/130us,
#       FIXED ibm_boston readout 3.54e-3 (NOT CNN), and decoder-backlog idle ON.
#   "futuristic_fixedreadout" -> futuristic gates (/10) + T1/T2 x3, BUT readout =
#       ibm_boston median /10 = 3.54e-4 FIXED (not CNN -> no 200ns cliff).
MODE = os.environ.get("LER_MODE", "current")
_MODE_MODELS = {"current": NOISE_MODELS,
                "futuristic": ["heron_cnn_futuristic"],
                "highfid": ["heron_highfid"],
                "boston_lowt_backlog": ["boston_lowt"],
                "futuristic_fixedreadout": ["futuristic_fixedreadout"]}
_MODE_SUFFIX = {"current": "", "futuristic": "_futuristic", "highfid": "_highfid",
                "boston_lowt_backlog": "_boston_lowt_backlog",
                "futuristic_fixedreadout": "_futuristic_fixedreadout"}
RUN_MODELS = _MODE_MODELS[MODE]
BACKLOG = (MODE == "boston_lowt_backlog")

# Default-Heron coherence (FakeIBMHeron) and the ibm_boston fixed readout median.
HERON_T1_LOW, HERON_T2_LOW = 190e-6, 130e-6
BOSTON_READOUT = 3.54e-3

# ---------------------------------------------------------------------------
# Config matrix
# ---------------------------------------------------------------------------
CODE_DISTANCES = {
    "surface": [5, 7, 9, 11, 13],
    "bacon":   [5, 7, 9, 11, 13],
    "color":   [5, 7, 9, 11, 13],
    "hh":      [5, 7, 9, 11, 13],
    "gross":   [12],   # fixed [[144,12,12]] code; d argument ignored (no d=13)
    "steane":  [9],    # concat steane supports d in {3,9,27} only (no d=13)
}
DECODER = {
    "surface": "mwpm", "hh": "mwpm",
    "bacon": "bposd", "color": "bposd", "steane": "bposd", "gross": "bposd",
}

# Reduced-error runs have ~10x lower LER, so bump MWPM shots (cheap) for resolution.
MWPM_SHOTS = 50_000 if MODE != "current" else 20_000
BPOSD_SHOTS = 5_000
# gross ([[144,12,12]], 288 qubits, 12 observables) BP-OSD is ~1.5 s/shot --
# far slower than the other BP-OSD codes -- so it gets a smaller shot budget.
GROSS_SHOTS = 1_000

OUTPUT_DIR = "experiment_results/noise_models"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"ler_noise_models{_MODE_SUFFIX[MODE]}.csv")

# ---------------------------------------------------------------------------
# Fast uniform-coherence noise model (memoize the T1/T2 channel by duration).
# Numerically identical to NoiseModel for uniform T1/T2, ~order faster.
# ---------------------------------------------------------------------------
class FastUniformNoise(NoiseModel):
    def __init__(self, *a, fixed_t1, fixed_t2, **k):
        super().__init__(*a, **k)
        self._t1, self._t2, self._cache = fixed_t1, fixed_t2, {}

    def _t1t2_pauli_channel(self, qubit_idx, duration, circuit):
        if duration <= 0:
            return
        key = round(duration * self.idle_multiplier, 15)
        p = self._cache.get(key)
        if p is None:
            te = duration * self.idle_multiplier
            px = 0.25 * (1 - np.exp(-te / self._t1)); py = px
            pz = (1 - np.exp(-te / self._t2)) / 2 - (1 - np.exp(-te / self._t1)) / 4
            p = (float(np.clip(px, 0, 1)), float(np.clip(py, 0, 1)), float(np.clip(pz, 0, 1)))
            self._cache[key] = p
        circuit.append_operation("PAULI_CHANNEL_1", [qubit_idx], list(p))


def build_noisy_circuit(code_name, d, model, readout_ns, backend):
    code = get_code(code_name, d, d if code_name not in ("gross",) else None)
    qt = QubitTracking(backend, getattr(code, "qc", None))
    measure = cnn_measure_error(readout_ns)
    t1, t2 = T1_S, T2_S
    if model == "heron_cnn":
        sq, tq, meas = HERON_SQ, HERON_TQ, measure
    elif model == "readout_decoherence":
        sq, tq, meas = 0.0, 0.0, measure
    elif model == "decoherence_only":
        sq, tq, meas = 0.0, 0.0, 0.0
    elif model == "heron_cnn_futuristic":
        # Futuristic: all errors / 10 (gates AND readout flip), T1/T2 x3.
        sq, tq, meas = HERON_SQ / 10.0, HERON_TQ / 10.0, measure / 10.0
        t1, t2 = T1_S * 3.0, T2_S * 3.0
    elif model == "heron_highfid":
        # 10x better readout fidelity only; gates and coherence stay ibm_boston.
        sq, tq, meas = HERON_SQ, HERON_TQ, measure / 10.0
    elif model == "boston_lowt":
        # ibm_boston gates, default-Heron coherence (190/130us), FIXED ibm_boston
        # readout 3.54e-3 (no CNN, no U-shape).
        sq, tq, meas = HERON_SQ, HERON_TQ, BOSTON_READOUT
        t1, t2 = HERON_T1_LOW, HERON_T2_LOW
    elif model == "futuristic_fixedreadout":
        # Futuristic gates/coherence (errors/10, T1/T2 x3) with the ibm_boston
        # median readout ALSO /10 = 3.54e-4, FIXED (no CNN 200ns cliff).
        sq, tq, meas = HERON_SQ / 10.0, HERON_TQ / 10.0, BOSTON_READOUT / 10.0
        t1, t2 = T1_S * 3.0, T2_S * 3.0
    else:
        raise ValueError(model)
    nm = FastUniformNoise(
        sq=sq, tq=tq, measure=meas, reset=0.0,
        gate_times={"SQ": SQ_NS * 1e-9, "TQ": TQ_NS * 1e-9,
                    "M": readout_ns * 1e-9, "R": R_S},
        qt=qt, backend=backend, idle_multiplier=IDLE_MULTIPLIER,
        fixed_t1=t1, fixed_t2=t2,
    )
    circ = nm.noisy_circuit(code.stim_circuit)
    if BACKLOG:
        rounds = 12 if code_name == "gross" else d
        circ = _inject_backlog(circ, nm, rounds * readout_ns * 1e-9)
    return circ


def _inject_backlog(circ, nm, t_idle):
    """Insert decoder-backlog idle (T1/T2 on ALL qubits) just before the final
    measurement tail, sized t_idle. Robust to repeat-block and unrolled circuits."""
    delay = nm.get_idle_delay_circuit(t_idle, circ.num_qubits)
    if len(delay) == 0:
        return circ
    SKIP = {"M", "MR", "MX", "MZ", "MY", "MRX", "MRY", "MRZ", "MPP",
            "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK", "QUBIT_COORDS"}
    split = len(circ)
    for i in range(len(circ) - 1, -1, -1):
        op = circ[i]
        if isinstance(op, stim.CircuitRepeatBlock):
            split = i + 1
            break
        if isinstance(op, stim.CircuitInstruction) and op.name in SKIP:
            split = i
            continue
        break
    return circ[:split] + delay + circ[split:]


def decode_mwpm(circuit, num_shots):
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    det, obs = sampler.sample(num_shots, separate_observables=True)
    pred = matcher.decode_batch(det)
    return int(np.any(pred != obs, axis=1).sum()) / num_shots


def decode_bposd(circuit, num_shots):
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    chk, obs, priors, _ = dem_to_check_matrices(dem, return_col_dict=True)
    bpd = bposd_decoder(chk, channel_probs=list(priors), max_iter=100,
                        bp_method="minimum_sum", ms_scaling_factor=1.0,
                        osd_method="osd_0", osd_order=0, input_vector_type="syndrome")
    ds = dem.compile_sampler()
    det, obs_data, _ = ds.sample(shots=num_shots, return_errors=False, bit_packed=False)
    num_err = 0
    for i in range(num_shots):
        e_hat = bpd.decode(det[i])
        num_err += ((obs @ e_hat + obs_data[i]) % 2).any()
    return num_err / num_shots


def run_one(args):
    code_name, d, model, readout_ns = args
    t0 = time.time()
    backend = get_backend("real_heron", None)
    try:
        circuit = build_noisy_circuit(code_name, d, model, readout_ns, backend)
        decoder = DECODER[code_name]
        if decoder == "mwpm":
            ler = decode_mwpm(circuit, MWPM_SHOTS)
            shots = MWPM_SHOTS
        else:
            shots = GROSS_SHOTS if code_name == "gross" else BPOSD_SHOTS
            ler = decode_bposd(circuit, shots)
    except Exception as e:
        print(f"FAIL {code_name} d={d} {model} ro={readout_ns}: {type(e).__name__}: {e}", flush=True)
        return (code_name, d, model, readout_ns, DECODER[code_name], None, None, None, time.time() - t0)
    dt = time.time() - t0
    print(f"done {code_name:8} d={d:2} {model:20} ro={readout_ns:4} "
          f"LER={ler:.5f} ({dt:.1f}s)", flush=True)
    return (code_name, d, model, readout_ns, DECODER[code_name],
            cnn_measure_error(readout_ns), shots, ler, dt)


def main():
    configs = []
    for code_name, dists in CODE_DISTANCES.items():
        for d in dists:
            for model in RUN_MODELS:
                for ro in READOUT_LENGTHS_NS:
                    configs.append((code_name, d, model, ro))

    print(f"Total configs: {len(configs)}")
    print(f"  MWPM shots: {MWPM_SHOTS}, BP-OSD shots: {BPOSD_SHOTS}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    header = ["code", "distance", "noise_model", "readout_ns", "decoder",
              "measure_error", "num_shots", "logical_error_rate", "seconds"]
    num_procs = max(1, min(mp.cpu_count() - 2, 60))

    results = []
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()
        with mp.Pool(num_procs) as pool:
            for res in pool.imap_unordered(run_one, configs):
                results.append(res)
                writer.writerow(res)
                f.flush()

    results.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            writer.writerow(r)

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
