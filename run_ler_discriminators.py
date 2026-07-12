"""
run_ler_discriminators.py
=========================
How do different readout DISCRIMINATORS affect the logical error rate (LER)?

Same noise model as the `heron_cnn` model in run_ler_noise_models.py -- copy-pasted
ibm_boston (Heron r3) gates + coherence -- EXCEPT the measurement-flip probability,
which is taken from each readout discriminator in ../oraqle_reports/master_fidelity.csv.

Lines in the figure are DISCRIMINATORS (not distances). We sweep:
  codes        : surface, bacon, color, hh, gross, steane
  readout (ns) : 200, 400, 600, 800, 1000  (sets both the flip error AND the M-gate
                 duration that drives readout decoherence)
  distance     : 5, then 7   (run twice via LER_DIST env, default 5)
                 gross is the fixed [[144,12,12]] code (d=12, ignores the request);
                 steane only supports d in {3,9,27} so it is pinned to d=9.

Discriminators (from master_fidelity.csv): Linear Threshold, Baseline FNN, HERQULES,
QubiCML, KLiNQ, MCMit-T.  MCMit-CNN ("CNN") is SKIPPED here -- those LERs already
exist as the `heron_cnn` model in ler_noise_models.csv and are merged in at plot time.

Readout flip error = 1 - geomean(Q1, Q3, Q4, Q5)   (Q2 EXCLUDED; NOT the F5Q_gmean
column, which is the gmean5 trap).

Output: experiment_results/discriminators/ler_discriminators_d{5,7}.csv
"""

import sys, os, csv, time
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
# Discriminator readout fidelities (master_fidelity.csv). measure_error =
# 1 - geomean(Q1, Q3, Q4, Q5); Q2 excluded.  CNN is skipped (reuse existing).
# ---------------------------------------------------------------------------
MASTER_CSV = "../oraqle_reports/master_fidelity.csv"
SKIP_DISCRIMINATORS = {"MCMit-CNN"}     # CNN already computed (heron_cnn model)
READOUT_LENGTHS_NS = [200, 400, 600, 800, 1000]

# The REAL CNN ("MCMit-CNN") readout table = the 0.910-gmean5 table (NOT the stale
# master_fidelity MCMit-CNN row). Identical to run_ler_noise_models.py CNN_ACC and to
# the 0.910 table in synchronization-artifact/run_tradeoff_final.py (1000ns tuple
# matches exactly), sampled at 200-1000ns. Used only when LER_INCLUDE_CNN=1 -- i.e. at
# d=12, where heron_cnn has no data to reuse. (Q1,Q3,Q4,Q5; Q2 excluded.)
CNN_ACC_0910 = {
    200:  (0.782, 0.864, 0.774, 0.770),
    400:  (0.907, 0.926, 0.901, 0.951),
    600:  (0.953, 0.938, 0.937, 0.970),
    800:  (0.965, 0.941, 0.944, 0.970),
    1000: (0.970, 0.942, 0.947, 0.970),
}
INCLUDE_CNN = os.environ.get("LER_INCLUDE_CNN", "") == "1"


def load_discriminators(path):
    """-> {model: {duration_ns: (Q1, Q3, Q4, Q5)}}, preserving CSV row order."""
    table, order = {}, []
    with open(path) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model in SKIP_DISCRIMINATORS:
                continue
            if model not in table:
                table[model] = {}
                order.append(model)
            dur = int(row["duration_ns"])
            table[model][dur] = (float(row["Q1_fidelity"]), float(row["Q3_fidelity"]),
                                 float(row["Q4_fidelity"]), float(row["Q5_fidelity"]))
    return table, order


DISC_TABLE, DISC_ORDER = load_discriminators(MASTER_CSV)
if INCLUDE_CNN:
    DISC_TABLE["MCMit-CNN"] = dict(CNN_ACC_0910)   # 0.910 table, not master_fidelity
    DISC_ORDER.append("MCMit-CNN")


def measure_error(discriminator, readout_ns):
    q1, q3, q4, q5 = DISC_TABLE[discriminator][readout_ns]
    return 1.0 - (q1 * q3 * q4 * q5) ** 0.25


# ---------------------------------------------------------------------------
# ibm_boston (Heron r3) noise -- EXACTLY the heron_cnn model in
# run_ler_noise_models.py, only the readout flip comes from the discriminator.
# ---------------------------------------------------------------------------
HERON_SQ = 1.637e-4     # single-qubit (SX) error median, ibm_boston
HERON_TQ = 1.191e-3     # two-qubit  (CZ) error median, ibm_boston
SQ_NS, TQ_NS = 32, 68   # ibm_boston SX/CZ gate latencies
R_S = 0.0               # ibm_boston reset: noiseless & instantaneous
T1_S, T2_S = 284.95e-6, 322.68e-6   # ibm_boston coherence
IDLE_MULTIPLIER = 1.0

# Requested distance for the tunable codes; gross/steane override below.
DIST = int(os.environ.get("LER_DIST", "5"))

# Futuristic noise (LER_FUTURISTIC=1): every error source /10 (gates AND the
# discriminator readout flip), T1/T2 x3 -- same definition as the heron_cnn_futuristic
# model in run_ler_noise_models.py. Readout-flip is still per-discriminator (then /10).
FUTURISTIC = os.environ.get("LER_FUTURISTIC", "") == "1"
ERR_SCALE = 0.1 if FUTURISTIC else 1.0
T_SCALE = 3.0 if FUTURISTIC else 1.0

CODES = ["surface", "bacon", "color", "hh", "gross", "steane"]
# Restrict which codes to run (comma-separated) e.g. LER_CODES="surface,bacon,color,hh"
# -- used for the d=9/d=12 extensions where gross(d=12)/steane(d=9) are reused, not rerun.
_codes_env = os.environ.get("LER_CODES", "").strip()
if _codes_env:
    CODES = [c for c in _codes_env.split(",") if c]
DECODER = {
    "surface": "mwpm", "hh": "mwpm",
    "bacon": "bposd", "color": "bposd", "steane": "bposd", "gross": "bposd",
}


def code_distance(code_name):
    """Actual achievable distance for this code at the requested DIST."""
    if code_name == "gross":
        return 12          # fixed [[144,12,12]] code
    if code_name == "steane":
        return 9           # concat steane supports d in {3,9,27}; pin to 9
    return DIST


# Futuristic LERs are lower -> bump MWPM shots (cheap) for resolution. BP-OSD codes
# (bacon/gross) stay at the standard budget -- their LERs aren't ultra-low and BP-OSD
# is ~0.08-1.5 s/shot, so 10k shots is prohibitively slow for little gain.
MWPM_SHOTS = 100_000 if FUTURISTIC else 20_000
BPOSD_SHOTS = 5_000
GROSS_SHOTS = 1_000

OUTPUT_DIR = "experiment_results/discriminators"
_SUFFIX = "_futuristic" if FUTURISTIC else ""
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"ler_discriminators_d{DIST}{_SUFFIX}.csv")


# ---------------------------------------------------------------------------
# Fast uniform-coherence noise model (memoize the T1/T2 channel by duration).
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


def build_noisy_circuit(code_name, d, discriminator, readout_ns, backend):
    cycles = None if code_name == "gross" else d
    code = get_code(code_name, d, cycles)
    qt = QubitTracking(backend, getattr(code, "qc", None))
    meas = measure_error(discriminator, readout_ns) * ERR_SCALE
    nm = FastUniformNoise(
        sq=HERON_SQ * ERR_SCALE, tq=HERON_TQ * ERR_SCALE, measure=meas, reset=0.0,
        gate_times={"SQ": SQ_NS * 1e-9, "TQ": TQ_NS * 1e-9,
                    "M": readout_ns * 1e-9, "R": R_S},
        qt=qt, backend=backend, idle_multiplier=IDLE_MULTIPLIER,
        fixed_t1=T1_S * T_SCALE, fixed_t2=T2_S * T_SCALE,
    )
    return nm.noisy_circuit(code.stim_circuit)


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
    code_name, d, discriminator, readout_ns = args
    t0 = time.time()
    backend = get_backend("real_heron", None)
    try:
        circuit = build_noisy_circuit(code_name, d, discriminator, readout_ns, backend)
        if DECODER[code_name] == "mwpm":
            shots = MWPM_SHOTS
            ler = decode_mwpm(circuit, shots)
        else:
            shots = GROSS_SHOTS if code_name == "gross" else BPOSD_SHOTS
            ler = decode_bposd(circuit, shots)
    except Exception as e:
        print(f"FAIL {code_name} d={d} {discriminator} ro={readout_ns}: "
              f"{type(e).__name__}: {e}", flush=True)
        return (code_name, d, discriminator, readout_ns, DECODER[code_name],
                measure_error(discriminator, readout_ns) * ERR_SCALE, None, None, time.time() - t0)
    dt = time.time() - t0
    print(f"done {code_name:8} d={d:2} {discriminator:18} ro={readout_ns:4} "
          f"LER={ler:.5f} ({dt:.1f}s)", flush=True)
    return (code_name, d, discriminator, readout_ns, DECODER[code_name],
            measure_error(discriminator, readout_ns) * ERR_SCALE, shots, ler, dt)


def main():
    configs = []
    for code_name in CODES:
        d = code_distance(code_name)
        for discriminator in DISC_ORDER:
            for ro in READOUT_LENGTHS_NS:
                configs.append((code_name, d, discriminator, ro))

    print(f"Requested distance: {DIST}  (gross->12, steane->9)")
    print(f"Discriminators: {DISC_ORDER}")
    print(f"Total configs: {len(configs)}  (MWPM={MWPM_SHOTS}, BP-OSD={BPOSD_SHOTS}, gross={GROSS_SHOTS})")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    header = ["code", "distance", "discriminator", "readout_ns", "decoder",
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

    results.sort(key=lambda r: (r[0], r[2], r[3]))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            writer.writerow(r)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
