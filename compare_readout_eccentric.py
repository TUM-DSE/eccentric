"""
compare_readout_eccentric.py
============================
ECCentric half of the "does readout length affect logical-CNOT LER, and does it
differ between ECCentric and the synchronization artifact?" study.

Surface-code lattice-surgery logical CNOT (tqec, d+1 rounds), swept over readout
(= measurement) length. Readout length here is the M-gate DURATION; its effect on
LER is via idle T1/T2 decoherence during the measurement window. The readout flip
error is a FIXED ibm_boston value (NOT swept), so the two tools share one readout
model and the comparison is apples-to-apples.

Noise (SOTA ibm_boston / Heron r3, from the ibm_boston dashboard):
  current    : CZ=1.191e-3, SX=1.637e-4, readout=3.54e-3, T1=284.95us, T2=322.68us
  futuristic : all errors / 10, T1/T2 * 3
  idle_multiplier = 1.0   (ECCentric: NO artifact inflation)
Gate times : SX=32ns, CZ/TQ=70ns (matches artifact CNOT latency), R=0, M=readout_ns.
Decoder    : MWPM. A shot fails if any of the 3 logical observables is wrong.

Output: experiment_results/readout_compare/eccentric_readout_ler.csv
        columns: tool,distance,readout_ns,noise_model,measure_error,logical_error_rate
"""

import sys, os, csv, itertools, time
import multiprocessing as mp

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))
sys.path.append(os.getcwd())

import numpy as np
import stim
import pymatching
from tqdm import tqdm
from qiskit.providers import QubitProperties

from backends import get_backend, QubitTracking
from noise import NoiseModel
from main import single_cnot_n_rounds

# ---------------------------------------------------------------------------
# Readout model
#   "cnn" : length-dependent flip error from the CNN table (readout
#                error and readout duration move together -- the realistic case).
#                Lengths capped at 1000 ns (CNN table range).
#   "fixed"    : flip error fixed at ibm_boston median 3.54e-3; length only sets
#                the M-gate duration (the "magically fast/clean readout" counterfactual).
# ---------------------------------------------------------------------------
READOUT_MODE = "cnn"

# CNN per-qubit accuracies (Q1,Q3,Q4,Q5; Q2 excluded); measure_error =
# 1 - geomean(.)  (from evaluate_mcm_latency.py).
CNN_ACC = {
    200:  (0.782, 0.864, 0.774, 0.770),   # CNN per-qubit accuracies (Q2 excluded)
    400:  (0.907, 0.926, 0.901, 0.951),
    600:  (0.953, 0.938, 0.937, 0.970),
    800:  (0.965, 0.941, 0.944, 0.970),
    1000: (0.970, 0.942, 0.947, 0.970),
}
def cnn_measure_error(readout_ns):
    q1, q3, q4, q5 = CNN_ACC[readout_ns]
    return 1.0 - (q1 * q3 * q4 * q5) ** 0.25

# ---------------------------------------------------------------------------
# Swept parameters
# ---------------------------------------------------------------------------
DISTANCES = [5, 7, 9, 11, 13]
READOUT_LENGTHS_NS = ([200, 400, 600, 800, 1000] if READOUT_MODE == "cnn"
                      else [250, 500, 750, 1000, 1500, 2000])
NOISE_MODELS = ["current", "futuristic"]

# ---------------------------------------------------------------------------
# SOTA ibm_boston (Heron r3) noise — current preset
# ---------------------------------------------------------------------------
CZ_ERROR = 1.191e-3
SX_ERROR = 1.637e-4
READOUT_ERROR = 3.54e-3        # ibm_boston median readout error (FIXED, not swept)
T1_S, T2_S = 284.95e-6, 322.68e-6
SX_LATENCY_NS = 32
CZ_LATENCY_NS = 70             # match artifact CNOT_LATENCY_NS
IDLE_MULTIPLIER = 1.0          # ECCentric: no artifact x3 inflation

# Futuristic preset
FUT_GATE_DIV = 10.0
FUT_MEAS_DIV = 10.0
FUT_T1T2_FAC = 3.0

BACKEND_NAME = "real_heron"
# Adaptive shots: the tqec CNOT decode cost grows steeply with d (d=13 ~ 0.08 s/shot)
# and high-d is saturated (few shots suffice), while low-d is cheap and informative.
SHOTS_BY_D = {5: 100_000, 7: 50_000, 9: 30_000, 11: 20_000, 13: 15_000}
CHUNK_SIZE = 10_000

# tqec logical-timestep "memory rounds". "dplus1" = d+1 = the original (long)
# experiment and the canonical plotted file; "1" = minimal fault-tolerant CNOT
# (~2d physical rounds), to match lattice-sim's physical depth (the nmr1 control).
# Override with env MEMORY_ROUNDS=dplus1 | 1.
MEMORY_ROUNDS = os.environ.get("MEMORY_ROUNDS", "dplus1")

OUTPUT_DIR = "experiment_results/readout_compare"
_base = "eccentric_readout_ler_cnn" if READOUT_MODE == "cnn" else "eccentric_readout_ler"
if MEMORY_ROUNDS != "dplus1":
    _base += f"_nmr{MEMORY_ROUNDS}"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, _base + ".csv")


def memory_rounds_for(d):
    return d + 1 if MEMORY_ROUNDS == "dplus1" else int(MEMORY_ROUNDS)


class FastUniformNoise(NoiseModel):
    """NoiseModel with uniform T1/T2 — memoize the T1/T2 channel by duration."""
    def __init__(self, *a, fixed_t1, fixed_t2, **k):
        super().__init__(*a, **k)
        self._t1, self._t2, self._cache = fixed_t1, fixed_t2, {}

    def _t1t2_pauli_channel(self, qubit_idx, duration, circuit):
        if duration <= 0 or self._t1 <= 0 or self._t2 <= 0:
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


_CIRCUIT_CACHE = {}
def _clean_circuit(d):
    key = (d, memory_rounds_for(d))
    c = _CIRCUIT_CACHE.get(key)
    if c is None:
        k = (d - 1) // 2
        c = single_cnot_n_rounds(None, distance_scale=k, num_memory_rounds=memory_rounds_for(d))
        _CIRCUIT_CACHE[key] = c
    return c


def _decode_chunked(circuit, num_shots, chunk_size=CHUNK_SIZE):
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    num_errors = 0
    remaining = num_shots
    while remaining > 0:
        n = min(chunk_size, remaining)
        det, obs = sampler.sample(n, separate_observables=True)
        pred = matcher.decode_batch(det)
        num_errors += int(np.any(pred != obs, axis=1).sum())
        remaining -= n
    return num_errors / num_shots


def _run_one(args):
    d, readout_ns, noise_model = args
    fut = (noise_model == "futuristic")
    base_meas = cnn_measure_error(readout_ns) if READOUT_MODE == "cnn" else READOUT_ERROR
    cz, sx, meas = CZ_ERROR, SX_ERROR, base_meas
    t1, t2 = T1_S, T2_S
    if fut:
        cz /= FUT_GATE_DIV; sx /= FUT_GATE_DIV; meas /= FUT_MEAS_DIV
        t1 *= FUT_T1T2_FAC; t2 *= FUT_T1T2_FAC

    circuit = _clean_circuit(d)
    backend = get_backend(BACKEND_NAME, None)
    qt = QubitTracking(backend, None)
    nm = FastUniformNoise(
        sq=sx, tq=cz, measure=meas, reset=0.0,
        gate_times={"SQ": SX_LATENCY_NS * 1e-9, "TQ": CZ_LATENCY_NS * 1e-9,
                    "M": readout_ns * 1e-9, "R": 0.0},
        qt=qt, backend=backend, idle_multiplier=IDLE_MULTIPLIER,
        fixed_t1=t1, fixed_t2=t2,
    )
    noisy = nm.noisy_circuit(circuit)
    shots = SHOTS_BY_D[d]
    t0 = time.time()
    try:
        ler = _decode_chunked(noisy, shots)
    except Exception as e:
        print(f"FAIL eccentric d={d} ro={readout_ns} {noise_model}: {e}", flush=True)
        return ("eccentric", d, readout_ns, noise_model, meas, None)
    print(f"done eccentric d={d:2} ro={readout_ns:4} {noise_model:10} "
          f"LER={ler:.6f} ({time.time()-t0:.1f}s, {shots} sh)", flush=True)
    return ("eccentric", d, readout_ns, noise_model, meas, ler)


def main():
    configs = list(itertools.product(DISTANCES, READOUT_LENGTHS_NS, NOISE_MODELS))
    print(f"ECCentric readout-length sweep — logical CNOT, ibm_boston, idle_mult={IDLE_MULTIPLIER}")
    print(f"  distances={DISTANCES} readout_ns={READOUT_LENGTHS_NS} shots={SHOTS_BY_D}")
    print(f"  configs={len(configs)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-build the clean tqec circuits in the MAIN process: tqec's detector
    # annotator spawns its own Pool, which is forbidden inside daemon workers.
    # Forked workers then inherit _CIRCUIT_CACHE and never call the builder.
    for d in DISTANCES:
        t = time.time()
        _clean_circuit(d)
        print(f"  built clean circuit d={d} ({time.time()-t:.1f}s)", flush=True)

    num_procs = max(1, min(mp.cpu_count() - 2, 32))

    results = []
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "distance", "readout_ns", "noise_model", "measure_error", "logical_error_rate"])
        f.flush()
        with mp.Pool(num_procs) as pool:
            for r in pool.imap_unordered(_run_one, configs):
                results.append(r); w.writerow(r); f.flush()

    results.sort(key=lambda x: (x[1], x[3], x[2]))
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "distance", "readout_ns", "noise_model", "measure_error", "logical_error_rate"])
        w.writerows(results)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
