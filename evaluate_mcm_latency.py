"""
evaluate_mcm_latency.py
=======================
Logical error rate (LER) of a lattice-surgery logical CNOT between two
surface-code patches, as a function of MCM readout length, for two MCM decoder
systems (HERQULES vs MCMit-CNN), on a realistic IBM Heron r3 noise model.

ECCentric analogue of the synchronization artifact's evaluate_mcm_latency.py.

Operation
---------
A lattice-surgery logical CNOT built with the tqec library (main.single_cnot_
n_rounds): control patch + target patch + ancilla merge column, then d + 1
syndrome rounds. tqec distance d = 2k + 1, so k = (d - 1) / 2. MWPM scores a
shot as a failure if any of the 3 logical observables (control / target / joint)
is mispredicted -- i.e. "the logical CNOT failed".

Baseline QPU noise -- IBM Heron r3 (ibm_boston dashboard)
---------------------------------------------------------
- 2-qubit (CZ) error  : 1.13e-3 median        (futuristic: /10)
- 1-qubit (SX) error  : 1.569e-4 median        (futuristic: /10)
- T1 / T2             : 284.95 / 337.29 us      (futuristic: x3)
- CZ / SX latency     : 68 / 32 ns
- readout latency     : swept (M gate time)
- reset               : noiseless, instantaneous
- idle scaling        : idle_multiplier = 3 (IBM crosstalk/thermal)

MCM measurement error
---------------------
Per-readout-length measurement error comes from each system's per-qubit readout
accuracies, excluding the problematic qubit Q2:
    measure_error = 1 - geomean(Q1, Q3, Q4, Q5)
(futuristic: /10). These tables are exactly what the experiment compares.

Output
------
CSV experiment_results/benchmark/mcm_latency_ler.csv with columns:
    distance, readout_ns, system, noise_model, measure_error, logical_error_rate
plus per-(distance, noise_model) plots and a stdout summary.
"""

import sys
import os
import csv
import itertools
import multiprocessing as mp
import numpy as np
import stim
import pymatching
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit.providers import QubitProperties

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))
sys.path.append(os.getcwd())

from backends import get_backend, QubitTracking
from noise import NoiseModel
from main import single_cnot_n_rounds  # tqec lattice-surgery logical CNOT builder

# ---------------------------------------------------------------------------
# Swept parameters
# ---------------------------------------------------------------------------

DISTANCES = [5, 7, 9, 11]
READOUT_LENGTHS_NS = [200, 400, 600, 800, 1000]
SYSTEMS = ["HERQULES"]
NOISE_MODELS = ["current", "futuristic"]

# Per-qubit readout accuracies (Q1, Q3, Q4, Q5) -- Q2 excluded (problematic).
# Measurement error = 1 - geomean(Q1, Q3, Q4, Q5).
SYSTEM_QUBIT_ACCURACIES = {
    "HERQULES": {
        200:  (0.7782, 0.8712, 0.7587, 0.7159),
        400:  (0.9340, 0.9473, 0.9445, 0.9717),
        600:  (0.9654, 0.9565, 0.9597, 0.9827),
        800:  (0.9725, 0.9547, 0.9587, 0.9797),
        1000: (0.9685, 0.9472, 0.9532, 0.9815),
    },
    "MCMit-CNN": {
        200:  (0.7819, 0.8637, 0.7739, 0.7700),
        400:  (0.9069, 0.9262, 0.9014, 0.9514),
        600:  (0.9527, 0.9384, 0.9368, 0.9700),
        800:  (0.9648, 0.9407, 0.9441, 0.9703),
        1000: (0.9701, 0.9416, 0.9467, 0.9704),
    },
}


def measure_error_for(system: str, readout_ns: int) -> float:
    q1, q3, q4, q5 = SYSTEM_QUBIT_ACCURACIES[system][readout_ns]
    return 1.0 - (q1 * q3 * q4 * q5) ** 0.25


# ---------------------------------------------------------------------------
# Fixed noise assumptions -- IBM Heron r3 (ibm_boston)
# ---------------------------------------------------------------------------

CNOT_ERROR = 1.13e-3      # 2-qubit (CZ) error, Heron r3 median
SQ_ERROR = 1.569e-4       # single-qubit (SX) error, Heron r3 median
CNOT_LATENCY_NS = 68      # CZ gate duration
SQ_LATENCY_NS = 32        # SX gate duration
T1_S = 284.95e-6          # T1 median (ibm_boston)
T2_S = 337.29e-6          # T2 median (ibm_boston)
IDLE_MULTIPLIER = 3.0     # IBM crosstalk/thermal idle scaling

# Futuristic preset: gate error /10, measure error /10, T1/T2 x3.
FUTURISTIC_GATE_DIVISOR = 10.0
FUTURISTIC_MEASURE_DIVISOR = 10.0
FUTURISTIC_T1T2_FACTOR = 3.0

BACKEND_NAME = "real_heron"
DECODER = "mwpm"

def memory_rounds_for(d: int) -> int:
    return d + 1

NUM_SHOTS = 50_000
# Decode in chunks to cap memory: a d=11 (50000 x 42360) detection array is
# ~2.1 GB; chunking keeps each worker's peak bounded (10000 x 42360 ~ 0.4 GB).
CHUNK_SIZE = 10_000
NUM_PROCS = None  # default: min(cpu_count - 2, 8) -- d=11 is memory-heavy


def _decode_chunked(circuit, num_shots, chunk_size=CHUNK_SIZE):
    """MWPM LER with chunked sampling/decoding to bound memory.

    Matches decoders.decode's behaviour for real backends
    (approximate_disjoint_errors=True); a shot fails if any observable is wrong.
    """
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

OUTPUT_DIR = "experiment_results/benchmark"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "mcm_latency_ler.csv")


# ---------------------------------------------------------------------------
# Fast noise model: uniform T1/T2 -> memoize the T1/T2 Pauli channel by duration
# ---------------------------------------------------------------------------

class FastUniformNoise(NoiseModel):
    """NoiseModel with uniform per-qubit T1/T2.

    The baseline NoiseModel recomputes the T1/T2 Pauli channel (numpy exp/clip
    + per-qubit backend lookup) on every idle qubit every TICK -- the dominant
    cost on large lattice-surgery circuits. With uniform coherence the channel
    depends only on the idle duration, so we memoize it. Numerically identical
    to the base model; ~order-of-magnitude faster on d=11.
    """

    def __init__(self, *args, fixed_t1: float, fixed_t2: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_t1 = fixed_t1
        self._fixed_t2 = fixed_t2
        self._channel_cache = {}

    def _t1t2_pauli_channel(self, qubit_idx, duration, circuit):
        if duration <= 0 or self._fixed_t1 <= 0 or self._fixed_t2 <= 0:
            return
        key = round(duration * self.idle_multiplier, 15)
        probs = self._channel_cache.get(key)
        if probs is None:
            te = duration * self.idle_multiplier
            px = 0.25 * (1 - np.exp(-te / self._fixed_t1))
            py = px
            pz = (1 - np.exp(-te / self._fixed_t2)) / 2 - (1 - np.exp(-te / self._fixed_t1)) / 4
            probs = (
                float(np.clip(px, 0.0, 1.0)),
                float(np.clip(py, 0.0, 1.0)),
                float(np.clip(pz, 0.0, 1.0)),
            )
            self._channel_cache[key] = probs
        circuit.append_operation("PAULI_CHANNEL_1", [qubit_idx], list(probs))


# Per-worker cache of the clean tqec circuit, keyed by (distance, rounds).
_CIRCUIT_CACHE = {}


def _clean_circuit(d: int):
    key = (d, memory_rounds_for(d))
    circ = _CIRCUIT_CACHE.get(key)
    if circ is None:
        k = (d - 1) // 2
        circ = single_cnot_n_rounds(None, distance_scale=k, num_memory_rounds=memory_rounds_for(d))
        _CIRCUIT_CACHE[key] = circ
    return circ


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_one(args):
    """args : (distance, readout_ns, system, noise_model)
    returns : (distance, readout_ns, system, noise_model, measure_error, ler)"""
    d, readout_ns, system, noise_model = args
    futuristic = (noise_model == "futuristic")

    measure_error = measure_error_for(system, readout_ns)
    cnot_error = CNOT_ERROR
    sq_error = SQ_ERROR
    t1, t2 = T1_S, T2_S
    if futuristic:
        measure_error /= FUTURISTIC_MEASURE_DIVISOR
        cnot_error /= FUTURISTIC_GATE_DIVISOR
        sq_error /= FUTURISTIC_GATE_DIVISOR
        t1 *= FUTURISTIC_T1T2_FACTOR
        t2 *= FUTURISTIC_T1T2_FACTOR

    circuit = _clean_circuit(d)

    backend = get_backend(BACKEND_NAME, None)
    backend.target.qubit_properties = [
        QubitProperties(t1=t1, t2=t2, frequency=p.frequency)
        for p in backend.target.qubit_properties
    ]
    qt = QubitTracking(backend, None)

    nm = FastUniformNoise(
        sq=sq_error,
        tq=cnot_error,
        measure=measure_error,
        reset=0.0,
        gate_times={
            "SQ": SQ_LATENCY_NS * 1e-9,
            "TQ": CNOT_LATENCY_NS * 1e-9,
            "M": readout_ns * 1e-9,
            "R": 0.0,
        },
        qt=qt,
        backend=backend,
        idle_multiplier=IDLE_MULTIPLIER,
        fixed_t1=t1,
        fixed_t2=t2,
    )

    noisy_circuit = nm.noisy_circuit(circuit)

    try:
        ler = _decode_chunked(noisy_circuit, NUM_SHOTS)
    except Exception as e:
        print(f"Error decoding d={d} readout={readout_ns} {system}/{noise_model}: {e}")
        ler = None

    return (d, readout_ns, system, noise_model, measure_error, ler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configs = list(itertools.product(DISTANCES, READOUT_LENGTHS_NS, SYSTEMS, NOISE_MODELS))

    print("MCM readout length vs logical-CNOT LER (ECCentric, IBM Heron r3)")
    print(f"  Operation       : lattice-surgery logical CNOT (tqec), d+1 rounds")
    print(f"  Coherence       : T1={T1_S*1e6:.1f}us T2={T2_S*1e6:.1f}us (futuristic x{FUTURISTIC_T1T2_FACTOR})")
    print(f"  CZ / SX error   : {CNOT_ERROR} / {SQ_ERROR} (futuristic /{FUTURISTIC_GATE_DIVISOR})")
    print(f"  idle_multiplier : {IDLE_MULTIPLIER}")
    print(f"  Distances       : {DISTANCES}")
    print(f"  Readout lengths : {READOUT_LENGTHS_NS} ns")
    print(f"  Systems         : {SYSTEMS}")
    print(f"  Noise models    : {NOISE_MODELS}")
    print(f"  Total configs   : {len(configs)}")
    print(f"  Shots per config: {NUM_SHOTS:,}")
    print(f"  Output file     : {OUTPUT_CSV}")
    print()
    print("  Measurement error (1 - geomean(Q1,Q3,Q4,Q5), Q2 excluded):")
    for system in SYSTEMS:
        vals = "  ".join(f"{ro}ns={measure_error_for(system, ro):.4f}" for ro in READOUT_LENGTHS_NS)
        print(f"    {system:10}: {vals}")
    print()

    num_procs = NUM_PROCS or max(1, min(mp.cpu_count() - 2, 8))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    header = ["distance", "readout_ns", "system", "noise_model",
              "measure_error", "logical_error_rate"]

    # Incremental, crash-safe CSV: write each result as it completes and flush,
    # so a kill mid-run preserves the configs done so far.
    results = []
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()
        with mp.Pool(num_procs) as pool:
            for result in tqdm(pool.imap_unordered(_run_one, configs),
                               total=len(configs), desc="Simulating"):
                if result[-1] is not None:
                    results.append(result)
                    writer.writerow(result)
                    f.flush()

    # Rewrite the CSV sorted for readability (the incremental file was unsorted).
    results.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print()
    print(f"{'distance':>8}  {'readout_ns':>10}  {'system':>10}  "
          f"{'noise_model':>12}  {'measure_err':>12}  {'LER':>12}")
    print("-" * 80)
    for d, ro, sys_name, nm, merr, ler in results:
        print(f"{d:>8}  {ro:>10}  {sys_name:>10}  {nm:>12}  {merr:>12.6f}  {ler:>12.6f}")
    print()
    print(f"Results saved to {OUTPUT_CSV}")

    # Plots: one per (distance, noise_model), LER vs readout length, line per system.
    markers = {"HERQULES": "o", "MCMit-CNN": "s"}
    for d in DISTANCES:
        for nm in NOISE_MODELS:
            subset = [r for r in results if r[0] == d and r[3] == nm]
            if not subset:
                continue
            plt.figure(figsize=(8, 5))
            for system in SYSTEMS:
                branch = sorted([r for r in subset if r[2] == system], key=lambda x: x[1])
                if not branch:
                    continue
                x = [r[1] for r in branch]
                y = [r[5] for r in branch]
                plt.plot(x, y, marker=markers.get(system, "X"), label=system)
            plt.xlabel("Readout Duration (ns)")
            plt.ylabel("Logical CNOT Error Rate (LER)")
            plt.title(f"MCM Readout Length vs LER — logical CNOT d={d} ({nm})")
            plt.yscale("log")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, f"mcm_latency_ler_d{d}_{nm}.pdf")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
