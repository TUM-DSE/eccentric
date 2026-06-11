"""
compare_readout_artifact.py
===========================
Synchronization-artifact half of the readout-length-vs-logical-CNOT-LER study.

Surface-code lattice-surgery logical CNOT (merge=True), swept over readout
(= measurement) length. Mirrors the ECCentric run: the readout length is the
measurement DURATION and affects LER via idle T1/T2 decoherence during the
measurement window; the readout flip error is FIXED at the ibm_boston median.

Noise (SOTA ibm_boston / Heron r3):
  current    : CZ=1.191e-3, readout=3.54e-3, T1=284.95us, T2=322.68us
  futuristic : all errors / 10, T1/T2 * 3
  idle_multiplier = 3  (artifact's IBM default -- KEPT, per request)
CNOT latency = 70 ns. Decoder: MWPM (built into the artifact's get_error_rate).

Output: artifact_readout_ler.csv (copied into experiment_results/readout_compare/)
        columns: tool,distance,readout_ns,noise_model,measure_error,logical_error_rate
"""

import sys, os, csv, itertools, time
import multiprocessing as mp
from tqdm import tqdm

# Readout model: "herqules" = length-dependent flip error (200-1000ns, realistic);
# "fixed" = ibm_boston median 3.54e-3 regardless of length (fast-readout counterfactual).
READOUT_MODE = "herqules"

# HERQULES per-qubit accuracies (Q1,Q3,Q4,Q5; Q2 excluded); err = 1 - geomean(.).
HERQULES_ACC = {
    200:  (0.7782, 0.8712, 0.7587, 0.7159),
    400:  (0.9340, 0.9473, 0.9445, 0.9717),
    600:  (0.9654, 0.9565, 0.9597, 0.9827),
    800:  (0.9725, 0.9547, 0.9587, 0.9797),
    1000: (0.974, 0.955, 0.958, 0.982),   # updated 1000ns row (Q2=0.732 excluded)
}
def herqules_measure_error(readout_ns):
    q1, q3, q4, q5 = HERQULES_ACC[readout_ns]
    return 1.0 - (q1 * q3 * q4 * q5) ** 0.25

DISTANCES = [5, 7, 9, 11, 13]
READOUT_LENGTHS_NS = ([200, 400, 600, 800, 1000] if READOUT_MODE == "herqules"
                      else [250, 500, 750, 1000, 1500, 2000])
NOISE_MODELS = ["current", "futuristic"]

# SOTA ibm_boston current preset
CZ_ERROR = 1.191e-3
READOUT_ERROR = 3.54e-3
T1_US, T2_US = 284.95, 322.68
CNOT_LATENCY_NS = 70
IDLE_MUL = 3                 # artifact IBM default (kept per request)

FUT_GATE_DIV = 10.0
FUT_MEAS_DIV = 10.0
FUT_T1T2_FAC = 3.0

NUM_SHOTS = 500_000
OUTPUT_CSV = ("artifact_readout_ler_herqules.csv" if READOUT_MODE == "herqules"
             else "artifact_readout_ler.csv")


def _run_one(args):
    d, readout_ns, noise_model = args
    fut = (noise_model == "futuristic")
    base_meas = herqules_measure_error(readout_ns) if READOUT_MODE == "herqules" else READOUT_ERROR
    cnot, meas = CZ_ERROR, base_meas
    t1, t2 = T1_US, T2_US
    if fut:
        cnot /= FUT_GATE_DIV; meas /= FUT_MEAS_DIV
        t1 *= FUT_T1T2_FAC; t2 *= FUT_T1T2_FAC

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sim"))
    from circuit_4 import circuit  # noqa: PLC0415

    t0 = time.time()
    sim = circuit(
        distance=d, num_patches_x=20, num_patches_y=20, spacing=1,
        disable_noise=False,
        fixed_cnot_latency=CNOT_LATENCY_NS, fixed_cnot_noise=cnot,
        fixed_measure_latency=readout_ns, fixed_measure_noise=meas,
        fixed_t1=t1, fixed_t2=t2, idle_multiplier=IDLE_MUL,
        merge=True, basis="Z", ls_basis="X", sync=None,
        rounds_per_op=d + 1,
    ).from_string("qreg q[2];")
    try:
        ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)
    except Exception as e:
        print(f"FAIL artifact d={d} ro={readout_ns} {noise_model}: {e}", flush=True)
        return ("artifact", d, readout_ns, noise_model, meas, None)
    print(f"done artifact d={d:2} ro={readout_ns:4} {noise_model:10} "
          f"LER={ler:.6f} ({time.time()-t0:.1f}s)", flush=True)
    return ("artifact", d, readout_ns, noise_model, meas, ler)


def main():
    configs = list(itertools.product(DISTANCES, READOUT_LENGTHS_NS, NOISE_MODELS))
    print(f"Artifact readout-length sweep — logical CNOT, ibm_boston, idle_mult={IDLE_MUL}")
    print(f"  distances={DISTANCES} readout_ns={READOUT_LENGTHS_NS} shots={NUM_SHOTS}")
    print(f"  configs={len(configs)}")
    num_procs = max(1, min(mp.cpu_count() - 2, 32))

    results = []
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "distance", "readout_ns", "noise_model", "measure_error", "logical_error_rate"])
        f.flush()
        with mp.Pool(num_procs) as pool:
            for r in tqdm(pool.imap_unordered(_run_one, configs), total=len(configs)):
                results.append(r); w.writerow(r); f.flush()

    results.sort(key=lambda x: (x[1], x[3], x[2]))
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "distance", "readout_ns", "noise_model", "measure_error", "logical_error_rate"])
        w.writerows(results)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
