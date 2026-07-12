"""
run_tradeoff_final.py
=====================
Regenerate ALL FOUR mcm_tradeoff CSVs (CNN + HERQULES, current + futuristic)
for panels (a)/(b), using lattice-sim (sim/circuit_4).

Authoritative inputs (from the two accuracy tables):
  measure_error = 1 - geomean(Q1, Q3, Q4, Q5)   # gmean4, Q2 EXCLUDED
CNN table  = the 0.910-gmean5 table (MCMit-CNN).
HERQULES   = the 0.905-gmean5 table.

Fixed:
  CNOT error   = 0.003          # per spec (not 0.0002)
  CNOT latency = 70 ns
  idle_mult    = 3
  shots        = 100_000, distances 7/9/11, merge=True, d+1 rounds, sync=None

Presets (coherence only differs):
  current    (ibm)        T1=130 us  T2=170 us  -> *_ibm.csv
  futuristic (ibm x3)     T1=390 us  T2=510 us  -> *_futuristic_2.csv
"""

import sys
import os
import csv
import multiprocessing as mp
from tqdm import tqdm

DISTANCES = [7, 9, 11]
DURATIONS = [250, 500, 750, 1000]

# Per-qubit accuracies (Q1, Q3, Q4, Q5) -- Q2 EXCLUDED.
CNN_ACC = {
    250:  (0.835, 0.895, 0.833, 0.856),
    500:  (0.939, 0.934, 0.927, 0.968),
    750:  (0.964, 0.940, 0.943, 0.970),
    1000: (0.970, 0.942, 0.947, 0.970),
}
HERQULES_ACC = {
    250:  (0.500, 0.506, 0.500, 0.500),
    500:  (0.513, 0.915, 0.631, 0.936),
    750:  (0.831, 0.922, 0.895, 0.969),
    1000: (0.969, 0.925, 0.942, 0.970),
}
SYSTEMS = {"mcm": CNN_ACC, "herqules": HERQULES_ACC}  # file stems


def measure_error(acc, ns):
    q1, q3, q4, q5 = acc[ns]
    return 1.0 - (q1 * q3 * q4 * q5) ** 0.25


CNOT_LATENCY_NS = 70
CNOT_ERROR      = 0.003          # <-- per spec
IDLE_MUL        = 3
BASIS, LS_BASIS = 'Z', 'X'
NUM_SHOTS = 100_000
NUM_PROCS = int(os.environ.get("PROCS", "0")) or None

PRESETS = [
    # (file_suffix, T1_us, T2_us)
    ("ibm",          130.0,     170.0),       # current
    ("futuristic_2", 130.0 * 3, 170.0 * 3),   # futuristic
]


def _run_one(args):
    d, latency_ns, merr, t1, t2 = args
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa
    sim = circuit(
        distance=d, num_patches_x=20, num_patches_y=20, spacing=1,
        disable_noise=False,
        fixed_measure_latency=latency_ns, fixed_measure_noise=merr,
        fixed_cnot_latency=CNOT_LATENCY_NS, fixed_cnot_noise=CNOT_ERROR,
        fixed_t1=t1, fixed_t2=t2, idle_multiplier=IDLE_MUL,
        merge=True, basis=BASIS, ls_basis=LS_BASIS, sync=None,
        rounds_per_op=d + 1,
    ).from_string('qreg q[2];')
    ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)
    return (d, latency_ns, merr, ler)


def main():
    print(f"CNOT_ERROR={CNOT_ERROR}  shots={NUM_SHOTS}  measure_error=1-gmean4(Q1,Q3,Q4,Q5)",
          flush=True)
    for sysname, acc in SYSTEMS.items():
        pairs = [(ns, measure_error(acc, ns)) for ns in DURATIONS]
        print(f"\n{sysname} measure_error: "
              + "  ".join(f"{ns}ns={e:.4f}" for ns, e in pairs), flush=True)

    # Build all configs across systems x presets x d x lat into one pool.
    tagged = []  # (sysname, suffix, d, lat, err, t1, t2)
    for sysname, acc in SYSTEMS.items():
        for suffix, t1, t2 in PRESETS:
            for d in DISTANCES:
                for ns in DURATIONS:
                    tagged.append((sysname, suffix, d, ns, measure_error(acc, ns), t1, t2))

    num_procs = NUM_PROCS or mp.cpu_count()
    work = [(d, lat, err, t1, t2) for (_s, _suf, d, lat, err, t1, t2) in tagged]

    buckets = {}
    with mp.Pool(num_procs) as pool:
        for tag, res in tqdm(zip(tagged, pool.imap(_run_one, work)),
                             total=len(work), desc='Simulating'):
            buckets.setdefault((tag[0], tag[1]), []).append(res)

    for (sysname, suffix), rows in buckets.items():
        rows.sort(key=lambda x: (x[0], x[1]))
        out = f"mcm_tradeoff_{sysname}_{suffix}.csv"
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['distance', 'measure_latency_ns', 'measure_error', 'logical_error_rate'])
            w.writerows(rows)
        print(f"\nwrote {out}", flush=True)
        for d, lat, err, ler in rows:
            print(f"  d={d:<2} {lat:>4}ns merr={err:.4f} LER={ler:.5f}", flush=True)


if __name__ == '__main__':
    main()
