"""
run_herqules_both.py
====================
Re-run BOTH herqules presets (ibm + futuristic) with the CORRECTED measure_error
pairs derived from the new HERQULES per-qubit accuracy table as 1 - gmean4
(Q2 excluded -- geomean of Q1,Q3,Q4,Q5):

    250 -> 0.499   (gmean4 0.501)
    500 -> 0.274   (gmean4 0.726)
    750 -> 0.097   (gmean4 0.903)
   1000 -> 0.049   (gmean4 0.951)

Writes:
    mcm_tradeoff_herqules_ibm.csv          (ibm        : T1=130,  T2=170  us)
    mcm_tradeoff_herqules_futuristic_2.csv (futuristic : T1=390,  T2=510  us)
Both: idle_multiplier=3, merge=True, d+1 rounds, 100k shots, distances 7/9/11.
"""

import sys
import os
import csv
import multiprocessing as mp
from tqdm import tqdm

DISTANCES = [7, 9, 11]

# Corrected HERQULES noise = 1 - gmean4 (Q2 excluded) of the new accuracy table.
LATENCY_ERROR_PAIRS = [
    (250, 0.499),
    (500, 0.274),
    (750, 0.097),
    (1000, 0.049),
]

CNOT_LATENCY_NS = 70
CNOT_ERROR      = 0.0002
IDLE_MUL        = 3
BASIS, LS_BASIS = 'Z', 'X'
NUM_SHOTS = 100_000
NUM_PROCS = int(os.environ.get("PROCS", "0")) or None

PRESETS = [
    # (output_csv, T1_us, T2_us)
    ("mcm_tradeoff_herqules_ibm.csv",          130.0,     170.0),
    ("mcm_tradeoff_herqules_futuristic_2.csv", 130.0 * 3, 170.0 * 3),
]


def _run_one(args):
    d, latency_ns, measure_error, t1, t2 = args
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa
    sim = circuit(
        distance=d, num_patches_x=20, num_patches_y=20, spacing=1,
        disable_noise=False,
        fixed_measure_latency=latency_ns, fixed_measure_noise=measure_error,
        fixed_cnot_latency=CNOT_LATENCY_NS, fixed_cnot_noise=CNOT_ERROR,
        fixed_t1=t1, fixed_t2=t2, idle_multiplier=IDLE_MUL,
        merge=True, basis=BASIS, ls_basis=LS_BASIS, sync=None,
        rounds_per_op=d + 1,
    ).from_string('qreg q[2];')
    ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)
    return (d, latency_ns, measure_error, ler)


def main():
    # All configs across both presets share one pool to maximize utilisation.
    tagged = []  # (preset_idx, d, lat, err, t1, t2)
    for pi, (_out, t1, t2) in enumerate(PRESETS):
        for d in DISTANCES:
            for lat, err in LATENCY_ERROR_PAIRS:
                tagged.append((pi, d, lat, err, t1, t2))

    print(f"corrected HERQULES re-run: {len(tagged)} configs "
          f"({len(PRESETS)} presets x {len(DISTANCES)} d x {len(LATENCY_ERROR_PAIRS)} lat), "
          f"{NUM_SHOTS:,} shots", flush=True)
    print(f"  pairs (1-gmean4): {LATENCY_ERROR_PAIRS}", flush=True)

    num_procs = NUM_PROCS or mp.cpu_count()
    work = [(d, lat, err, t1, t2) for (_pi, d, lat, err, t1, t2) in tagged]

    results_by_preset = {i: [] for i in range(len(PRESETS))}
    with mp.Pool(num_procs) as pool:
        for tag, res in tqdm(zip(tagged, pool.imap(_run_one, work)),
                             total=len(work), desc='Simulating'):
            results_by_preset[tag[0]].append(res)

    for pi, (out, _t1, _t2) in enumerate(PRESETS):
        rows = sorted(results_by_preset[pi], key=lambda x: (x[0], x[1]))
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['distance', 'measure_latency_ns', 'measure_error', 'logical_error_rate'])
            w.writerows(rows)
        print(f"\nwrote {out}", flush=True)
        for d, lat, err, ler in rows:
            print(f"  d={d:<2} {lat:>4}ns merr={err:.3f} LER={ler:.5f}", flush=True)


if __name__ == '__main__':
    main()
