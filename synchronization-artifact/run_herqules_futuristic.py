"""
run_herqules_futuristic.py
==========================
Re-run mcm_tradeoff with the UPDATED herqules noise for the FUTURISTIC preset,
producing the panel-(b) data of data/mcm_qec_evaluation.pdf.

Consistent with mcm_tradeoff_herqules_ibm.csv (re-run 2026-06-13 11:29):
  herqules measure_error pairs = (250,0.5) (500,0.292) (750,0.104) (1000,0.081)
Futuristic preset = ibm coherence x3 (T1=130*3 us, T2=170*3 us, idle_mul=3).

Writes mcm_tradeoff_herqules_futuristic_2.csv.
"""

import sys
import os
import csv
import multiprocessing as mp
from tqdm import tqdm

DISTANCES = [7, 9, 11]

# Updated herqules noise (same pairs as mcm_tradeoff_herqules_ibm.csv).
LATENCY_ERROR_PAIRS = [
    (250, 0.5),
    (500, 0.292),
    (750, 0.104),
    (1000, 0.081),
]

CNOT_LATENCY_NS = 70
CNOT_ERROR      = 0.0002

# Futuristic preset: ibm coherence x3.
T1_US    = 130 * 3
T2_US    = 170 * 3
IDLE_MUL = 3

BASIS    = 'Z'
LS_BASIS = 'X'
NUM_SHOTS = 100_000
NUM_PROCS = int(os.environ.get("PROCS", "0")) or None
OUTPUT_CSV = 'mcm_tradeoff_herqules_futuristic_2.csv'


def _run_one(args):
    d, latency_ns, measure_error = args
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa
    sim = circuit(
        distance          = d,
        num_patches_x     = 20,
        num_patches_y     = 20,
        spacing           = 1,
        disable_noise     = False,
        fixed_measure_latency = latency_ns,
        fixed_measure_noise   = measure_error,
        fixed_cnot_latency    = CNOT_LATENCY_NS,
        fixed_cnot_noise      = CNOT_ERROR,
        fixed_t1         = T1_US,
        fixed_t2         = T2_US,
        idle_multiplier  = IDLE_MUL,
        merge    = True,
        basis    = BASIS,
        ls_basis = LS_BASIS,
        sync     = None,
        rounds_per_op = d + 1,
    ).from_string('qreg q[2];')
    ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)
    return (d, latency_ns, measure_error, ler)


def main():
    configs = [(d, lat, err) for d in DISTANCES for lat, err in LATENCY_ERROR_PAIRS]
    print(f"herqules futuristic re-run: {len(configs)} configs, {NUM_SHOTS:,} shots", flush=True)
    print(f"  pairs   : {LATENCY_ERROR_PAIRS}", flush=True)
    print(f"  T1/T2   : {T1_US}/{T2_US} us  idle_mul={IDLE_MUL}", flush=True)
    print(f"  output  : {OUTPUT_CSV}", flush=True)

    num_procs = NUM_PROCS or mp.cpu_count()
    results = []
    with mp.Pool(num_procs) as pool:
        for r in tqdm(pool.imap_unordered(_run_one, configs),
                      total=len(configs), desc='Simulating'):
            results.append(r)

    results.sort(key=lambda x: (x[0], x[1]))
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['distance', 'measure_latency_ns', 'measure_error', 'logical_error_rate'])
        w.writerows(results)

    print(f"\n{'distance':>10}  {'latency(ns)':>12}  {'measure_error':>15}  {'LER':>12}")
    print('-' * 60)
    for d, lat, err, ler in results:
        print(f"{d:>10}  {lat:>12}  {err:>15.4f}  {ler:>12.6f}")
    print(f"\nResults saved to {OUTPUT_CSV}", flush=True)


if __name__ == '__main__':
    main()
