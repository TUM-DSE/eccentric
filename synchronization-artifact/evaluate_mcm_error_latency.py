"""
evaluate_mcm_error_latency.py
=============================
Evaluates how BOTH mid-circuit measurement (MCM) error rate AND
MCM latency affect the logical error rate (LER) of a surface-code
memory experiment with lattice surgery (merge=True).

Fixed parameters
----------------
- CNOT latency        : CNOT_LATENCY_NS
- CNOT error rate     : CNOT_ERROR
- T1, T2              : fixed per hardware preset
- idle_multiplier     : 1 (Google) or 3 (IBM)
- No synchronization  : sync=None

Swept parameters
----------------
- Measurement error   : MEASURE_ERRORS (10 values)
- MCM latency (ns)    : MEASURE_LATENCIES_NS
- Code distance       : DISTANCES

Output
------
CSV file  ``mcm_error_latency_ler.csv``  with columns:
    distance, measure_error, measure_latency_ns, logical_error_rate
"""

import sys
import os
import csv
import itertools
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

# 10 standard measurement error values
MEASURE_ERRORS = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05
]

# Latencies to sweep (ns)
MEASURE_LATENCIES_NS = [250, 500, 1000, 1500]

# Code distances (odd integers only)
DISTANCES = [7]

# Fixed CNOT parameters
CNOT_LATENCY_NS = 70
CNOT_ERROR      = 0.002

# Hardware decoherence
T1_US    = 130
T2_US    = 170
IDLE_MUL = 3

# Logical operator settings
BASIS    = 'Z'
LS_BASIS = 'X'

# Shots per configuration
NUM_SHOTS = 500_000

# Parallel workers
NUM_PROCS = None

# Output file
OUTPUT_CSV = 'mcm_error_latency_ler.csv'

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_one(args):
    """
    Simulate one (distance, measure_error, measure_latency) configuration.
    """
    d, measure_error, measure_latency = args

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa

    sim = circuit(
        distance          = d,
        num_patches_x     = 20,
        num_patches_y     = 20,
        spacing           = 1,
        disable_noise     = False,

        # --- swept parameters ---
        fixed_measure_latency = measure_latency,
        fixed_measure_noise   = measure_error,

        # --- fixed CNOT parameters ---
        fixed_cnot_latency    = CNOT_LATENCY_NS,
        fixed_cnot_noise      = CNOT_ERROR,

        # --- decoherence ---
        fixed_t1         = T1_US,
        fixed_t2         = T2_US,
        idle_multiplier  = IDLE_MUL,

        # --- logical operator ---
        merge    = True,
        basis    = BASIS,
        ls_basis = LS_BASIS,
        sync     = None,
        rounds_per_op = d + 1,
    ).from_string('qreg q[2];')

    ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)

    return (d, measure_error, measure_latency, ler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    configs = list(itertools.product(
        DISTANCES,
        MEASURE_ERRORS,
        MEASURE_LATENCIES_NS
    ))

    print("Evaluating combined MCM error + latency effect on LER")
    print(f"  Distances        : {DISTANCES}")
    print(f"  Measure errors   : {MEASURE_ERRORS}")
    print(f"  Latencies (ns)   : {MEASURE_LATENCIES_NS}")
    print(f"  Total configs    : {len(configs)}")
    print(f"  Shots per config : {NUM_SHOTS:,}")
    print(f"  Output file      : {OUTPUT_CSV}")
    print()

    num_procs = NUM_PROCS or mp.cpu_count()

    results = []
    with mp.Pool(num_procs) as pool:
        for result in tqdm(pool.imap_unordered(_run_one, configs),
                           total=len(configs),
                           desc='Simulating'):
            results.append(result)

    # Sort by distance → error → latency
    results.sort(key=lambda x: (x[0], x[1], x[2]))

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'distance',
            'measure_error',
            'measure_latency_ns',
            'logical_error_rate'
        ])
        writer.writerows(results)

    # Pretty-print summary
    print()
    print(f"{'distance':>10}  {'measure_error':>15}  {'latency(ns)':>12}  {'LER':>12}")
    print('-' * 60)
    for d, err, lat, ler in results:
        print(f"{d:>10}  {err:>15.5f}  {lat:>12}  {ler:>12.6f}")

    print()
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
