"""
evaluate_mcm_error.py
=====================
Evaluates how mid-circuit measurement (MCM) error rate affects the logical
error rate (LER) of a surface-code memory experiment with a lattice-surgery
logical operator (merge=True).

Fixed parameters
----------------
- MCM latency         : MEASURE_LATENCY_NS = 2000 ns (2 µs)
- CNOT latency        : CNOT_LATENCY_NS
- CNOT error rate     : CNOT_ERROR
- T1, T2              : fixed per hardware preset
- idle_multiplier     : 1 (Google) or 3 (IBM)
- No synchronization  : sync=None

Swept parameters
----------------
- Measurement error   : MEASURE_ERRORS (list of floats)
- Code distance       : DISTANCES (list of odd integers)

Output
------
CSV file  ``mcm_error_ler.csv``  with columns:
    distance, measure_error, logical_error_rate

and a summary printed to stdout.
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

# Measurement error rate sweep
MEASURE_ERRORS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

# Code distances to evaluate (odd integers only)
DISTANCES = [7, 9, 11]

# Fixed MCM latency (ns) — 2 µs
MEASURE_LATENCY_NS = 1500

# Fixed gate parameters
CNOT_LATENCY_NS = 70      # ns
CNOT_ERROR      = 0.002   # depolarising error per CX (kept fixed)

# Hardware decoherence
T1_US    = 170   # µs
T2_US    = 130   # µs
IDLE_MUL = 3     # 1 = Google, 3 = IBM

# Lattice surgery basis
BASIS    = 'Z'
LS_BASIS = 'X'

# Shots per configuration
NUM_SHOTS = 500_000

# Number of parallel worker processes (None → all CPUs)
NUM_PROCS = None

# Output CSV path
OUTPUT_CSV = 'mcm_error_ler.csv'

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_one(args):
    """
    Simulate one (distance, measure_error) configuration.

    Parameters
    ----------
    args : tuple
        (distance: int, measure_error: float)

    Returns
    -------
    tuple
        (distance, measure_error, ler: float)
    """
    d, measure_error = args

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa: PLC0415

    sim = circuit(
        distance          = d,
        num_patches_x     = 20,
        num_patches_y     = 20,
        spacing           = 1,
        disable_noise     = False,
        # --- fixed MCM latency ---
        fixed_measure_latency = MEASURE_LATENCY_NS,
        # --- swept measurement error ---
        fixed_measure_noise   = measure_error,
        # --- fixed CNOT parameters ---
        fixed_cnot_latency    = CNOT_LATENCY_NS,
        fixed_cnot_noise      = CNOT_ERROR,
        # --- decoherence model ---
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
    return (d, measure_error, ler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configs = list(itertools.product(DISTANCES, MEASURE_ERRORS))

    print(f"Evaluating MCM error rate effect on LER")
    print(f"  Distances         : {DISTANCES}")
    print(f"  Measure errors    : {MEASURE_ERRORS}")
    print(f"  Fixed MCM latency : {MEASURE_LATENCY_NS} ns")
    print(f"  Total configs     : {len(configs)}")
    print(f"  Shots per config  : {NUM_SHOTS:,}")
    print(f"  Output file       : {OUTPUT_CSV}")
    print()

    num_procs = NUM_PROCS or mp.cpu_count()

    results = []
    with mp.Pool(num_procs) as pool:
        for result in tqdm(pool.imap_unordered(_run_one, configs),
                           total=len(configs),
                           desc='Simulating'):
            results.append(result)

    results.sort(key=lambda x: (x[0], x[1]))

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['distance', 'measure_error', 'logical_error_rate'])
        writer.writerows(results)

    # Pretty-print table
    print()
    print(f"{'distance':>10}  {'measure_error':>15}  {'LER':>12}")
    print('-' * 42)
    for d, err, ler in results:
        print(f"{d:>10}  {err:>15.4f}  {ler:>12.6f}")

    print()
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
