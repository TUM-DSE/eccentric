"""
evaluate_mcm_patch_latency_error_adjusted.py
============================================
Evaluates logical error rate (LER) for a surface-code lattice-surgery merge
with varying total patch counts, mapping of latency to error, and patch-dependent latency.
"""

import sys
import os
import csv
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

DISTANCES = [7, 9, 13]                      # code distances
TOTAL_PATCHES = [16, 32, 48, 64, 80, 96]  # total patches

# Base latency → measurement error mapping
BASE_LATENCY_ERROR_PAIRS = [
    (1000, 0.089),
]

# Extra latency per patch beyond 16 (ns)
PER_PATCH_LATENCY_NS = 16
BASE_PATCH_COUNT = 16

# Fixed CNOT parameters
CNOT_LATENCY_NS = 70
CNOT_ERROR      = 0.002

# Hardware decoherence
T1_US    = 130
T2_US    = 170
IDLE_MUL = 3

# Lattice surgery basis
BASIS    = 'Z'
LS_BASIS = 'X'

# Shots per configuration
NUM_SHOTS = 100_000
NUM_PROCS = None

# Output CSV
OUTPUT_CSV = 'mcm_patch_dist.csv'

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_one(args):
    d, total_patches, base_lat, meas_err = args

    import math
    num_patches_side = math.isqrt(total_patches)

    # Adjust latency by number of extra patches above baseline
    extra_patches = max(total_patches - BASE_PATCH_COUNT, 0)
    adjusted_latency = base_lat + extra_patches * PER_PATCH_LATENCY_NS

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa

    sim = circuit(
        distance          = d,
        num_patches_x     = num_patches_side,
        num_patches_y     = num_patches_side,
        spacing           = 1,
        disable_noise     = False,
        fixed_measure_latency = adjusted_latency,
        fixed_measure_noise   = meas_err,
        fixed_cnot_latency    = CNOT_LATENCY_NS,
        fixed_cnot_noise      = CNOT_ERROR,
        fixed_t1              = T1_US,
        fixed_t2              = T2_US,
        idle_multiplier       = IDLE_MUL,
        merge                 = True,
        basis                 = BASIS,
        ls_basis              = LS_BASIS,
        sync                  = None,
        rounds_per_op         = d + 1,
    ).from_string('qreg q[2];')

    ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)
    return (d, total_patches, base_lat, meas_err, ler)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Build configs
    configs = [
        (d, total, lat, err)
        for d in DISTANCES
        for total in TOTAL_PATCHES
        for lat, err in BASE_LATENCY_ERROR_PAIRS
    ]

    print("Evaluating LER with patch-dependent latency and latency-error mapping")
    print(f"Distances: {DISTANCES}")
    print(f"Total patches: {TOTAL_PATCHES}")
    print(f"Latency → Error pairs: {BASE_LATENCY_ERROR_PAIRS}")
    print(f"Total configs: {len(configs)}")
    print(f"Shots per config: {NUM_SHOTS:,}")
    print(f"Output CSV: {OUTPUT_CSV}\n")

    num_procs = NUM_PROCS or mp.cpu_count()
    results = []

    with mp.Pool(num_procs) as pool:
        for result in tqdm(pool.imap_unordered(_run_one, configs),
                           total=len(configs),
                           desc='Simulating'):
            results.append(result)

    results.sort(key=lambda x: (x[0], x[1], x[2]))

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['distance', 'total_patches', 'measure_latency_ns',
                         'measure_error', 'logical_error_rate'])
        writer.writerows(results)

    # Pretty-print
    print(f"{'d':>5}  {'patches':>7}  {'latency(ns)':>12}  {'err':>8}  {'LER':>10}")
    print('-' * 50)
    for d, total, lat, err, ler in results:
        print(f"{d:>5}  {total:>7}  {lat:>12}  {err:>8.3f}  {ler:>10.6f}")

    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
