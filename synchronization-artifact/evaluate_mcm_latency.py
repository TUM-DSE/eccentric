"""
evaluate_mcm_latency.py
=======================
Evaluates how mid-circuit measurement (MCM) latency affects the logical error
rate (LER) of a surface-code memory experiment with a lattice-surgery logical
operator (merge=True).

Fixed parameters
----------------
- CNOT latency        : CNOT_LATENCY_NS  (default 50 ns, Google-like)
- CNOT error rate     : CNOT_ERROR
- Measurement error   : MEASURE_ERROR  (kept equal to CNOT_ERROR so that only
                        the *idling* effect of latency changes)
- T1, T2              : fixed per hardware preset
- idle_multiplier     : 1 (Google) or 3 (IBM)
- No synchronization  : sync=None  (pure latency sweep, no active/passive sync)

Swept parameters
----------------
- MCM latency         : MEASURE_LATENCIES_NS (list)
- Code distance       : DISTANCES (list of odd integers)

Output
------
CSV file  ``mcm_latency_ler.csv``  with columns:
    distance, mcm_latency_ns, logical_error_rate

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

# MCM latency sweep (ns)
MEASURE_LATENCIES_NS = [250, 500, 750, 1000, 1500, 2000]

# Code distances to evaluate (odd integers only)
DISTANCES = [7, 9, 11]

# Fixed gate parameters
CNOT_LATENCY_NS   = 70          # ns
CNOT_ERROR        = 0.0002       # depolarising error per CX
MEASURE_ERROR     = 0.0005       # bit-flip error per measurement

# Hardware decoherence (Google Sycamore-like defaults)
T1_US    = 170     # µs  (fixed_t1 in the circuit constructor)
T2_US    = 130     # µs  (fixed_t2)
IDLE_MUL = 3      # scale factor for idling errors (1 = Google, 3 = IBM)

# Lattice surgery basis
BASIS    = 'Z'
LS_BASIS = 'X'    # merge along X boundary

# Shots per configuration (increase for lower statistical noise)
NUM_SHOTS = 500_000

# Number of parallel worker processes (None → use all available CPUs)
NUM_PROCS = None

# Output CSV path
OUTPUT_CSV = 'mcm_latency_ler.csv'

# ---------------------------------------------------------------------------
# Worker function (runs inside a subprocess via multiprocessing)
# ---------------------------------------------------------------------------

def _run_one(args):
    """
    Simulate one (distance, mcm_latency) configuration and return its LER.

    Parameters
    ----------
    args : tuple
        (distance: int, mcm_latency_ns: float)

    Returns
    -------
    tuple
        (distance, mcm_latency_ns, ler: float)
    """
    d, mcm_latency = args

    # Import here so each worker process initialises its own copy
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa: PLC0415

    sim = circuit(
        distance          = d,
        num_patches_x     = 20,
        num_patches_y     = 20,
        spacing           = 1,
        disable_noise     = False,
        # --- fixed gate parameters ---
        fixed_cnot_latency  = CNOT_LATENCY_NS,
        fixed_cnot_noise    = CNOT_ERROR,
        # --- swept MCM latency (error rate kept fixed) ---
        fixed_measure_latency = mcm_latency,
        fixed_measure_noise   = MEASURE_ERROR,
        # --- decoherence model ---
        fixed_t1         = T1_US,
        fixed_t2         = T2_US,
        idle_multiplier  = IDLE_MUL,
        # --- logical operator: full lattice-surgery merge ---
        merge    = True,
        basis    = BASIS,
        ls_basis = LS_BASIS,
        # --- no additional sync overhead ---
        sync     = None,
        # --- rounds per syndrome cycle ---
        rounds_per_op = d + 1,
    ).from_string('qreg q[2];')   # 2 logical qubits → one merge = one logical op

    ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)
    return (d, mcm_latency, ler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configs = list(itertools.product(DISTANCES, MEASURE_LATENCIES_NS))

    print(f"Evaluating MCM latency effect on LER")
    print(f"  Distances       : {DISTANCES}")
    print(f"  MCM latencies   : {MEASURE_LATENCIES_NS} ns")
    print(f"  Total configs   : {len(configs)}")
    print(f"  Shots per config: {NUM_SHOTS:,}")
    print(f"  Output file     : {OUTPUT_CSV}")
    print()

    num_procs = NUM_PROCS or mp.cpu_count()

    results = []
    with mp.Pool(num_procs) as pool:
        for result in tqdm(pool.imap_unordered(_run_one, configs),
                           total=len(configs),
                           desc='Simulating'):
            results.append(result)

    # Sort for readability: by distance, then by MCM latency
    results.sort(key=lambda x: (x[0], x[1]))

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['distance', 'mcm_latency_ns', 'logical_error_rate'])
        writer.writerows(results)

    # Pretty-print table
    print()
    print(f"{'distance':>10}  {'mcm_latency_ns':>16}  {'LER':>12}")
    print('-' * 44)
    for d, lat, ler in results:
        print(f"{d:>10}  {lat:>16}  {ler:>12.6f}")

    print()
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
