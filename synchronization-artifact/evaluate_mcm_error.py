"""
evaluate_mcm_error.py
=====================
Evaluates how mid-circuit measurement (MCM) error rate affects the logical
error rate (LER) of a surface-code memory experiment with a lattice-surgery
logical operator (merge=True).

Fixed parameters
----------------
- MCM latency         : MEASURE_LATENCY_NS, per hardware preset (see PRESETS)
- CNOT latency, error  : per hardware preset
- T1, T2               : per hardware preset
- idle_multiplier      : per hardware preset (1 = no IBM idling correction, 3 = IBM)
- No synchronization   : sync=None

Hardware presets
-----------------
Selected with ``--preset {google,ibm,ibm_futuristic}``. Each preset's numbers
are taken from the values already used consistently elsewhere in this
synchronization-artifact for the same three regimes, so results here line up
with the rest of the artifact (e.g. Fig. 13's four-panel evaluation):

  google         T1=25us,  T2=40us,  CNOT lat=50ns,  CNOT err=0.006,
                 MCM lat=660ns, idle_multiplier=1
                 -- matches the "Google-like" block in hybrid_sync.py /
                 synchronization_zzxx.py / ideal_synchronization_zxxz.py
                 (fixed_t1=25, fixed_t2=40, fixed_cnot_latency=50,
                 fixed_measure_latency=660, idle_multiplier=1), and the
                 CNOT error matches noise/sycamore_noise.py's tq=0.0062.

  ibm            T1=130us, T2=170us, CNOT lat=70ns,  CNOT err=0.0002,
                 MCM lat=500ns, idle_multiplier=3
                 -- matches the IBM baseline used throughout the Fig. 13
                 pipeline: evaluate_patches.py, evaluate_mcm_error_latency.py,
                 run_herqules_both.py, and the "IBM-like" block in
                 hybrid_sync.py / synchronization_zzxx.py (idle_multiplier=3
                 for IBM, per the "lifting the idling errors for IBM
                 systems" comment in sim/gate_lib.py).

  ibm_futuristic Same as ibm but T1/T2 x3 and CNOT err /10
                 -- matches run_herqules_futuristic.py and the MCMit paper's
                 futuristic regime (gate errors /10, decoherence x3, § 8.5).

Previously this script hardcoded T1=170us/T2=130us with idle_multiplier=3
(the IBM idle-correction flag) yet its output was saved as "..._google.csv"
-- i.e. the on-disk T1/T2 were swapped relative to the "ibm" preset used
everywhere else in this artifact, and idle_multiplier did not match the
"google" label at all. This version makes the preset explicit and
selectable instead of silently mismatched.

Swept parameters
----------------
- Measurement error   : MEASURE_ERRORS (list of floats)
- Code distance       : DISTANCES (list of odd integers)

Output
------
CSV file  ``mcm_error_ler_<preset>.csv``  with columns:
    distance, measure_error, logical_error_rate

and a summary printed to stdout.
"""

import sys
import os
import csv
import argparse
import itertools
import multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hardware presets -- see module docstring for provenance of every number.
# ---------------------------------------------------------------------------

PRESETS = {
    "google": dict(
        t1_us=25, t2_us=40,
        cnot_latency_ns=50, cnot_error=0.006,
        measure_latency_ns=660,
        idle_mul=1,
    ),
    "ibm": dict(
        t1_us=130, t2_us=170,
        cnot_latency_ns=70, cnot_error=0.0002,
        measure_latency_ns=500,
        idle_mul=3,
    ),
    "ibm_futuristic": dict(
        t1_us=130 * 3, t2_us=170 * 3,
        cnot_latency_ns=70, cnot_error=0.0002 / 10,
        measure_latency_ns=500,
        idle_mul=3,
    ),
}

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

# Measurement error rate sweep
MEASURE_ERRORS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

# Code distances to evaluate (odd integers only)
DISTANCES = [7, 9, 11]

# Lattice surgery basis
BASIS    = 'Z'
LS_BASIS = 'X'

# Shots per configuration
NUM_SHOTS = 500_000

# Number of parallel worker processes (None → all CPUs)
NUM_PROCS = None

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_one(args):
    """
    Simulate one (distance, measure_error) configuration under a given preset.

    Parameters
    ----------
    args : tuple
        (distance, measure_error, measure_latency_ns, cnot_latency_ns,
         cnot_error, t1_us, t2_us, idle_mul)

    Returns
    -------
    tuple
        (distance, measure_error, ler: float)
    """
    (d, measure_error, measure_latency_ns, cnot_latency_ns,
     cnot_error, t1_us, t2_us, idle_mul) = args

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sim'))
    from circuit_4 import circuit  # noqa: PLC0415

    sim = circuit(
        distance          = d,
        num_patches_x     = 20,
        num_patches_y     = 20,
        spacing           = 1,
        disable_noise     = False,
        # --- fixed MCM latency ---
        fixed_measure_latency = measure_latency_ns,
        # --- swept measurement error ---
        fixed_measure_noise   = measure_error,
        # --- fixed CNOT parameters ---
        fixed_cnot_latency    = cnot_latency_ns,
        fixed_cnot_noise      = cnot_error,
        # --- decoherence model ---
        fixed_t1         = t1_us,
        fixed_t2         = t2_us,
        idle_multiplier  = idle_mul,
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
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='google',
                         help="Hardware regime to simulate (default: google, "
                              "used by the paper's Fig. 3 motivation plot).")
    args = parser.parse_args()

    p = PRESETS[args.preset]
    output_csv = f'mcm_error_ler_{args.preset}.csv'

    configs = [
        (d, err, p['measure_latency_ns'], p['cnot_latency_ns'], p['cnot_error'],
         p['t1_us'], p['t2_us'], p['idle_mul'])
        for d, err in itertools.product(DISTANCES, MEASURE_ERRORS)
    ]

    print(f"Evaluating MCM error rate effect on LER")
    print(f"  Preset            : {args.preset}")
    print(f"  T1 / T2           : {p['t1_us']} us / {p['t2_us']} us")
    print(f"  CNOT lat / error  : {p['cnot_latency_ns']} ns / {p['cnot_error']}")
    print(f"  idle_multiplier   : {p['idle_mul']}")
    print(f"  Distances         : {DISTANCES}")
    print(f"  Measure errors    : {MEASURE_ERRORS}")
    print(f"  Fixed MCM latency : {p['measure_latency_ns']} ns")
    print(f"  Total configs     : {len(configs)}")
    print(f"  Shots per config  : {NUM_SHOTS:,}")
    print(f"  Output file       : {output_csv}")
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
    with open(output_csv, 'w', newline='') as f:
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
    print(f"Results saved to {output_csv}")


if __name__ == '__main__':
    main()
