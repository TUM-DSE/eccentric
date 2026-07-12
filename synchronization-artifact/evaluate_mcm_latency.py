"""
evaluate_mcm_latency.py
=======================
Evaluates how mid-circuit measurement (MCM) latency affects the logical error
rate (LER) of a surface-code memory experiment with a lattice-surgery logical
operator (merge=True).

Fixed parameters
----------------
- CNOT latency, error  : per hardware preset (see PRESETS)
- Measurement error    : per hardware preset, held fixed across the whole
                        sweep and set equal to that preset's CNOT error, so
                        that only the *idling* (decoherence) effect of
                        latency changes -- not the readout fidelity itself.
- T1, T2               : per hardware preset
- idle_multiplier      : per hardware preset (1 = no IBM idling correction, 3 = IBM)
- No synchronization   : sync=None  (pure latency sweep, no active/passive sync)

Hardware presets
-----------------
Selected with ``--preset {google,ibm,ibm_futuristic}``. See
evaluate_mcm_error.py's module docstring for the full provenance of every
number -- the same three presets are used here for consistency:

  google         T1=25us,  T2=40us,  CNOT lat=50ns,  CNOT/measure err=0.006,
                 idle_multiplier=1
  ibm            T1=130us, T2=170us, CNOT lat=70ns,  CNOT/measure err=0.0002,
                 idle_multiplier=3
  ibm_futuristic Same as ibm but T1/T2 x3 and CNOT/measure err /10

Previously this script hardcoded T1=170us/T2=130us with idle_multiplier=3,
mislabeled in a comment as "Google Sycamore-like defaults" even though
idle_multiplier=3 is this artifact's IBM idle-correction flag (see
sim/gate_lib.py) -- and MEASURE_ERROR (0.0005) didn't actually equal
CNOT_ERROR (0.0002) despite the docstring's claim. This version makes the
preset explicit/selectable and keeps measure error == CNOT error for real.

Swept parameters
----------------
- MCM latency         : MEASURE_LATENCIES_NS (list)
- Code distance       : DISTANCES (list of odd integers)

Output
------
CSV file  ``mcm_latency_ler_<preset>.csv``  with columns:
    distance, mcm_latency_ns, logical_error_rate

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
# Hardware presets -- see evaluate_mcm_error.py's docstring for provenance.
# ---------------------------------------------------------------------------

PRESETS = {
    "google": dict(
        t1_us=25, t2_us=40,
        cnot_latency_ns=50, cnot_error=0.006,
        idle_mul=1,
    ),
    "ibm": dict(
        t1_us=130, t2_us=170,
        cnot_latency_ns=70, cnot_error=0.0002,
        idle_mul=3,
    ),
    "ibm_futuristic": dict(
        t1_us=130 * 3, t2_us=170 * 3,
        cnot_latency_ns=70, cnot_error=0.0002 / 10,
        idle_mul=3,
    ),
}

# ---------------------------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------------------------

# MCM latency sweep (ns)
MEASURE_LATENCIES_NS = [250, 500, 750, 1000, 1500, 2000]

# Code distances to evaluate (odd integers only)
DISTANCES = [7, 9, 11]

# Lattice surgery basis
BASIS    = 'Z'
LS_BASIS = 'X'    # merge along X boundary

# Shots per configuration (increase for lower statistical noise)
NUM_SHOTS = 500_000

# Number of parallel worker processes (None → use all available CPUs)
NUM_PROCS = None

# ---------------------------------------------------------------------------
# Worker function (runs inside a subprocess via multiprocessing)
# ---------------------------------------------------------------------------

def _run_one(args):
    """
    Simulate one (distance, mcm_latency) configuration and return its LER.

    Parameters
    ----------
    args : tuple
        (distance, mcm_latency_ns, cnot_latency_ns, cnot_error,
         measure_error, t1_us, t2_us, idle_mul)

    Returns
    -------
    tuple
        (distance, mcm_latency_ns, ler: float)
    """
    (d, mcm_latency, cnot_latency_ns, cnot_error,
     measure_error, t1_us, t2_us, idle_mul) = args

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
        fixed_cnot_latency  = cnot_latency_ns,
        fixed_cnot_noise    = cnot_error,
        # --- swept MCM latency (error rate kept fixed) ---
        fixed_measure_latency = mcm_latency,
        fixed_measure_noise   = measure_error,
        # --- decoherence model ---
        fixed_t1         = t1_us,
        fixed_t2         = t2_us,
        idle_multiplier  = idle_mul,
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
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='google',
                         help="Hardware regime to simulate (default: google, "
                              "used by the paper's Fig. 3 motivation plot).")
    args = parser.parse_args()

    p = PRESETS[args.preset]
    measure_error = p['cnot_error']  # held fixed == CNOT error, see docstring
    output_csv = f'mcm_latency_ler_{args.preset}.csv'

    configs = [
        (d, lat, p['cnot_latency_ns'], p['cnot_error'], measure_error,
         p['t1_us'], p['t2_us'], p['idle_mul'])
        for d, lat in itertools.product(DISTANCES, MEASURE_LATENCIES_NS)
    ]

    print(f"Evaluating MCM latency effect on LER")
    print(f"  Preset          : {args.preset}")
    print(f"  T1 / T2         : {p['t1_us']} us / {p['t2_us']} us")
    print(f"  CNOT lat / error: {p['cnot_latency_ns']} ns / {p['cnot_error']}")
    print(f"  Measure error   : {measure_error}")
    print(f"  idle_multiplier : {p['idle_mul']}")
    print(f"  Distances       : {DISTANCES}")
    print(f"  MCM latencies   : {MEASURE_LATENCIES_NS} ns")
    print(f"  Total configs   : {len(configs)}")
    print(f"  Shots per config: {NUM_SHOTS:,}")
    print(f"  Output file     : {output_csv}")
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
    with open(output_csv, 'w', newline='') as f:
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
    print(f"Results saved to {output_csv}")


if __name__ == '__main__':
    main()
