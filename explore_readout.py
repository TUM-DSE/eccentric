"""
explore_readout.py
==================
Test mechanisms that should make readout LENGTH affect LER through decoherence
(not just the readout flip error). Surface code, Model-1 base (ibm_boston gates +
CNN readout + decoherence). Variants:

  baseline   : Model 1, rounds = d (the original Plot-1 setup).
  backlog    : Model 1 + decoder-reaction idle = rounds x readout, applied to ALL
               qubits before the final measurement (option 1).
  rounds_5d  : Model 1 but with 5*d syndrome rounds (option 4).

Output: experiment_results/noise_models/surface_readout_mechanisms.csv
        columns: variant, distance, readout_ns, logical_error_rate
"""
import os, sys, csv
import multiprocessing as mp
sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))
sys.path.append(os.getcwd())
import numpy as np
import stim
from codes import get_code
from backends import get_backend, QubitTracking
import run_ler_noise_models as R   # reuse FastUniformNoise, constants, decode_mwpm

DISTANCES = [5, 7, 9, 11, 13]
READOUT_LENGTHS_NS = [200, 400, 600, 800, 1000]
VARIANTS = ["baseline", "backlog", "rounds_5d"]
SHOTS = 50_000
OUT = "experiment_results/noise_models/surface_readout_mechanisms.csv"


def build(d, readout_ns, variant):
    rounds = 5 * d if variant == "rounds_5d" else d
    code = get_code("surface", d, rounds)
    backend = get_backend("real_heron", None)
    qt = QubitTracking(backend, None)
    meas = R.cnn_measure_error(readout_ns)
    nm = R.FastUniformNoise(
        sq=R.HERON_SQ, tq=R.HERON_TQ, measure=meas, reset=0.0,
        gate_times={"SQ": R.SQ_NS * 1e-9, "TQ": R.TQ_NS * 1e-9,
                    "M": readout_ns * 1e-9, "R": R.R_S},
        qt=qt, backend=backend, idle_multiplier=R.IDLE_MULTIPLIER,
        fixed_t1=R.T1_S, fixed_t2=R.T2_S,
    )
    circ = nm.noisy_circuit(code.stim_circuit)

    if variant == "backlog":
        # Reaction backlog: a longer readout lengthens the cycle, so the patch is
        # held (all qubits idle) ~ rounds x readout while decoding catches up.
        t_idle = rounds * readout_ns * 1e-9
        final = len(circ)
        for i in range(len(circ) - 1, -1, -1):
            if isinstance(circ[i], stim.CircuitRepeatBlock):
                final = i + 1
                if final < len(circ) and circ[final].name == "TICK":
                    final += 1
                break
        delay = nm.get_idle_delay_circuit(t_idle, circ.num_qubits)
        circ = circ[0:final] + delay + circ[final:]
    return circ


def run_one(args):
    variant, d, ro = args
    circ = build(d, ro, variant)
    ler = R.decode_mwpm(circ, SHOTS)
    print(f"done {variant:10} d={d:2} ro={ro:4} LER={ler:.6f}", flush=True)
    return (variant, d, ro, ler)


def main():
    configs = [(v, d, ro) for v in VARIANTS for d in DISTANCES for ro in READOUT_LENGTHS_NS]
    print(f"surface readout-mechanism sweep: {len(configs)} configs, {SHOTS} shots")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "distance", "readout_ns", "logical_error_rate"])
        with mp.Pool(max(1, min(mp.cpu_count() - 2, 30))) as pool:
            res = []
            for r in pool.imap_unordered(run_one, configs):
                res.append(r); w.writerow(r); f.flush()
    res.sort(key=lambda x: (x[0], x[1], x[2]))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "distance", "readout_ns", "logical_error_rate"])
        w.writerows(res)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
