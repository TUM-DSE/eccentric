# Usage: python <this file> <lattice-sim dir> <out.json>, where <lattice-sim dir> is synchronization-artifact/ of the
# mcm-latency-eccentric branch (git archive origin/mcm-latency-eccentric synchronization-artifact | tar -x)
"""ECCentric's tqec lattice surgery against lattice-sim (Maurya & Tannu, ISCA'25) at matched noise:
DEPOLARIZE2(p) after two-qubit gates, X flip p before measurements, DEPOLARIZE1(1e-4) after single-qubit gates,
no reset or idle error; MWPM on both; logical error per round."""
import sys, os, json, multiprocessing as mp
import numpy as np, stim, pymatching

ART = sys.argv[1]
P_LIST, D_LIST, SHOTS = (0.002, 0.004), (3, 5, 7), 400_000

def per_round(ler, rounds):
    return float("nan") if ler >= 0.5 else (1 - (1 - 2 * ler) ** (1 / rounds)) / 2

def mwpm(c, shots):
    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True))
    errs, done = 0, 0
    while done < shots:
        n = min(100_000, shots - done)
        dets, obs = c.compile_detector_sampler().sample(n, separate_observables=True)
        errs += int(np.any(m.decode_batch(dets) != obs, axis=1).sum()); done += n
    return errs / shots

def eccentric(args):
    op, d, p = args
    import eccentric.experiments.base as base
    from eccentric.noise.noise import NoiseModel
    from eccentric.codes.lattice_surgery import NUM_QUBITS
    from eccentric.codes.catalog import num_rounds
    base.get_noise_model = lambda error_type, qt, prob, backend: NoiseModel(sq=1e-4, tq=p, measure=p, reset=0, idle=0, qt=qt)
    t = base.Task(experiment="x", num_samples=1, backend="custom_full", backend_size=NUM_QUBITS[op](d), code=op,
                  distance=d, cycles=d, error_type="matched", error_probability=p)
    c = base.noisy_circuit(t, base.load_backend(t), d, d)
    ler = mwpm(c, SHOTS)
    return {"tool": "eccentric_tqec", "op": op[3:], "d": d, "p": p, "rounds": num_rounds(op, d), "ler": ler,
            "per_round": per_round(ler, num_rounds(op, d)), "qubits": c.num_qubits}

def lattice_sim(args):
    op, d, p = args
    sys.path.insert(0, os.path.join(ART, "sim"))
    from circuit_4 import circuit
    sim = circuit(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=1e12, fixed_t2=1e12,
                  fixed_cnot_latency=100, fixed_measure_latency=500, fixed_cnot_noise=p, fixed_measure_noise=p,
                  rounds_per_op=d + 1, idle_multiplier=1, basis="Z", ls_basis="X", merge=(op == "merge"),
                  ).from_string("qreg q[2];" if op == "merge" else "qreg q[1];")
    c = stim.Circuit(sim.ckt) if isinstance(sim.ckt, str) else sim.ckt
    rounds = (2 if op == "merge" else 1) * (d + 1)
    ler = mwpm(c, SHOTS)
    used = len({t.value for i in c.flattened() for t in i.targets_copy() if t.is_qubit_target})
    return {"tool": "lattice_sim", "op": op, "d": d, "p": p, "rounds": rounds, "ler": ler,
            "per_round": per_round(ler, rounds), "qubits": used}

def run(job):
    f, a = job
    return f(a)

if __name__ == "__main__":
    jobs = [(f, (o, d, p)) for p in P_LIST for d in D_LIST for f, o in
            ((eccentric, "ls_memory"), (eccentric, "ls_merge"), (lattice_sim, "memory"), (lattice_sim, "merge"))]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(16, mp_context=mp.get_context("spawn")) as pool:
        rows = list(pool.map(run, jobs))
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
    for r in sorted(rows, key=lambda r: (r["op"], r["p"], r["d"], r["tool"])):
        print(f'{r["op"]:6s} p={r["p"]} d={r["d"]} {r["tool"]:15s} qubits {r["qubits"]:4d} rounds {r["rounds"]:3d} LER {r["ler"]:.3e} per round {r["per_round"]:.3e}')
