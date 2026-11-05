"""Circuits from outside ECCentric's pipeline, as decode jobs for ECCentric's sweep worker, so that they are
decoded and timed exactly like Matchmaker's training data:

- Stim's generated memory circuits (rotated and unrotated surface code, color code, repetition code) with
  uniform circuit noise p after every Clifford, reset and data round and before every measurement;
- decoder-bench's circuits (Maurya et al., IISWC 2025: surface and color code d = 7, bivariate bicycle
  d = 6 and 12) with its SI1000 noise, built by ECCentric's analysis/crosscheck_decoder_bench.py.

Usage, in ECCentric's environment on the cluster (decoder-bench's generators need igraph):
    PYTHONPATH=<qec-matchmaker>:<eccentric>/analysis python make_external.py <run_dir> <decoder-bench checkout>
then run the jobs with ECCentric's start_sweep.sh <run_dir> <n>, and predict_external.py reads the results.
MWPM runs where the errors decompose into edges and Chromobius where detectors carry colors, as
Matchmaker's applicable() says; every circuit also gets the BP-type decoders and Tesseract.
"""
import json
import os
import sys

import stim

from qec_matchmaker.decoders import applicable
from qec_matchmaker.features import features_from_circuit

DECODE = dict(target_errors=100, max_shots=200_000, time_budget=600, first_batch=100, max_batch=5000, timeout=1800)
ALWAYS = ["bposd", "bplsd", "belief_find", "relay_bp", "tesseract"]


def stim_circuits():
    def gen(task, d, p):
        return stim.Circuit.generated(task, distance=d, rounds=d, after_clifford_depolarization=p,
                                      after_reset_flip_probability=p, before_measure_flip_probability=p,
                                      before_round_data_depolarization=p)

    for d in (3, 5, 7, 9, 11):
        for p in (0.001, 0.003, 0.007):
            if d >= 9 and p == 0.001:
                continue  # far below threshold: no errors within the budget
            yield f"stim_surface_rotated_d{d}_p{p}", "surface", gen("surface_code:rotated_memory_z", d, p)
    for d in (3, 5, 7):
        yield f"stim_surface_unrotated_d{d}_p0.003", "surface", gen("surface_code:unrotated_memory_z", d, 0.003)
    for d in (3, 5, 7, 9):
        for p in (0.001, 0.003):
            yield f"stim_color_d{d}_p{p}", "color", gen("color_code:memory_xyz", d, p)
    for d in (5, 11, 21):
        yield f"stim_repetition_d{d}_p0.01", "repetition", gen("repetition_code:memory", d, 0.01)


def decoder_bench_circuits(checkout):
    from crosscheck_decoder_bench import CIRCUITS, circuit_of

    for code, d, p in CIRCUITS:
        yield f"decoder_bench_{code}_d{d}_p{p}", {"bb": "bb72" if d == 6 else "gross"}.get(code, code), circuit_of(checkout, code, d, p)


def main():
    run_dir, checkout = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    for sub in ("jobs", "claims", "results", "logs", "circuits"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    index, jobs = {}, 0
    for name, code, circuit in list(stim_circuits()) + list(decoder_bench_circuits(checkout)):
        path = os.path.join(run_dir, "circuits", f"{name}.stim")
        circuit.to_file(path)
        ok = applicable(features_from_circuit(circuit))
        decoders = ALWAYS + [d for d in ("mwpm", "chromobius") if ok[d]]
        index[name] = {"code": code, "circuit_file": path, "decoders": decoders}
        for decoder in decoders:
            job = dict(id=f"decode.external.{name}.decoder={decoder}", type="decode", module="sweep.run_job",
                       slice="external", priority=2 if decoder == "tesseract" else 1, code=code, decoder=decoder,
                       backend="custom_full", error_type="sd6", circuit_file=path, **DECODE)
            with open(os.path.join(run_dir, "jobs", f"{job['id']}.json"), "w") as f:
                json.dump(job, f)
            jobs += 1
    with open(os.path.join(run_dir, "circuits.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"{len(index)} circuits, {jobs} decode jobs in {run_dir}")


if __name__ == "__main__":
    main()
