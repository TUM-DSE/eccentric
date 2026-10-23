"""Run one validation job and write its result: python -m validation.run_job <run_dir> <job_id>"""
import json
import logging
import os
import sys
import time

import numpy as np
import stim

logging.disable(logging.CRITICAL)


def seeded_samplers(seed_box):
    """Make every detector sampler deterministic, so all decoders see the same shots"""
    circuit_sampler = stim.Circuit.compile_detector_sampler
    dem_sampler = stim.DetectorErrorModel.compile_sampler
    stim.Circuit.compile_detector_sampler = lambda self, *a, **k: circuit_sampler(self, seed=seed_box[0])
    stim.DetectorErrorModel.compile_sampler = lambda self, *a, **k: dem_sampler(self, seed=seed_box[0])


def with_sd6_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """SD6-like noise straight on the Stim circuit, for the circuit-distance search"""
    noisy = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name in ("M", "MX", "MY", "MR", "MRX"):
            noisy.append("X_ERROR" if inst.name in ("M", "MR") else "Z_ERROR", inst.targets_copy(), p)
        noisy.append(inst)
        if inst.name in ("CX", "CZ", "XCX", "CY", "SWAP"):
            noisy.append("DEPOLARIZE2", inst.targets_copy(), p)
        elif inst.name in ("H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "X", "Y", "Z"):
            noisy.append("DEPOLARIZE1", inst.targets_copy(), p)
        elif inst.name in ("R", "RX", "MR", "MRX"):
            noisy.append("X_ERROR" if inst.name in ("R", "MR") else "Z_ERROR", inst.targets_copy(), p)
    return noisy


def circuit_check(job):
    from eccentric.codes import get_min_n
    from eccentric.codes.catalog import make_circuit
    from validation.specs import expected_observables, test_distance

    code, schedule = job["code"], job["schedule"]
    d = test_distance(code) if job.get("distance") is None else job["distance"]
    circuit = make_circuit(code, d, 2, schedule)
    result = {"num_qubits": circuit.num_qubits, "expected_num_qubits": get_min_n(code, d)}
    try:
        circuit.detector_error_model()
        result["deterministic"] = True
    except ValueError as e:
        result["deterministic"] = False
        result["error"] = str(e)[:300]
    used = {inst.gate_args_copy()[0] for inst in circuit.flattened() if inst.name == "OBSERVABLE_INCLUDE"}
    result["observables"] = len(used)
    result["expected_observables"] = expected_observables(code)
    result["layers"] = sum(1 for inst in circuit if inst.name == "TICK")
    return result


def code_distance(job):
    from qldpc import codes
    from quits.qldpc_code import QldpcCode
    from quits.simulation import estimate_spacetime_distance
    from validation.specs import css_matrices

    hx, hz = css_matrices(job["code"])
    n = hx.shape[1]
    if n <= job["exact_up_to"]:
        return {"n": int(n), "distance": int(codes.CSSCode(hx, hz).get_distance()), "method": "exact"}
    if not job["trials"]:
        return {"n": int(n), "distance": None, "method": f"too large for an exact computation (n > {job['exact_up_to']})"}
    code = QldpcCode.from_parity_checks(hz, hx)
    dx = estimate_spacetime_distance(code.hz, code.lz, num_trials=job["trials"], osd_order=4)
    dz = estimate_spacetime_distance(code.hx, code.lx, num_trials=job["trials"], osd_order=4)
    return {"n": int(n), "distance": int(min(dx, dz)), "method": f"estimate ({job['trials']} trials)"}


def circuit_distance(job):
    from eccentric.codes.catalog import make_circuit
    from validation.specs import test_distance

    code = job["code"]
    d = test_distance(code)
    circuit = with_sd6_noise(make_circuit(code, d, d, job["schedule"]), 1e-3)
    try:
        errors = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=job["max_detection_events"],
            dont_explore_edges_with_degree_above=job["max_detection_events"],
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True,
        )
    except ValueError:
        # Nothing found within the search limits, which says nothing about the distance
        return {"distance": d, "circuit_distance_upper_bound": None}
    return {"distance": d, "circuit_distance_upper_bound": len(errors)}


def noisy(job, run_dir):
    from eccentric.codes import get_min_n
    from eccentric.experiments import Task
    from eccentric.experiments.base import load_backend, noisy_circuit
    from validation.specs import test_distance

    code, p = job["code"], job["p"]
    d = test_distance(code)
    task = Task(experiment="validation", num_samples=1, backend="custom_full", backend_size=get_min_n(code, d),
                code=code, distance=d, cycles=job["cycles"], error_type="sd6", error_probability=p)
    circuit = noisy_circuit(task, load_backend(task), d, job["cycles"])
    path = os.path.join(run_dir, "circuits", f"{code}_p{p}.stim")
    circuit.to_file(path + ".tmp")
    os.replace(path + ".tmp", path)
    return {"distance": d, "num_qubits": circuit.num_qubits, "num_detectors": circuit.num_detectors}


def decode_adaptively(job, decode_batch):
    seed_box = [0]
    seeded_samplers(seed_box)
    shots = errors = 0
    batch, start = job["first_batch"], time.time()
    while errors < job["target_errors"] and shots < job["max_shots"] and time.time() - start < job["time_budget"]:
        seed_box[0] = 1000 + shots
        ler = decode_batch(batch)
        if ler is None:
            return {"failed": True, "shots": shots}
        errors += int(round(ler * batch))
        shots += batch
        batch = min(2 * batch, job["max_batch"], job["max_shots"] - shots) or batch
    return {"shots": shots, "errors": errors, "logical_error_rate": errors / shots if shots else None,
            "seconds": time.time() - start}


def decode(job, run_dir):
    from eccentric.decoders import decode as eccentric_decode

    circuit = stim.Circuit.from_file(os.path.join(run_dir, "circuits", f"{job['code']}_p{job['p']}.stim"))
    return decode_adaptively(job, lambda n: eccentric_decode(job["code"], circuit, n, job["decoder"], "custom_full", "sd6"))


JOBS = {"circuit": circuit_check, "code_distance": code_distance, "circuit_distance": circuit_distance}


def main():
    run_dir, job_id = sys.argv[1], sys.argv[2]
    with open(os.path.join(run_dir, "jobs", f"{job_id}.json")) as f:
        job = json.load(f)
    start = time.time()
    try:
        if job["type"] in JOBS:
            result = JOBS[job["type"]](job)
        elif job["type"] == "noisy":
            result = noisy(job, run_dir)
        else:
            result = decode(job, run_dir)
        status = "failed" if result.get("failed") else "done"
    except Exception as e:
        result, status = {"error": f"{type(e).__name__}: {str(e)[:500]}"}, "failed"
    result.update(job=job, status=status, seconds_total=time.time() - start, host=os.uname().nodename)
    path = os.path.join(run_dir, "results", f"{job_id}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(result, f, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
