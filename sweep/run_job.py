"""Run one sweep job and write its result: python -m sweep.run_job <run_dir> <job_id>

A "circuit" job builds the noisy circuit of one configuration through ECCentric's pipeline and
records its resources; a "decode" job decodes that circuit with one decoder until it has seen
enough logical errors, used enough shots or spent its time budget.
"""
import json
import logging
import os
import subprocess
import sys
import time
from importlib import metadata

# Many jobs run side by side on a machine: one thread each, not a thread pool per core
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import stim

from validation.run_job import decode_adaptively

logging.disable(logging.CRITICAL)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ["stim", "qiskit", "ldpc", "PyMatching", "qldpc", "quits", "relay-bp", "tesseract-decoder", "beliefmatching"]
TWO_QUBIT = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "SWAP"}


def provenance():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "versions": versions}


def circuit_resources(circuit: stim.Circuit) -> dict:
    pairs = swaps = measurements = 0
    for inst in circuit.flattened():
        if inst.name in TWO_QUBIT:
            pairs += len(inst.targets_copy()) // 2
            swaps += len(inst.targets_copy()) // 2 if inst.name == "SWAP" else 0
    return {
        "circuit_qubits": circuit.num_qubits,
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        "two_qubit_gates": pairs,
        "swaps": swaps,
        "layers": sum(1 for inst in circuit.flattened() if inst.name == "TICK"),
        "measurements": circuit.num_measurements,
    }


def task_of(job):
    from eccentric.experiments import Task
    fields = ("backend", "code", "distance", "cycles", "decoder", "error_type", "error_probability",
              "layout_method", "routing_method", "translating_method", "schedule")
    return Task(experiment="sweep", num_samples=1, backend_size=job.get("backend_size"),
                **{f: job.get(f) for f in fields})


def circuit_job(job, run_dir):
    from eccentric.codes import get_min_n, num_rounds
    from eccentric.experiments.base import SINGLE_QPU_SIZES, load_backend, noisy_circuit, resolve_code_schedule

    task = task_of(job)
    backend = load_backend(task)
    capacity = SINGLE_QPU_SIZES.get(task.backend, backend.num_qubits)
    needed = get_min_n(task.code, task.distance)
    result = {"code_qubits": needed, "device_qubits": capacity, "schedule_used": resolve_code_schedule(task),
              "rounds": num_rounds(task.code, task.cycles), **provenance()}
    if needed > capacity:
        return {**result, "status": "skipped", "reason": f"does not fit: needs {needed} qubits, device has {capacity}"}
    start = time.time()
    circuit = noisy_circuit(task, backend, task.distance, task.cycles)
    path = os.path.join(run_dir, "circuits", f"{job['id']}.stim")
    circuit.to_file(path + ".tmp")
    os.replace(path + ".tmp", path)
    return {**result, **circuit_resources(circuit), "seconds_circuit": time.time() - start}


def decode_job(job, run_dir):
    from eccentric.decoders import decode

    # A refinement job decodes a circuit of another run directory (sweep.refine)
    circuit_file = job.get("circuit_file") or os.path.join(run_dir, "circuits", f"{job['needs']}.stim")
    circuit = stim.Circuit.from_file(circuit_file)
    partial = os.path.join(run_dir, "results", f"{job['id']}.partial")
    origin = provenance()

    def checkpoint(counts):
        # What a decode has seen so far, kept if the worker stops it at its timeout
        with open(partial + ".tmp", "w") as f:
            json.dump({**counts, **origin}, f)
        os.replace(partial + ".tmp", partial)

    result = decode_adaptively(
        job, lambda n: decode(job["code"], circuit, n, job["decoder"], job["backend"], job["error_type"]), checkpoint)
    if os.path.exists(partial):
        os.remove(partial)
    return {**result, **origin}


def main():
    run_dir, job_id = sys.argv[1], sys.argv[2]
    with open(os.path.join(run_dir, "jobs", f"{job_id}.json")) as f:
        job = json.load(f)
    start = time.time()
    try:
        result = (circuit_job if job["type"] == "circuit" else decode_job)(job, run_dir)
        status = result.pop("status", "failed" if result.get("failed") else "done")
    except Exception as e:
        result, status = {"error": f"{type(e).__name__}: {str(e)[:500]}"}, "failed"
    result.update(job=job, status=status, seconds_total=time.time() - start, host=os.uname().nodename,
                  finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = os.path.join(run_dir, "results", f"{job_id}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(result, f, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
