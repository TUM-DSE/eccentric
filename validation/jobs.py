"""Create a validation run: python -m validation.jobs <run_dir> <bp_osd.py of the main branch>

Phase 1 checks every code circuit (and schedule), the code distance and the circuit-level distance.
Phase 3 decodes the same noisy circuits with every decoder, including the BP-OSD of the main branch.
"""
import json
import os
import shutil
import sys

from eccentric.codes.catalog import CODES
from validation.specs import DECODERS, ERROR_RATES, PAPER_CODES, css_matrices

DECODE = dict(target_errors=100, max_shots=100_000, time_budget=1800, first_batch=100, max_batch=5000, timeout=7200)


def make_jobs():
    jobs = []
    for code, spec in CODES.items():
        for schedule in spec.schedules or (None,):
            tag = f"{code}.{schedule}" if schedule else code
            jobs.append(dict(id=f"circuit.{tag}", type="circuit", code=code, schedule=schedule, priority=0, timeout=3600))
            jobs.append(dict(id=f"circuit_distance.{tag}", type="circuit_distance", code=code, schedule=schedule,
                             max_detection_events=5, priority=1, timeout=7200))
        if css_matrices(code) is not None:
            jobs.append(dict(id=f"code_distance_exact.{code}", type="code_distance", code=code, exact_up_to=150,
                             trials=0, priority=1, timeout=10800))
            jobs.append(dict(id=f"code_distance_estimate.{code}", type="code_distance", code=code, exact_up_to=0,
                             trials=2000, priority=1, timeout=10800))
    for code in list(CODES) + list(PAPER_CODES):
        for p in ERROR_RATES:
            noisy = f"noisy.{code}.p{p}"
            jobs.append(dict(id=noisy, type="noisy", code=code, p=p, cycles=3, priority=0, timeout=7200))
            for decoder in [d for d in DECODERS if d != "bposd"] + ["bposd_batch", "bposd_chk"]:
                jobs.append(dict(id=f"decode.{code}.p{p}.{decoder}", type="decode", code=code, p=p, decoder=decoder,
                                 needs=noisy, priority=2, **DECODE))
            for decoder in ["bposd_batch_v1", "bposd_chk_v1"]:
                jobs.append(dict(id=f"decode_v1.{code}.p{p}.{decoder}", type="decode_v1", code=code, p=p, decoder=decoder,
                                 needs=noisy, priority=2, **DECODE))
    return jobs


def main():
    run_dir, v1_module = sys.argv[1], sys.argv[2]
    for sub in ("jobs", "claims", "results", "logs", "circuits"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    shutil.copy(v1_module, os.path.join(run_dir, "bp_osd_v1.py"))
    jobs = make_jobs()
    for job in jobs:
        with open(os.path.join(run_dir, "jobs", f"{job['id']}.json"), "w") as f:
            json.dump(job, f)
    print(f"{len(jobs)} jobs in {run_dir}")


if __name__ == "__main__":
    main()
