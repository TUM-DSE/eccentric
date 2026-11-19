"""Decode a validation circuit with the BP-OSD of ECCentric's main branch (ldpc v1 and stimbposd).

Runs in an environment with ldpc 0.1.x: python -m validation.decode_v1 <run_dir> <job_id> <bp_osd.py of main>
"""
import importlib.util
import json
import os
import sys
import time

import stim

from validation.run_job import decode_adaptively


def main():
    run_dir, job_id, module_path = sys.argv[1:4]
    with open(os.path.join(run_dir, "jobs", f"{job_id}.json")) as f:
        job = json.load(f)
    spec = importlib.util.spec_from_file_location("bp_osd_v1", module_path)
    bp_osd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp_osd)
    circuit = stim.Circuit.from_file(os.path.join(run_dir, "circuits", f"{job['code']}_p{job['p']}.stim"))
    decoder = getattr(bp_osd, job["decoder"].removesuffix("_v1"))
    start = time.time()
    try:
        result = decode_adaptively(job, lambda n: decoder(circuit, n, False))
        status = "failed" if result.get("failed") else "done"
    except Exception as e:
        result, status = {"error": f"{type(e).__name__}: {str(e)[:500]}"}, "failed"
    result.update(job=job, status=status, seconds_total=time.time() - start, host=os.uname().nodename)
    path = os.path.join(run_dir, "results", f"{job_id}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(result, f)
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
