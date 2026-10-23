"""Run validation jobs from a shared run directory, on as many machines as share it.

Each job is claimed by atomically creating claims/<job_id>, so workers on different machines never
run the same job. Decoding jobs wait until the noisy circuit they decode exists.

    python -m validation.worker <run_dir> <parallel jobs> [<python of the ldpc v1 environment>]
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_jobs(run_dir):
    jobs = []
    for name in sorted(os.listdir(os.path.join(run_dir, "jobs"))):
        with open(os.path.join(run_dir, "jobs", name)) as f:
            jobs.append(json.load(f))
    return sorted(jobs, key=lambda j: j["priority"])


def done(run_dir, job_id):
    return os.path.exists(os.path.join(run_dir, "results", f"{job_id}.json"))


def write_failure(run_dir, job, reason):
    path = os.path.join(run_dir, "results", f"{job['id']}.json")
    with open(path + ".tmp", "w") as f:
        json.dump({"job": job, "status": "failed", "error": reason, "host": os.uname().nodename}, f)
    os.replace(path + ".tmp", path)


def ready(run_dir, job):
    """None if the job can run now, else why not: 'wait' or a failure reason"""
    needs = job.get("needs")
    if not needs:
        return None
    if not done(run_dir, needs):
        return "wait"
    with open(os.path.join(run_dir, "results", f"{needs}.json")) as f:
        if json.load(f)["status"] != "done":
            return f"needed job {needs} failed"
    return None


def claim(run_dir, job):
    try:
        os.mkdir(os.path.join(run_dir, "claims", job["id"]))
        return True
    except FileExistsError:
        return False


def run(run_dir, job, v1_python):
    if job["type"] == "decode_v1":
        cmd = [v1_python, "-m", "validation.decode_v1", run_dir, job["id"], os.path.join(run_dir, "bp_osd_v1.py")]
    else:
        cmd = [sys.executable, "-m", "validation.run_job", run_dir, job["id"]]
    log = open(os.path.join(run_dir, "logs", f"{job['id']}.log"), "w")
    try:
        subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=job["timeout"])
        if not done(run_dir, job["id"]):
            write_failure(run_dir, job, "crashed without a result, see logs/")
    except subprocess.TimeoutExpired:
        write_failure(run_dir, job, f"timeout after {job['timeout']} s")


def worker_loop(run_dir, jobs, v1_python):
    while True:
        finished = set(n[:-5] for n in os.listdir(os.path.join(run_dir, "results")) if n.endswith(".json"))
        claimed = set(os.listdir(os.path.join(run_dir, "claims")))
        open_jobs = [j for j in jobs if j["id"] not in finished and j["id"] not in claimed
                     and (j["type"] != "decode_v1" or v1_python)]
        if not open_jobs:
            return
        progressed = False
        for job in open_jobs:
            state = ready(run_dir, job)
            if state == "wait" or not claim(run_dir, job):
                continue
            if state is not None:
                write_failure(run_dir, job, state)
            else:
                run(run_dir, job, v1_python)
            progressed = True
            break
        if not progressed:
            time.sleep(30)


def main():
    run_dir, parallel = sys.argv[1], int(sys.argv[2])
    v1_python = sys.argv[3] if len(sys.argv) > 3 else None
    jobs = load_jobs(run_dir)
    with ThreadPoolExecutor(parallel) as pool:
        for _ in range(parallel):
            pool.submit(worker_loop, run_dir, jobs, v1_python)


if __name__ == "__main__":
    main()
