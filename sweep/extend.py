"""Decode again, with more shots and time, the decodes of a run that timed out or stopped before
reaching their target number of logical errors: python -m sweep.extend <run_dir>

Each such decode gets a new job "<id>.long"; the old job and result move to superseded/, and the
old claim stays so that workers started earlier skip the old job.
"""
import json
import os
import sys

LONG = dict(max_shots=1_000_000, time_budget=4 * 3600, timeout=6 * 3600)


def needs_more(result):
    job = result["job"]
    if job["type"] != "decode" or job.get("max_shots", 0) >= LONG["max_shots"]:
        return False
    if result["status"] == "failed":
        return str(result.get("error", "")).startswith("timeout")
    return result["status"] == "done" and 0 < result.get("errors", 0) < job["target_errors"]


def main():
    run_dir = sys.argv[1]
    os.makedirs(os.path.join(run_dir, "superseded"), exist_ok=True)
    count = 0
    for name in sorted(os.listdir(os.path.join(run_dir, "results"))):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(run_dir, "results", name)) as f:
            result = json.load(f)
        if not needs_more(result):
            continue
        job = result["job"]
        long_job = {**job, **LONG, "id": job["id"] + ".long"}
        with open(os.path.join(run_dir, "jobs", long_job["id"] + ".json"), "w") as f:
            json.dump(long_job, f)
        for sub in ("jobs", "results"):
            os.replace(os.path.join(run_dir, sub, job["id"] + ".json"),
                       os.path.join(run_dir, "superseded", f"{sub}.{job['id']}.json"))
        count += 1
    print(f"{count} decodes queued again with {LONG}")


if __name__ == "__main__":
    main()
