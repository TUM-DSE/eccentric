"""Decode the weak results of a run further, split into chunks that run side by side on any machine:
python -m sweep.refine <run_dir> <refine_dir> [<max errors>]

A decode is weak if it timed out without a result, or saw fewer than <max errors> (default 20)
logical errors, or belongs to one of the paper's slices and did not reach its target. Each becomes
up to 16 chunk jobs in <refine_dir>, which decode the run's circuit with seeds the run did not use;
sweep.collect adds their shots and errors to the run's result.
"""
import json
import math
import os
import sys

# A chunk's seeds start at (chunk + 1) * SEED_STRIDE; the runs used seeds below 1000 + 1M
SEED_STRIDE = 10_000_000
CHUNK = dict(time_budget=8 * 3600, timeout=9 * 3600, first_batch=100, max_batch=5000)
PAPER_SLICES = ("paper_dqc", "paper_topology")
# Workers take cheap and important chunks first: the paper's rows, then low statistics, then timeouts,
# with Tesseract (the slowest decoder) after the others of each kind
PRIORITY = {"paper": 0, "low statistics": 1, "timeout": 3}


def weak(job, result, max_errors):
    """Why the decode should be refined, or None"""
    if result["status"] == "failed":
        return "timeout" if str(result.get("error", "")).startswith("timeout") else None
    if result["status"] != "done":
        return None
    errors = result.get("errors", 0)
    if 0 < errors < max_errors:
        return "low statistics"
    if job.get("slice") in PAPER_SLICES and errors < job["target_errors"]:
        return "paper"
    return None


def chunks(job, result):
    """How many chunks, and the shots and errors each should aim for"""
    target = job["target_errors"]
    if result["status"] == "failed":
        return 8, 1_000_000, math.ceil(target / 8) + 1
    shots, errors, seconds = result["shots"], result["errors"], result.get("seconds") or 1
    rate = max(errors, 1) / shots
    needed = math.ceil((target - errors) / rate)
    hours = needed * seconds / shots / 3600
    n = min(max(math.ceil(hours / 6), 2), 16)
    return n, min(math.ceil(1.5 * needed / n), SEED_STRIDE - 1000), math.ceil((target - errors) / n) + 1


def main():
    run_dir, refine_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    max_errors = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    for sub in ("jobs", "claims", "results", "logs"):
        os.makedirs(os.path.join(refine_dir, sub), exist_ok=True)
    counts, total = {}, 0
    for name in sorted(os.listdir(os.path.join(run_dir, "jobs"))):
        with open(os.path.join(run_dir, "jobs", name)) as f:
            job = json.load(f)
        result_path = os.path.join(run_dir, "results", name)
        if job["type"] != "decode" or not os.path.exists(result_path):
            continue
        with open(result_path) as f:
            result = json.load(f)
        reason = weak(job, result, max_errors)
        if reason is None:
            continue
        n, shots, target = chunks(job, result)
        for i in range(n):
            chunk = {k: v for k, v in job.items() if k != "needs"}
            chunk.update(CHUNK, id=f"{job['id']}.refine{i}", max_shots=shots, target_errors=target,
                         seed_offset=(i + 1) * SEED_STRIDE, refines=f"{os.path.basename(run_dir)}/{job['id']}",
                         circuit_file=os.path.join(run_dir, "circuits", f"{job['needs']}.stim"),
                         priority=PRIORITY[reason] + (1 if job["decoder"] == "tesseract" else 0))
            with open(os.path.join(refine_dir, "jobs", f"{chunk['id']}.json"), "w") as f:
                json.dump(chunk, f)
        counts[reason] = counts.get(reason, 0) + 1
        total += n
    print(f"{sum(counts.values())} decodes to refine ({counts}) in {total} chunk jobs")


if __name__ == "__main__":
    main()
