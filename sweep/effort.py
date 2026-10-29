"""Count what the sweeps cost, for the website: python -m sweep.effort <out.json> <run_dir> [<run_dir> ...]

Every job that finished left a result with its run time (one core per job), including results that
later runs replaced (superseded/). Jobs that were stopped without a result are not counted, so the
totals are lower bounds.
"""
import datetime
import json
import os
import sys


def results(run_dir):
    for sub, prefix in (("results", ""), ("superseded", "results.")):
        path = os.path.join(run_dir, sub)
        for name in os.listdir(path) if os.path.isdir(path) else []:
            if name.startswith(prefix) and name.endswith(".json"):
                with open(os.path.join(path, name)) as f:
                    yield json.load(f)


def main():
    out, run_dirs = sys.argv[1], sys.argv[2:]
    jobs = seconds = shots = errors = 0
    codes, devices, noise, decoders = set(), set(), set(), set()
    for run_dir in run_dirs:
        for r in results(run_dir):
            job = r.get("job", {})
            jobs += 1
            seconds += r.get("seconds_total") or r.get("seconds") or 0
            shots += r.get("shots") or 0
            errors += r.get("errors") or 0
            if r.get("status") in ("done", "skipped") and job.get("code"):
                codes.add(job["code"])
                devices.add(job.get("backend"))
                noise.add(job.get("error_type"))
                if job.get("decoder"):
                    decoders.add(job["decoder"])
    effort = {"updated": datetime.date.today().isoformat(), "jobs": jobs, "cpu_hours": round(seconds / 3600),
              "shots": shots, "logical_errors": errors, "codes": len(codes), "devices": len(devices),
              "noise_models": len(noise), "decoders": len(decoders)}
    with open(out, "w") as f:
        json.dump(effort, f, indent=1)
    print(effort)


if __name__ == "__main__":
    main()
