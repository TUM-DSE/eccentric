"""Count the work behind the results the website shows:
python -m sweep.effort <out.json> <collected results.csv> <run_dir> [<run_dir> ...]

Counts the rows of sweep.collect's table that hold a logical error rate or bound, and for each the
decode whose result the row holds, the refinement chunks that added their shots to it, and the circuit
it decoded (once, though several decoders share it). Jobs that failed or were stopped without counts,
and results that later runs replaced, are left out. Each job ran on one core.
"""
import csv
import datetime
import json
import os
import sys

COUNTED = ("measured", "low statistics", "upper bound")


def load(run_dir, path):
    with open(os.path.join(run_dir, path)) as f:
        return json.load(f)


def chunks(run_dirs):
    """The finished refinement chunks, keyed by the run dir name/job id of the decode they refine"""
    added = {}
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "results")
        for name in os.listdir(path) if os.path.isdir(path) else []:
            if name.endswith(".json"):
                r = load(path, name)
                base = r.get("job", {}).get("refines")
                if base and r.get("status") == "done" and r.get("shots"):
                    added.setdefault(base, []).append(r)
    return added


def main():
    out, table, run_dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    runs = {os.path.basename(os.path.abspath(d)): d for d in run_dirs}
    added = chunks(run_dirs)
    with open(table, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["status"] in COUNTED]
    jobs, circuits = [], set()
    for row in rows:
        run_dir = runs[row["run"]]
        decode = load(run_dir, row["result_file"])
        # A decode that timed out adds nothing; its chunks hold the row's shots
        if decode["status"] == "done":
            jobs.append(decode)
        jobs += added.get(f"{row['run']}/{row['job_id']}", [])
        needs = decode["job"]["needs"]
        if (row["run"], needs) not in circuits:
            circuits.add((row["run"], needs))
            jobs.append(load(run_dir, f"results/{needs}.json"))
    seconds = sum(r.get("seconds_total") or r.get("seconds") or 0 for r in jobs)
    effort = {"updated": datetime.date.today().isoformat(), "results": len(rows), "jobs": len(jobs),
              "cpu_hours": round(seconds / 3600), "shots": sum(r.get("shots") or 0 for r in jobs),
              "logical_errors": sum(r.get("errors") or 0 for r in jobs),
              "codes": len({row["code"] for row in rows}), "devices": len({row["backend"] for row in rows}),
              "noise_models": len({row["error_type"] for row in rows}),
              "decoders": len({row["decoder"] for row in rows})}
    with open(out, "w") as f:
        json.dump(effort, f, indent=1)
    print(effort)


if __name__ == "__main__":
    main()
