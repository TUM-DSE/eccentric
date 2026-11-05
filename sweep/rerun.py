"""Decode again, in a run directory of their own, the decodes of one decoder that a decoder fix changes:
python -m sweep.rerun <rerun_dir> <results.csv> <runs base dir> <decoder> [--detectors-not-multiple-of N]
                      [--also <run_dir> --started-before "<YYYY-mm-dd HH:MM:SS UTC>"]

The decodes are the rows of <decoder> in a collected table (sweep.collect) that have a result (not
pending, not "does not fit"); --detectors-not-multiple-of keeps those whose circuit's number of
detectors is not a multiple of N. --also adds the decodes of <decoder> of a run that is still going,
whose result exists and whose job started before the given time (a result file's time less the job's
seconds). Each decode job is copied unchanged except that it decodes its run's circuit file; the
circuit job's result is copied too, so that the workers and sweep.collect find the circuit. Pass the
rerun directory last to sweep.collect: its rows replace the earlier ones of the same configuration.
"""
import argparse
import calendar
import json
import os
import shutil
import time


def _copy(base, run, job_id, rerun_dir):
    with open(os.path.join(base, run, "jobs", f"{job_id}.json")) as f:
        job = json.load(f)
    if job.get("type") != "decode" or job.get("refines"):
        return False
    job["circuit_file"] = job.get("circuit_file") or os.path.join(base, run, "circuits", f"{job['needs']}.stim")
    job["reruns"] = f"{run}/{job_id}"
    with open(os.path.join(rerun_dir, "jobs", f"{job_id}.json"), "w") as f:
        json.dump(job, f)
    circuit = os.path.join(base, run, "results", f"{job['needs']}.json")
    if os.path.exists(circuit):
        shutil.copy(circuit, os.path.join(rerun_dir, "results", f"{job['needs']}.json"))
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rerun_dir")
    p.add_argument("results_csv")
    p.add_argument("base")
    p.add_argument("decoder")
    p.add_argument("--detectors-not-multiple-of", type=int)
    p.add_argument("--also")
    p.add_argument("--started-before")
    a = p.parse_args()
    import pandas as pd
    for sub in ("jobs", "claims", "results", "logs"):
        os.makedirs(os.path.join(a.rerun_dir, sub), exist_ok=True)
    df = pd.read_csv(a.results_csv, low_memory=False)
    rows = df[(df.decoder == a.decoder) & ~df.status.isin(["pending", "does not fit"]) & df.result_file.notna()]
    if a.detectors_not_multiple_of:
        rows = rows[rows.detectors % a.detectors_not_multiple_of != 0]
    copied = sum(_copy(a.base, r.run, r.job_id, a.rerun_dir) for r in rows.itertuples())
    also = 0
    if a.also:
        cutoff = calendar.timegm(time.strptime(a.started_before.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S"))
        run = os.path.basename(os.path.abspath(a.also))
        for name in os.listdir(os.path.join(a.also, "results")):
            if not name.endswith(".json") or f".decoder={a.decoder}" not in name:
                continue
            path = os.path.join(a.also, "results", name)
            with open(path) as f:
                r = json.load(f)
            started = os.path.getmtime(path) - r.get("seconds_total", 0)
            if started >= cutoff or r.get("status") == "skipped":
                continue
            with open(os.path.join(a.also, "results", f"{r['job']['needs']}.json")) as f:
                circuit = json.load(f)
            n = a.detectors_not_multiple_of
            if n and circuit.get("detectors") is not None and circuit["detectors"] % n == 0:
                continue
            if not os.path.exists(os.path.join(a.rerun_dir, "jobs", name)):
                also += _copy(os.path.dirname(os.path.abspath(a.also)), run, name[:-5], a.rerun_dir)
    print(f"{copied} decodes from {a.results_csv}, {also} more from {a.also}: {copied + also} jobs in {a.rerun_dir}")


if __name__ == "__main__":
    main()
