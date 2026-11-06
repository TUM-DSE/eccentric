"""Decode circuits of earlier runs with another decoder, in a run directory of its own:
python -m sweep.add_decoder <run_dir> <results.csv> <runs base dir> <decoder> --query "<pandas query>"
                            [--missing] [--priority P] [--sample N] [--seed S] [--set key=value ...]

The circuits are those of the rows of a collected table (sweep.collect) that the query selects, one per
circuit. Each gets a decode job of <decoder>: a copy of the selected row's decode job with the decoder
replaced and the --set decode settings (time_budget, timeout, max_shots, target_errors, ...), decoding
its run's circuit file; the circuit job's result is copied too, so that the workers and sweep.collect
find the circuit. Pass the run directory to sweep.collect: its rows add the decoder to those
configurations. Calling it again on the same run directory adds more jobs. --missing keeps only the
configurations that have no row of <decoder> in the table (the decoder fill-in); --priority sets the
jobs' priority (workers take lower numbers first).
"""
import argparse
import json
import os
import re
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("results_csv")
    p.add_argument("base")
    p.add_argument("decoder")
    p.add_argument("--query", required=True)
    p.add_argument("--sample", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--missing", action="store_true")
    p.add_argument("--priority", type=int, default=1)
    a = p.parse_args()
    import pandas as pd
    for sub in ("jobs", "claims", "results", "logs"):
        os.makedirs(os.path.join(a.run_dir, sub), exist_ok=True)
    df = pd.read_csv(a.results_csv, low_memory=False)
    rows = df.query(a.query, engine="python")
    rows = rows[rows.job_id.notna()].assign(circuit=lambda r: r.job_id.str.replace(r"\.decoder=.*$", "", regex=True))
    rows = rows[rows.decoder != a.decoder]
    if a.missing:
        from sweep.collect import CONFIG
        columns = [c for c in CONFIG if c in df.columns]
        key = lambda d: d[columns].astype(str).agg("|".join, axis=1)
        have = set(key(df[df.decoder == a.decoder]))
        rows = rows.assign(config=key(rows))
        rows = rows[~rows.config.isin(have)].drop_duplicates("config")
    else:
        rows = rows.drop_duplicates(["run", "circuit"])
    if a.sample and len(rows) > a.sample:
        rows = rows.sample(a.sample, random_state=a.seed)
    settings = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        settings[k] = json.loads(v)
    added = 0
    for r in rows.itertuples():
        with open(os.path.join(a.base, r.run, "jobs", f"{r.job_id}.json")) as f:
            job = json.load(f)
        if job.get("type") != "decode" or job.get("refines"):
            continue
        new_id = re.sub(r"\.decoder=[^.]+", f".decoder={a.decoder}", job["id"])
        if os.path.exists(os.path.join(a.run_dir, "jobs", f"{new_id}.json")):
            continue
        job.update(settings, id=new_id, decoder=a.decoder, priority=a.priority, added_from=f"{r.run}/{r.job_id}",
                   circuit_file=job.get("circuit_file") or os.path.join(a.base, r.run, "circuits", f"{job['needs']}.stim"))
        with open(os.path.join(a.run_dir, "jobs", f"{new_id}.json"), "w") as f:
            json.dump(job, f)
        circuit = os.path.join(a.base, r.run, "results", f"{job['needs']}.json")
        if os.path.exists(circuit):
            shutil.copy(circuit, os.path.join(a.run_dir, "results", f"{job['needs']}.json"))
        added += 1
    print(f"{added} {a.decoder} decode jobs added to {a.run_dir} ({len(rows)} circuits selected)")


if __name__ == "__main__":
    main()
