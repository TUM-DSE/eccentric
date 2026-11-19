"""X-basis runs of configurations that have only Z-basis results, in a run directory of its own:
python -m sweep.add_basis <run_dir> <results.csv> <runs base dir> [--below 0.4] [--near 1.5]
                          [--time-budget 3600] [--timeout 10800] [--long 1800] [--dry-run]

From a collected table (sweep.collect), every configuration of a code that has an X-basis circuit that
is below threshold in the Z basis (best logical error rate under --below) and has no X-basis result yet
gets an X-basis circuit job and decode jobs for two of its decoders: the most accurate in Z, and the
cheapest per shot among those within --near times its rate. The jobs belong to the basis_x slice (the
original slice is kept as origin_slice), copy the Z decode job's settings with the time budget and
timeout capped, and decodes that took longer than --long seconds in Z get priority 1 (workers take them
before the others, priority 2), so that the longest start first. Pass the run directory to
sweep.collect after the others.
"""
import argparse
import json
import os

import numpy as np

from sweep.jobs import KEY, config_id

X_CODES_EXCLUDED = ("steane", "repetition", "honeycomb", "planar_honeycomb")
RATED = ("measured", "low statistics")
SETTINGS = ("target_errors", "max_shots", "time_budget", "first_batch", "max_batch", "timeout")


def has_x_circuit(code):
    return code not in X_CODES_EXCLUDED and "hyperbolic_floquet" not in code


def select(df, below, near):
    """(config row, [decoder rows]) for the configurations to run in X"""
    import pandas as pd

    df = df.assign(basis=df.basis.fillna("Z"))
    pair = ["code", "distance", "cycles", "backend", "backend_size", "error_type", "error_probability",
            "schedule", "layout_method", "routing_method", "translating_method"]
    key = lambda d: d[pair].apply(lambda c: c.map(str)).agg("|".join, axis=1)
    done_x = set(key(df[df.basis == "X"]))
    z = df[(df.basis == "Z") & (df.slice != "basis_x") & df.code.map(has_x_circuit) & df.status.isin(RATED)
           & df.job_id.notna() & (df.shots > 0)].copy()
    z["key"] = key(z)
    z = z[~z.key.isin(done_x)]
    z["ms"] = z.decode_seconds / z.shots
    out = []
    for k, g in z.groupby("key", sort=True):
        best = g.loc[g.logical_error_rate.idxmin()]
        if not best.logical_error_rate < below:
            continue
        good = g[(g.logical_error_rate <= near * best.logical_error_rate) & g.ms.notna()]
        rows = [best] + ([good.loc[good.ms.idxmin()]] if len(good) else [])
        out.append((best, list({r.decoder: r for r in rows}.values())))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("results_csv")
    p.add_argument("base")
    p.add_argument("--below", type=float, default=0.4)
    p.add_argument("--near", type=float, default=1.5)
    p.add_argument("--time-budget", type=int, default=3600)
    p.add_argument("--timeout", type=int, default=3 * 3600)
    p.add_argument("--long", type=float, default=1800)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    import pandas as pd

    df = pd.read_csv(a.results_csv, low_memory=False)
    chosen = select(df, a.below, a.near)
    if not a.dry_run:
        for sub in ("jobs", "claims", "results", "logs", "circuits"):
            os.makedirs(os.path.join(a.run_dir, sub), exist_ok=True)
    circuits = decodes = 0
    expected = 0.0
    seen = set()
    for best, rows in chosen:
        config = {k: (None if pd.isna(best[k]) else best[k]) for k in KEY if k in best and k != "basis"}
        for k in ("distance", "cycles", "backend_size"):
            if config.get(k) is not None:
                config[k] = int(config[k])
        config["basis"] = "X"
        cid = config_id("basis_x", config)
        if cid in seen:
            continue
        seen.add(cid)
        circuit = dict(id=f"circuit.{cid}", type="circuit", module="sweep.run_job", slice="basis_x", priority=0,
                       timeout=4 * 3600, origin_slice=best.slice, **config)
        jobs = [circuit]
        for r in rows:
            with open(os.path.join(a.base, r.run, "jobs", f"{r.job_id}.json")) as f:
                z_job = json.load(f)
            settings = {s: z_job[s] for s in SETTINGS if s in z_job}
            settings["time_budget"] = min(settings.get("time_budget", a.time_budget), a.time_budget)
            settings["timeout"] = min(settings.get("timeout", a.timeout), a.timeout)
            # The usual target, also where the Z row came from a refinement chunk with a smaller one
            settings["target_errors"] = 100
            seconds = 0.0 if pd.isna(r.decode_seconds) else float(r.decode_seconds)
            long = seconds > a.long
            jobs.append(dict(id=f"decode.{cid}.decoder={r.decoder}", type="decode", module="sweep.run_job",
                             slice="basis_x", needs=circuit["id"], decoder=r.decoder, priority=1 if long else 2,
                             origin_slice=best.slice, **config, **settings))
            expected += min(seconds, settings["timeout"])
            decodes += 1
        circuits += 1
        if not a.dry_run:
            for job in jobs:
                with open(os.path.join(a.run_dir, "jobs", f"{job['id']}.json"), "w") as f:
                    json.dump(job, f)
    print(f"{circuits} X-basis configurations, {decodes} decodes, about {expected / 3600:.0f} CPU-h by the Z decodes"
          + (" (dry run)" if a.dry_run else f" in {a.run_dir}"))


if __name__ == "__main__":
    main()
