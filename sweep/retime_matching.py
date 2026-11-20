"""Re-time the matching decoders (mwpm, mwpm_corr) on the sweep's circuits, both on the same circuit, back to back
in one process, with the same number of shots:

    python -m sweep.retime_matching <results.csv> <runs base dir> <out.jsonl> [--shots 100000] [--processes 16]
    python -m sweep.retime_matching collect <out.jsonl> experiment_results/Sweep/matching_timing.csv

The sweep's own times for these decoders are dominated by per-batch overhead (a median of 6,300 shots in several
batches, and before the fix plain MWPM rebuilt its matching graph every batch), so they cannot compare decoders
that take microseconds per shot. Per configuration and decoder, this records the seconds of one decode() call
of `shots` shots (sampling, building the decoder and checking the predictions included, as the sweep times every
decoder) and of PyMatching's decode_batch alone. Configurations already in out.jsonl are skipped, so it resumes.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

DECODERS = ("mwpm", "mwpm_corr")


def _circuit_file(base, run, job_id):
    with open(os.path.join(base, run, "jobs", f"{job_id}.json")) as f:
        job = json.load(f)
    return job.get("circuit_file") or os.path.join(base, run, "circuits", f"{job['needs']}.stim")


def _time(task):
    import numpy as np
    import stim
    from eccentric.decoders import correlated_matching, mwpm as plain
    from eccentric.decoders.decode import decode

    key, config, decoders, path, shots, order = task
    circuit = stim.Circuit.from_file(path)
    out = {"key": key, **config, "shots": shots, "circuit_file": path, "host": os.uname().nodename}
    approx = config["backend"].split("_")[0] in ("variance", "real", "native") and (
        config["error_type"] == "variance" or config["error_type"].split("_")[0] == "real")
    dets, _ = circuit.compile_detector_sampler(seed=0).sample(shots, separate_observables=True)
    for decoder in (decoders if order == 0 else decoders[::-1]):
        start = time.perf_counter()
        ler = decode(config["code"], circuit, shots, decoder, config["backend"], config["error_type"],
                     distance=config.get("distance"))
        call = time.perf_counter() - start
        rec = {"call_seconds": call, "ler": ler}
        if ler is not None:
            if decoder == "mwpm":
                matcher = plain._matcher(circuit, approx)
                start = time.perf_counter()
                matcher.decode_batch(dets)
            else:
                matcher, (det_frame, _) = correlated_matching._matcher(circuit, approx, True)
                start = time.perf_counter()
                matcher.decode_batch(dets ^ det_frame, enable_correlations=True)
            rec["decode_only_seconds"] = time.perf_counter() - start
        out[decoder] = rec
    return out


def collect(jsonl: str, csv: str):
    """One row per configuration and decoder: the sweep's configuration columns, the decoder, the shots timed,
    the seconds of the whole decode call and of PyMatching alone, and the ms per shot of the call"""
    import pandas as pd
    from eccentric import dataset
    rows = []
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            for decoder in DECODERS:
                t = r.get(decoder)
                if t and t.get("ler") is not None:
                    rows.append({**{k: r.get(k) for k in dataset.CONFIG}, "decoder": decoder, "shots": r["shots"],
                                 "call_seconds": t["call_seconds"], "decode_only_seconds": t.get("decode_only_seconds"),
                                 "ms_per_shot": 1000 * t["call_seconds"] / r["shots"], "host": r["host"]})
    df = pd.DataFrame(rows).sort_values(list(dataset.CONFIG) + ["decoder"], na_position="first")
    df.to_csv(csv, index=False, float_format="%.6g")
    print(f"{len(df)} timings of {df.drop_duplicates(list(dataset.CONFIG)).shape[0]} configurations -> {csv}")


def main(argv=None):
    import sys
    import pandas as pd
    from eccentric import dataset

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "collect":
        return collect(*argv[1:3])

    p = argparse.ArgumentParser(prog="python -m sweep.retime_matching")
    p.add_argument("results")
    p.add_argument("base")
    p.add_argument("out")
    p.add_argument("--shots", type=int, default=100_000)
    p.add_argument("--processes", type=int, default=16)
    a = p.parse_args(argv)

    table = dataset.with_rates(dataset.load(a.results, check=False))
    rows = table[table.decoder.isin(DECODERS)]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    tasks = []
    for i, (cfg, g) in enumerate(rows.groupby(list(dataset.CONFIG), dropna=False, sort=True)):
        config = {k: (None if pd.isna(v) else v) for k, v in zip(dataset.CONFIG, cfg)}
        key = json.dumps(config, sort_keys=True, default=str)
        if key in done:
            continue
        first = g.iloc[0]
        tasks.append((key, config, [d for d in DECODERS if d in set(g.decoder)],
                      _circuit_file(a.base, first.run, first.job_id), a.shots, i % 2))
    print(f"{len(tasks)} configurations to time ({len(done)} done)", flush=True)
    with open(a.out, "a") as out, Pool(a.processes, maxtasksperchild=50) as pool:
        for n, rec in enumerate(pool.imap_unordered(_time, tasks), 1):
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            if n % 100 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
