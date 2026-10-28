"""Collect a sweep into one table, one row per configuration and decoder:
python -m sweep.collect <run_dir> [<run_dir> ...] <out_dir>

Of several run directories, a later one replaces the rows of the same configuration and decoder in
earlier ones (a rerun of some slices after a fix).

Writes results.csv and results.json. Each row says how far its logical error rate can be trusted:
"measured" (enough logical errors), "low statistics" (stopped by the shot or time limit),
"upper bound" (no logical error seen, the 95% bound is given), "does not fit", "decoder not
applicable" or "failed".
"""
import json
import math
import os
import sys

import pandas as pd

Z95 = 1.959964
CONFIG = ["slice", "code", "schedule", "distance", "cycles", "backend", "backend_size", "error_type",
          "error_probability", "layout_method", "routing_method", "translating_method"]
RESOURCES = ["schedule_used", "rounds", "code_qubits", "device_qubits", "circuit_qubits", "detectors", "observables",
             "two_qubit_gates", "swaps", "layers", "measurements"]


def wilson(errors, shots):
    if not shots:
        return None, None
    p = errors / shots
    denom = 1 + Z95 ** 2 / shots
    centre = (p + Z95 ** 2 / (2 * shots)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / shots + Z95 ** 2 / (4 * shots ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def per_round(ler, rounds):
    if ler is None or rounds is None or ler >= 0.5:
        return None
    return (1 - (1 - 2 * ler) ** (1 / rounds)) / 2


def load(run_dir):
    results = {}
    for name in os.listdir(os.path.join(run_dir, "results")):
        if name.endswith(".json"):
            with open(os.path.join(run_dir, "results", name)) as f:
                r = json.load(f)
            results[r["job"]["id"]] = r
    return results


def superseded(run_dir):
    """Results that sweep.extend set aside when it queued their decodes again with more shots"""
    results = {}
    path = os.path.join(run_dir, "superseded")
    for name in os.listdir(path) if os.path.isdir(path) else []:
        if name.startswith("results.") and name.endswith(".json"):
            with open(os.path.join(path, name)) as f:
                r = json.load(f)
            results[r["job"]["id"]] = r
    return results


def rows(run_dir):
    results, earlier = load(run_dir), superseded(run_dir)
    for name in sorted(os.listdir(os.path.join(run_dir, "jobs"))):
        with open(os.path.join(run_dir, "jobs", name)) as f:
            job = json.load(f)
        if job["type"] != "decode":
            continue
        circuit = results.get(job["needs"], {})
        decode = results.get(job["id"])
        if decode is None and job["id"].endswith(".long"):
            # Until the longer decode finishes, keep the result it is to replace
            decode = earlier.get(job["id"][:-len(".long")])
        row = {k: job.get(k) for k in CONFIG}
        row["decoder"] = job["decoder"]
        row.update({k: circuit.get(k) for k in RESOURCES})
        row["commit"] = (decode or circuit).get("commit")
        if decode is None:
            row["status"] = "pending"
        elif circuit.get("status") == "skipped":
            row["status"] = "does not fit"
            row["note"] = circuit.get("reason")
        elif decode["status"] == "failed":
            row["status"] = "decoder not applicable" if decode.get("failed") else "failed"
            row["note"] = decode.get("error")
        else:
            shots, errors = decode["shots"], decode["errors"]
            low, high = wilson(errors, shots)
            row.update(shots=shots, errors=errors, decode_seconds=decode.get("seconds"))
            if errors == 0:
                row.update(status="upper bound", logical_error_rate=None, ler_high=high)
            else:
                row.update(status="measured" if errors >= job["target_errors"] else "low statistics",
                           logical_error_rate=errors / shots, ler_low=low, ler_high=high)
            row["ler_per_round"] = per_round(row.get("logical_error_rate"), circuit.get("rounds"))
            row["ler_per_round_high"] = per_round(row.get("ler_high"), circuit.get("rounds"))
        yield row


def main():
    run_dirs, out_dir = sys.argv[1:-1], sys.argv[-1]
    os.makedirs(out_dir, exist_ok=True)
    df = pd.concat([pd.DataFrame(rows(run_dir)) for run_dir in run_dirs], ignore_index=True)
    df = df.drop_duplicates(subset=CONFIG + ["decoder"], keep="last")
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    df.to_json(os.path.join(out_dir, "results.json"), orient="records", indent=1)
    print(df.groupby(["slice", "status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
