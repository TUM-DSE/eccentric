"""Keeping Matchmaker's data up to date with ECCentric.

The feature store is a JSON-lines file, one record per circuit file: its key (`<run>/<circuit job id>`),
size, modification time, feature version and features. Extraction only computes circuits that are new,
changed or of an older feature version, so it can run after every ECCentric sweep update; a later record
of the same key replaces an earlier one.
"""
import json
import os
import time
from multiprocessing import Pool

import stim

from .detector_types import detector_type_features
from .features import FEATURE_VERSION, features_from_circuit


def load_store(path: str) -> dict:
    """{key: record} of a feature store, the last record of each key winning"""
    records = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r["key"]] = r
    return records


def features_of_store(records: dict) -> dict:
    """{key: features} of the store's current-version records that have features"""
    return {k: r["features"] for k, r in records.items() if r.get("version") == FEATURE_VERSION and "features" in r}


def _circuit_files(run_dirs, keys=None):
    """(key, path) of the circuit files of the run directories: their circuits/, and the circuit files of other
    runs that their decode jobs name (reruns and added decoders, sweep.rerun and sweep.add_decoder), keyed under
    the run that decoded them"""
    for run_dir in run_dirs:
        run = os.path.basename(os.path.abspath(run_dir))
        folder = os.path.join(run_dir, "circuits")
        seen = set()
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if name.endswith(".stim") and (keys is None or f"{run}/{name[:-5]}" in keys):
                seen.add(name[:-5])
                yield f"{run}/{name[:-5]}", os.path.join(folder, name)
        jobs = os.path.join(run_dir, "jobs")
        for name in sorted(os.listdir(jobs)) if os.path.isdir(jobs) else []:
            if not name.startswith("decode."):
                continue
            with open(os.path.join(jobs, name)) as f:
                job = json.load(f)
            needs, path = job.get("needs"), job.get("circuit_file")
            if needs and path and needs not in seen and (keys is None or f"{run}/{needs}" in keys) and os.path.exists(path):
                seen.add(needs)
                yield f"{run}/{needs}", path


def _extract(item):
    key, path, stat, old = item
    start = time.time()
    try:
        circuit = stim.Circuit.from_file(path)
        # A record of the version before needs only the features added since (features.ADDED_IN)
        features = ({**old, **detector_type_features(circuit)} if old is not None else features_from_circuit(circuit))
        return {"key": key, **stat, "version": FEATURE_VERSION, "seconds": round(time.time() - start, 3),
                "features": features}
    except Exception as e:
        return {"key": key, **stat, "version": FEATURE_VERSION, "error": f"{type(e).__name__}: {str(e)[:300]}"}


def extract(run_dirs, store_path: str, processes: int = 1, keys=None, log=print) -> dict:
    """Add the features of new or changed circuit files of the run directories to the store; keys, if
    given, limits extraction to those circuits. Returns counts"""
    store = load_store(store_path)
    todo = []
    for key, path in _circuit_files(run_dirs, keys):
        st = os.stat(path)
        stat = {"size": st.st_size, "mtime": int(st.st_mtime)}
        old = store.get(key)
        same = old and old.get("size") == stat["size"] and old.get("mtime") == stat["mtime"]
        if same and old.get("version") == FEATURE_VERSION:
            continue
        upgrade = same and old.get("version") == FEATURE_VERSION - 1 and "features" in old
        todo.append((key, path, stat, old["features"] if upgrade else None))
    # Big circuits first, so no straggler is left at the end
    todo.sort(key=lambda t: -t[2]["size"])
    done = failed = 0
    with open(store_path, "a") as out, Pool(processes) as pool:
        for record in pool.imap_unordered(_extract, todo, chunksize=1):
            out.write(json.dumps(record) + "\n")
            out.flush()
            done += 1
            failed += "error" in record
            if done % 200 == 0:
                log(f"  {done}/{len(todo)} circuits")
    return {"known": len(store), "extracted": done, "failed": failed}
