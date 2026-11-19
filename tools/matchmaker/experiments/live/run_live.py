"""Live run of the pilot runtime on a slice of ECCentric's sweep (prepare.py), on the cluster:
python run_live.py <slice.json> <out_dir> <eccentric runs base dir> [--workers N] [--shard i/n]

For every circuit, qec_matchmaker.runtime.execute with real decoders: a pilot starts ECCentric's adaptive
decode (decode_worker.py) and watches its first batch of 100 shots under the time cap; committing lets that
process continue to its error target, switching or stopping kills it. Times are the decode's own (from the
start of decoding, including the decoder's construction; not the interpreter's start-up). One JSON per
circuit in <out_dir>/results; circuits with a result are not run again.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from qec_matchmaker.runtime import Batch, execute

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = dict(target_errors=100, max_shots=100_000, time_budget=3600, first_batch=100, max_batch=5000)
TIMEOUT = 3 * 3600
STARTUP = 900


def locate(base, key):
    """The circuit file of a circuit key (<run>/<circuit job id>) and the shot limit of its decodes"""
    run, name = key.split("/", 1)
    jobs = sorted(glob.glob(os.path.join(base, run, "jobs", "decode." + glob.escape(name[len("circuit."):]) + ".decoder=*.json")))
    job = json.load(open(jobs[0])) if jobs else {}
    path = job.get("circuit_file") or os.path.join(base, run, "circuits", name + ".stim")
    return path, job.get("max_shots", SETTINGS["max_shots"])


class Live:
    def __init__(self, circuit, base, work):
        self.circuit, self.work, self.procs, self.started = circuit, work, {}, {}
        self.path, self.max_shots = locate(base, circuit["key"])
        os.makedirs(work, exist_ok=True)

    def progress(self, decoder):
        try:
            with open(os.path.join(self.work, decoder + ".progress.json")) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def kill(self, keep=None):
        for d, p in self.procs.items():
            if d != keep and p.poll() is None:
                p.kill()

    def start(self, decoder):
        spec = os.path.join(self.work, decoder + ".spec.json")
        with open(spec, "w") as f:
            json.dump({"circuit_file": self.path, "code": self.circuit["code"], "decoder": decoder,
                       "backend": self.circuit["backend"], "error_type": self.circuit["error_type"],
                       "settings": dict(SETTINGS, max_shots=self.max_shots)}, f)
        log = open(os.path.join(self.work, decoder + ".log"), "w")
        self.procs[decoder] = subprocess.Popen(
            [sys.executable, "-W", "ignore", os.path.join(HERE, "decode_worker.py"), spec,
             os.path.join(self.work, decoder + ".progress.json")], stdout=log, stderr=log,
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))

    def decode(self, decoder, shots, seconds):
        if shots is not None:
            self.kill()
            self.start(decoder)
            launched = time.time()
            while True:
                state, alive = self.progress(decoder), self.procs[decoder].poll() is None
                if state and state["state"] == "failed":
                    return None
                if state and state["state"] in ("running", "done") and state.get("shots", 0) >= shots:
                    return Batch(int(state["shots"]), int(state["errors"]), float(state["seconds"]))
                if state and state["state"] == "done":  # fewer shots than a pilot in all: the decode is over
                    return Batch(int(state["shots"]), int(state["errors"]), float(state["seconds"]))
                began = state["time"] if state and state["state"] == "started" else None
                if began:
                    self.started[decoder] = began
                if (began and time.time() - began > seconds) or (not began and time.time() - launched > STARTUP):
                    self.kill()
                    return Batch(0, 0, seconds, False) if began else None
                if not alive and not state:
                    return None
                if not alive and state["state"] == "started":
                    return None  # died while decoding its first batch
                time.sleep(0.2)
        began = self.started.get(decoder, time.time())
        while True:
            state, alive = self.progress(decoder), self.procs[decoder].poll() is None
            if state and state["state"] == "done":
                return Batch(int(state["shots"]), int(state["errors"]), float(state["seconds"]))
            if (state and state["state"] == "failed") or not alive or time.time() - began > TIMEOUT:
                self.kill()
                state = state or {}
                return Batch(int(state.get("shots", 0)), int(state.get("errors", 0)),
                             min(time.time() - began, TIMEOUT), False)
            time.sleep(1)


def run_circuit(circuit, base, out_dir, objective):
    safe = circuit["key"].replace("/", "__")
    result = os.path.join(out_dir, "results", safe + ".json")
    if os.path.exists(result):
        return
    live = Live(circuit, base, os.path.join(out_dir, "work", safe))
    start = time.time()
    try:
        o = execute(pd.DataFrame(circuit["predictions"]), live.decode, objective)
        r = o.result
        record = {"key": circuit["key"], "status": o.status, "decoder": o.decoder, "seconds": o.seconds,
                  "shots": r.shots if r else 0, "errors": r.errors if r else 0, "complete": bool(r and r.complete),
                  "trace": [[s, d, t] for s, d, t in o.trace], "wall": time.time() - start,
                  "host": os.uname().nodename}
    except Exception as e:  # keep the run going; the record says what happened
        record = {"key": circuit["key"], "status": "error", "error": repr(e)[:500], "wall": time.time() - start}
    finally:
        live.kill()
    with open(result + ".tmp", "w") as f:
        json.dump(record, f)
    os.replace(result + ".tmp", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("slice")
    p.add_argument("out_dir")
    p.add_argument("base")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    data = json.load(open(a.slice))
    i, n = map(int, a.shard.split("/"))
    # Longest (by the simulation) first, so that the tail of the run is short
    circuits = sorted(data["circuits"], key=lambda c: -c["simulated"]["seconds"])[i::n]
    os.makedirs(os.path.join(a.out_dir, "results"), exist_ok=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(lambda c: run_circuit(c, a.base, a.out_dir, data["objective"]), circuits))
    print(f"shard {a.shard}: {len(circuits)} circuits done")


if __name__ == "__main__":
    main()
