import json
import os
import subprocess
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def decode_job(i, decoder):
    return {"id": f"decode.s.c{i}.decoder={decoder}", "type": "decode", "needs": f"circuit.s.c{i}", "decoder": decoder,
            "slice": "s", "code": "c"}


def test_rerun_copies_the_selected_decodes_with_their_circuits(tmp_path):
    base, rerun = tmp_path / "base", tmp_path / "rerun"
    rows = []
    for i, detectors in enumerate([36, 40, 252]):
        for decoder in ("relay_bp", "bposd"):
            job = decode_job(i, decoder)
            write(base / "run1" / "jobs" / f"{job['id']}.json", job)
            rows.append({"run": "run1", "job_id": job["id"], "decoder": decoder, "status": "measured",
                         "result_file": f"results/{job['id']}.json", "detectors": detectors})
        write(base / "run1" / "results" / f"circuit.s.c{i}.json", {"status": "done", "detectors": detectors})
    rows.append({"run": "run1", "job_id": "decode.s.c9.decoder=relay_bp", "decoder": "relay_bp", "status": "pending",
                 "result_file": None, "detectors": 37})
    pd.DataFrame(rows).to_csv(tmp_path / "results.csv", index=False)
    # A run still going: one decode before the cutoff, one after, one on a multiple of 8 detectors
    cutoff = time.time() - 100
    for i, detectors, started in ((5, 60, cutoff - 50), (6, 60, cutoff + 50), (7, 64, cutoff - 50)):
        job = decode_job(i, "relay_bp")
        write(base / "run2" / "jobs" / f"{job['id']}.json", job)
        write(base / "run2" / "results" / f"circuit.s.c{i}.json", {"status": "done", "detectors": detectors})
        path = base / "run2" / "results" / f"{job['id']}.json"
        write(path, {"job": job, "status": "done", "seconds_total": 10})
        os.utime(path, (started + 10, started + 10))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(cutoff))
    out = subprocess.run([sys.executable, "-m", "sweep.rerun", str(rerun), str(tmp_path / "results.csv"), str(base), "relay_bp",
                          "--detectors-not-multiple-of", "8", "--also", str(base / "run2"), "--started-before", stamp],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert "2 decodes" in out and "1 more" in out
    jobs = sorted(os.listdir(rerun / "jobs"))
    assert jobs == ["decode.s.c0.decoder=relay_bp.json", "decode.s.c2.decoder=relay_bp.json", "decode.s.c5.decoder=relay_bp.json"]
    job = json.load(open(rerun / "jobs" / "decode.s.c0.decoder=relay_bp.json"))
    assert job["circuit_file"] == str(base / "run1" / "circuits" / "circuit.s.c0.stim") and job["reruns"] == "run1/decode.s.c0.decoder=relay_bp"
    assert sorted(os.listdir(rerun / "results")) == ["circuit.s.c0.json", "circuit.s.c2.json", "circuit.s.c5.json"]
