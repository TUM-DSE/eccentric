import json
import os
import subprocess
import sys

import pandas as pd

from test_rerun import decode_job, write

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_add_decoder_copies_the_selected_circuits_with_another_decoder(tmp_path):
    base, out = tmp_path / "base", tmp_path / "out"
    rows = []
    for i, status in enumerate(["decoder not applicable", "measured", "decoder not applicable"]):
        job = decode_job(i, "mwpm")
        write(base / "run1" / "jobs" / f"{job['id']}.json", {**job, "time_budget": 3600})
        write(base / "run1" / "results" / f"circuit.s.c{i}.json", {"status": "done"})
        rows.append({"run": "run1", "job_id": job["id"], "decoder": "mwpm", "status": status, "code": "color"})
    pd.DataFrame(rows).to_csv(tmp_path / "results.csv", index=False)
    run = lambda *extra: subprocess.run([sys.executable, "-m", "sweep.add_decoder", str(out), str(tmp_path / "results.csv"), str(base),
                                         "mwpf", *extra], cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert run("--query", "status == 'decoder not applicable'", "--set", "time_budget=60").startswith("2 mwpf")
    assert sorted(os.listdir(out / "jobs")) == ["decode.s.c0.decoder=mwpf.json", "decode.s.c2.decoder=mwpf.json"]
    job = json.load(open(out / "jobs" / "decode.s.c0.decoder=mwpf.json"))
    assert job["decoder"] == "mwpf" and job["time_budget"] == 60 and job["added_from"] == "run1/decode.s.c0.decoder=mwpm"
    assert job["circuit_file"] == str(base / "run1" / "circuits" / "circuit.s.c0.stim")
    assert sorted(os.listdir(out / "results")) == ["circuit.s.c0.json", "circuit.s.c2.json"]
    # Again, with the rest: only the new circuit is added
    assert run("--query", "code == 'color'").startswith("1 mwpf")
