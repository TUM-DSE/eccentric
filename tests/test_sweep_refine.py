import json
import os

import stim

from validation.run_job import decode_adaptively


def decode_ones(circuit):
    """A batch decoder that counts how often the first detector fires"""
    def run(n):
        samples = circuit.compile_detector_sampler().sample(n)
        return samples[:, 0].mean()
    return run


def test_seed_offset_gives_other_shots():
    circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2, before_round_data_depolarization=0.3)
    job = dict(first_batch=200, max_batch=200, max_shots=200, target_errors=10**9, time_budget=60)
    seen = []
    a = decode_adaptively(job, decode_ones(circuit), seen.append)
    b = decode_adaptively(job, decode_ones(circuit))
    c = decode_adaptively({**job, "seed_offset": 10_000_000}, decode_ones(circuit))
    assert a["errors"] == b["errors"] and a["errors"] != c["errors"]
    assert seen[-1]["shots"] == 200


def test_collect_adds_refinement_chunks(tmp_path):
    from sweep.collect import refinements, rows

    def write(run, name, obj, sub):
        os.makedirs(tmp_path / run / sub, exist_ok=True)
        with open(tmp_path / run / sub / f"{name}.json", "w") as f:
            json.dump(obj, f)

    base = dict(id="decode.x", type="decode", needs="circuit.x", decoder="bposd", target_errors=100, slice="s", code="c")
    write("run", "decode.x", base, "jobs")
    write("run", "circuit.x", {"job": {"id": "circuit.x"}, "status": "done", "rounds": 2}, "results")
    write("run", "decode.x", {"job": base, "status": "failed", "error": "timeout after 3 s"}, "results")
    for i, errors in enumerate((40, 70)):
        chunk = {**base, "id": f"decode.x.refine{i}", "refines": "run/decode.x"}
        write("refine", chunk["id"], chunk, "jobs")
        write("refine", chunk["id"], {"job": chunk, "status": "done", "shots": 1000, "errors": errors}, "results")
    dirs = [str(tmp_path / "run"), str(tmp_path / "refine")]
    added = refinements(dirs)
    (row,) = list(rows(dirs[0], added))
    assert (row["shots"], row["errors"], row["status"], row["refined_chunks"]) == (2000, 110, "measured", 2)
    assert list(rows(dirs[1], added)) == []


def test_worker_keeps_partial_counts(tmp_path):
    from validation.worker import keep_partial

    os.makedirs(tmp_path / "results")
    job = {"id": "decode.y"}
    with open(tmp_path / "results" / "decode.y.partial", "w") as f:
        json.dump({"shots": 300, "errors": 3}, f)
    assert keep_partial(str(tmp_path), job, "timeout after 1 s")
    with open(tmp_path / "results" / "decode.y.json") as f:
        result = json.load(f)
    assert result["status"] == "done" and result["shots"] == 300 and result["stopped"] == "timeout after 1 s"
    assert not keep_partial(str(tmp_path), {"id": "decode.z"}, "timeout")
