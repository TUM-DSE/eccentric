import json
import math
import os

import pandas as pd
import pytest
import stim

from qec_matchmaker.cli import main
from qec_matchmaker.dataset import circuit_key, from_eccentric, load_dataset, write_dataset
from qec_matchmaker.decoders import applicable
from qec_matchmaker.features import FEATURE_NAMES
from qec_matchmaker.pipeline import extract, features_of_store, load_store


def surface(d, p=0.003):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p)


def make_run(tmp_path, name="sweep_x"):
    run = tmp_path / name
    (run / "circuits").mkdir(parents=True)
    for d in (3, 5):
        surface(d).to_file(run / "circuits" / f"circuit.s.code=surface.distance={d}.stim")
    return run


def results_frame(run):
    rows = []
    for d, ler in ((3, 0.02), (5, 0.004)):
        for decoder, extra in (("mwpm", 1.0), ("bposd", 0.9)):
            shots = 10_000
            rows.append(dict(run=run, job_id=f"decode.s.code=surface.distance={d}.decoder={decoder}", code="surface",
                             distance=d, cycles=d, backend="custom_full", error_type="sd6", error_probability=0.003,
                             slice="s", decoder=decoder, status="measured", shots=shots, errors=int(ler * extra * shots),
                             ler_low=ler * 0.9, ler_high=ler * 1.1, decode_seconds=shots * 0.001, note=None,
                             result_file=f"results/decode.{d}.{decoder}.json"))
    rows.append({**rows[0], "job_id": "decode.s.code=gross.decoder=bposd", "status": "does not fit", "result_file": None})
    rows.append({**rows[1], "status": "timeout", "shots": None, "errors": None, "note": "timeout after 9 h",
                 "job_id": "decode.s.code=surface.distance=3.decoder=tesseract", "decoder": "tesseract"})
    return pd.DataFrame(rows)


def test_circuit_key():
    assert circuit_key("sweep_idle", "decode.a.code=bb72.decoder=bposd.long") == "sweep_idle/circuit.a.code=bb72"


def test_extract_is_incremental(tmp_path):
    run, store = make_run(tmp_path), str(tmp_path / "store.jsonl")
    assert extract([str(run)], store, processes=2)["extracted"] == 2
    assert extract([str(run)], store, processes=2)["extracted"] == 0
    path = run / "circuits" / "circuit.s.code=surface.distance=3.stim"
    surface(3, p=0.005).to_file(path)
    os.utime(path, (1, 1))
    assert extract([str(run)], store)["extracted"] == 1
    records = load_store(store)
    assert len(records) == 2 and set(features_of_store(records)["sweep_x/circuit.s.code=surface.distance=5"]) == set(FEATURE_NAMES)


def test_ingest_joins_results_and_features(tmp_path):
    run, store = make_run(tmp_path), str(tmp_path / "store.jsonl")
    extract([str(run)], store)
    table, left_out = from_eccentric(results_frame("sweep_x"), features_of_store(load_store(store)))
    assert len(table) == 5 and left_out["skipped (no circuit)"] == 1
    timeout = table[table.status == "timeout"].iloc[0]
    assert timeout.timeout_s == 9 * 3600 and math.isnan(timeout.ler)
    measured = table[table.decoder == "mwpm"].sort_values("distance")
    assert list(measured.ler) == pytest.approx([0.02, 0.004]) and list(measured.ms_per_shot) == pytest.approx([1, 1])
    assert measured.det_fire_mean.iloc[0] > 0 and measured.family.iloc[0] == "Surface"
    manifest = write_dataset(table, str(tmp_path / "data"), {"results": {"path": "x"}}, left_out)
    again, m = load_dataset(str(tmp_path / "data"))
    assert len(again) == 5 and m["circuits"] == manifest["circuits"] == 2


def test_cli_features_extract_ingest(tmp_path, capsys):
    run, store = make_run(tmp_path), str(tmp_path / "store.jsonl")
    main(["features", str(run / "circuits" / "circuit.s.code=surface.distance=3.stim")])
    assert json.loads(capsys.readouterr().out)["detectors"] == 24
    results = tmp_path / "results.csv"
    results_frame("sweep_x").to_csv(results, index=False)
    main(["extract", str(run), "--store", store, "--keys-from", str(results)])
    main(["ingest", "--results", str(results), "--store", store, "--out", str(tmp_path / "data")])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["rows"] == 5


def test_applicable():
    assert applicable({"decomposable": 1, "color_annotated": 0}) == {
        "mwpm": True, "mwpf": True, "bposd": True, "bposd_tuned": True, "bplsd": True, "belief_find": True,
        "relay_bp": True, "tesseract": True, "chromobius": False, "gari": True, "gari_pp": True}
    assert applicable({"decomposable": math.nan, "graphlike": 0, "color_annotated": 1})["mwpm"] is None
    assert applicable({"decomposable": math.nan, "graphlike": 1})["mwpm"] is True
    assert applicable({})["chromobius"] is None


def test_extract_follows_circuit_files_of_reruns(tmp_path):
    # A rerun directory (sweep.rerun, sweep.add_decoder) decodes the circuit files of another run; its rows are
    # keyed under the rerun directory, so its circuits must be too
    run = make_run(tmp_path)
    rerun = tmp_path / "sweep_rerun"
    (rerun / "jobs").mkdir(parents=True)
    path = run / "circuits" / "circuit.s.code=surface.distance=3.stim"
    job = {"id": "decode.s.code=surface.distance=3.decoder=relay_bp", "type": "decode", "needs": path.stem,
           "circuit_file": str(path)}
    (rerun / "jobs" / f"{job['id']}.json").write_text(json.dumps(job))
    store = tmp_path / "store.jsonl"
    counts = extract([str(run), str(rerun)], str(store))
    assert counts["extracted"] == 3
    records = load_store(str(store))
    assert records[f"sweep_rerun/{path.stem}"]["features"] == records[f"sweep_x/{path.stem}"]["features"]


def test_ingest_gives_a_configuration_one_circuit_key(tmp_path):
    # The Relay-BP decode of a configuration came from a rerun directory: it must join the configuration's
    # other decodes under one circuit key
    run = make_run(tmp_path)
    results = results_frame("sweep_x")
    rerun = results.iloc[[0]].assign(run="sweep_rerun", decoder="relay_bp", job_id=lambda r: r.job_id.str.replace("mwpm", "relay_bp"))
    results = pd.concat([results, rerun], ignore_index=True)
    store = tmp_path / "store.jsonl"
    extract([str(run)], str(store))
    records = features_of_store(load_store(str(store)))
    records["sweep_rerun/circuit.s.code=surface.distance=3"] = records["sweep_x/circuit.s.code=surface.distance=3"]
    table, left_out = from_eccentric(results, records)
    d3 = table[table.distance == 3]
    assert d3.circuit_key.nunique() == 1 and set(d3.decoder) >= {"mwpm", "bposd", "relay_bp"}
