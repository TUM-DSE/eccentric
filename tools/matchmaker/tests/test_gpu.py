"""The GPU option: pilots load, the packaged model predicts GPU versions with GPU times, accuracy unchanged"""
import json
import os

import numpy as np
import pytest

from qec_matchmaker.api import MODEL_DIR
from qec_matchmaker.gpu import GPU_ONLY, GPU_VERSIONS, load_gpu_only_pilots, load_gpu_only_rows, load_pilots
from qec_matchmaker.models import ModelBundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_load_pilots_keeps_measured_ones(tmp_path):
    lines = [
        {"kind": "circuit", "index": 0, "key": "k"},
        {"kind": "start", "index": 0, "key": "k", "decoder": "gari"},
        {"kind": "pilot", "index": 0, "key": "k", "decoder": "gari", "status": "measured", "gpu_shots_per_second": 500.0,
         "build_seconds": 0.5, "shots": 100000, "cpu_ms_per_shot": 20.0},
        {"kind": "pilot", "index": 0, "key": "k", "decoder": "bplsd", "status": "skipped: predicted too slow",
         "cpu_ms_per_shot": 9000.0},
        {"kind": "start", "index": 0, "key": "k", "decoder": "relay_bp"},  # cut off
    ]
    (tmp_path / "0.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    p = load_pilots(str(tmp_path))
    assert list(p.decoder) == ["gari"] and p.gpu_ms_per_shot.iloc[0] == pytest.approx(2.0)


@pytest.fixture(scope="module")
def bundle():
    b = ModelBundle.load(MODEL_DIR)
    if "gpu_time" not in b.models:
        pytest.skip("packaged model without a GPU time model")
    return b


def _one_circuit(bundle):
    pilots = load_pilots(os.path.join(ROOT, "experiments", "gpu", "results"))
    from qec_matchmaker.pipeline import features_of_store, load_store
    feats = features_of_store(load_store(os.path.join(ROOT, "data", "features.jsonl.gz")))
    key = pilots.circuit_key.iloc[0]
    return {**feats[key], "circuit_key": key}


def test_gpu_predictions_rename_and_retime(bundle):
    circuit = _one_circuit(bundle)
    cpu, gpu = bundle.predict(circuit), bundle.predict(circuit, gpu=True)
    assert set(cpu.device) == {"cpu core"}
    renamed = gpu[(gpu.device == "gpu") & ~gpu.decoder.isin(GPU_ONLY)]
    assert set(renamed.decoder) == {GPU_VERSIONS[d] for d in cpu.decoder if d in GPU_VERSIONS}
    by_cpu = cpu.set_index("decoder")
    for _, row in renamed.iterrows():
        base = by_cpu.loc[row.decoder.removesuffix("_gpu")]
        assert row.ler_decoder_q50 == base.ler_decoder_q50  # same corrections, same accuracy
        assert row.ms_q10 <= row.ms_q50 <= row.ms_q90
    kept = gpu[gpu.device == "cpu core"].set_index("decoder")
    assert np.allclose(kept.ms_q50, by_cpu.loc[kept.index, "ms_q50"])


def test_gpu_needs_a_gpu_model(bundle):
    plain = ModelBundle({k: v for k, v in bundle.models.items() if k != "gpu_time"}, bundle.meta)
    with pytest.raises(ValueError, match="no GPU time model"):
        plain.predict(_one_circuit(bundle), gpu=True)


def test_gpu_model_meta(bundle):
    meta = bundle.meta["gpu"]
    assert meta["device"] and set(meta["build_seconds"]) == set(GPU_VERSIONS) | set(meta.get("gpu_only", []))
    best = meta["evaluation"]["variants"][meta["evaluation"]["chosen"]]
    assert best["within_3x"] > 0.8


def _slice_result(key, errors, shots, target=100, timing=True):
    out = {"index": 0, "key": key, "settings": {"target_errors": target}, "ler": {"shots": shots, "errors": errors}}
    if timing:
        out["timing"] = {"gpu_shots_per_second": 4000.0, "build_seconds": 0.2, "shots": 100000}
    return out


def test_gpu_only_rows_follow_eccentric_status_rules(tmp_path):
    import pandas as pd
    d = tmp_path / "results_tcbp_jobs_sc_gpu"
    d.mkdir()
    for i, (key, e, n) in enumerate([("a", 120, 5000), ("b", 40, 100000), ("c", 0, 100000), ("unknown", 5, 10)]):
        (d / f"{i}.json").write_text(json.dumps(_slice_result(key, e, n)))
    table = pd.DataFrame({"circuit_key": ["a", "a", "b", "c"], "decoder": ["mwpm", "bposd", "mwpm", "mwpm"],
                          "code": ["surface", "surface", "color", "bb72"], "status": "measured", "ler": 0.01,
                          "ms_per_shot": 1.0, "detectors": [10, 10, 20, 30]})
    rows = load_gpu_only_rows(str(tmp_path), table).set_index("circuit_key")
    assert list(rows.index) == ["a", "b", "c"] and set(rows.decoder) == {"tcbp_jobs_sc_gpu"}
    assert list(rows.status) == ["measured", "low statistics", "upper bound"]
    assert rows.loc["a", "ler"] == pytest.approx(120 / 5000) and rows.loc["a", "ler_low"] < rows.loc["a", "ler"] < rows.loc["a", "ler_high"]
    assert np.isnan(rows.loc["c", "ler"]) and rows.loc["c", "ler_high"] > 0
    assert rows.ms_per_shot.isna().all()  # no CPU time
    assert rows.loc["b", "code"] == "color" and rows.loc["b", "detectors"] == 20  # the circuit's columns
    pilots = load_gpu_only_pilots(str(tmp_path))
    assert len(pilots) == 4 and pilots.gpu_ms_per_shot.iloc[0] == pytest.approx(0.25) and pilots.cpu_ms_per_shot.isna().all()


def test_gpu_only_decoders_only_with_a_gpu(bundle):
    circuit = _one_circuit(bundle)
    cpu, gpu = bundle.predict(circuit), bundle.predict(circuit, gpu=True)
    assert not cpu.decoder.isin(GPU_ONLY).any()
    only = gpu[gpu.decoder.isin(GPU_ONLY)]
    if not bundle.meta["gpu"].get("gpu_only"):
        pytest.skip("packaged model without GPU-only decoders")
    assert len(only) == len(bundle.meta["gpu"]["gpu_only"]) and set(only.device) == {"gpu"}
    assert only.ms_q50.notna().all() and only.ler_decoder_q50.notna().all()
