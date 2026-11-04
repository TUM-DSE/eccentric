import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from qec_matchmaker.dataset import LABELS
from qec_matchmaker.models import ModelBundle, circuit_features, circuit_labels, regret_rows, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Synthetic truth: log10 best LER = 2 log10(det_fire_mean) + 1.5; each decoder a fixed LER factor over the
# best; log10 ms per shot = a + b log10(mechanisms)
REGRET = {"tesseract": 1.0, "bposd": 1.3, "mwpm": 2.0}
TIME = {"tesseract": (-4.0, 1.5), "bposd": (-2.0, 1.0), "mwpm": (-3.0, 0.8)}


def true_ler(fire):
    return np.minimum(0.95, 10 ** (2 * np.log10(fire) + 1.5))


def true_ms(decoder, mechanisms):
    a, b = TIME[decoder]
    return 10 ** (a + b * np.log10(mechanisms))


def wilson(errors, shots, z=1.96):
    p = errors / shots
    denom = 1 + z * z / shots
    centre = (p + z * z / (2 * shots)) / denom
    half = z * np.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denom
    return centre - half, centre + half


def synthetic_features(circuits, seed):
    rng = np.random.default_rng(seed)
    fire, mechanisms = 10 ** rng.uniform(-3, -0.5, circuits), 10 ** rng.uniform(2, 5, circuits)
    return pd.DataFrame({"circuit_key": [f"s{seed}/c{i}" for i in range(circuits)], "detectors": mechanisms / 10,
                         "mechanisms": mechanisms, "det_fire_mean": fire, "error_mass": fire * mechanisms / 10,
                         "decomposable": (np.arange(circuits) % 2).astype(float), "graphlike": math.nan})


def synthetic_table(circuits=240, codes=12, seed=0, timeout_ms=None):
    """A training table with known relationships; tesseract decodes slower than timeout_ms time out"""
    rng = np.random.default_rng(seed + 1000)
    rows = []
    for i, f in synthetic_features(circuits, seed).iterrows():
        best = true_ler(f.det_fire_mean) * 10 ** rng.normal(0, 0.05)
        for decoder in REGRET:
            if decoder == "mwpm" and not f.decomposable:
                continue
            ler = min(1.0, best * REGRET[decoder])
            ms = true_ms(decoder, f.mechanisms) * 10 ** rng.normal(0, 0.05)
            row = {**f.to_dict(), "code": f"code{i % codes}", "decoder": decoder}
            if timeout_ms and decoder == "tesseract" and ms > timeout_ms:
                rows.append({**row, "status": "timeout", "timeout_s": timeout_ms * 100 / 1000})
                continue
            shots = int(400 / ler) + 100
            errors = rng.binomial(shots, ler)
            low, high = wilson(errors, shots)
            rows.append({**row, "status": "measured" if errors >= 100 else "low statistics", "shots": shots,
                         "errors": errors, "ler": errors / shots, "ler_low": low, "ler_high": high, "ms_per_shot": ms})
    table = pd.DataFrame(rows)
    return table.reindex(columns=list(table.columns) + [c for c in LABELS if c not in table.columns])


@pytest.fixture(scope="module")
def bundle():
    return train(synthetic_table())


@pytest.fixture(scope="module")
def fresh(bundle):
    features = synthetic_features(120, seed=7)
    return features, bundle.predict(features)


def test_labels_follow_the_definitions():
    rows = [("a", "bposd", "measured", 200, 0.02, 0.018, 0.022), ("a", "tesseract", "upper bound", 0, 0.0, 0.0, 0.001),
            ("b", "bposd", "measured", 500, 0.6, 0.58, 0.62), ("b", "tesseract", "measured", 400, 0.7, 0.68, 0.72),
            ("c", "bposd", "timeout", math.nan, math.nan, math.nan, math.nan)]
    df = pd.DataFrame(rows, columns=["circuit_key", "decoder", "status", "errors", "ler", "ler_low", "ler_high"])
    labels = circuit_labels(df)
    assert list(labels.index) == ["a", "b"]
    assert labels.loc["a"].best_ler == 0.001 and labels.loc["a"].best_status == "upper bound"
    assert labels.loc["b"].above and not labels.loc["a"].above
    reg = regret_rows(df)
    assert set(reg.circuit_key) == {"b"}
    assert reg.set_index("decoder").regret.to_dict() == pytest.approx({"bposd": 0.0, "tesseract": math.log10(0.7 / 0.6)})


def test_features_used_are_those_present_and_filled(bundle):
    assert bundle.features == ["detectors", "mechanisms", "error_mass", "det_fire_mean", "decomposable"]
    assert bundle.meta["decoders"]["time"] == ["bposd", "mwpm", "tesseract"]
    assert bundle.decoders() == ["mwpm", "bposd", "tesseract"]


def test_prediction_columns_and_consistency(fresh):
    features, pred = fresh
    assert list(pred.columns) == ["circuit_key", "decoder", "p_above", "ler_q10", "ler_q50", "ler_q90", "regret_q50",
                                  "ler_decoder_q50", "ms_q10", "ms_q50", "ms_q90", "applicable", "outside_training"]
    assert len(pred) == 3 * len(features)
    assert (pred.ler_q10 <= pred.ler_q50).all() and (pred.ler_q50 <= pred.ler_q90).all()
    assert (pred.ms_q10 <= pred.ms_q50).all() and (pred.ms_q50 <= pred.ms_q90).all()
    assert (pred.regret_q50 >= 1).all()
    assert np.allclose(pred.ler_decoder_q50, pred.ler_q50 * pred.regret_q50)
    mwpm = pred[pred.decoder == "mwpm"].set_index("circuit_key").applicable
    assert (mwpm == features.set_index("circuit_key").decomposable.astype(bool)).all()
    assert pred[pred.decoder != "mwpm"].applicable.all()


def test_ler_and_regime_are_recovered(fresh):
    features, pred = fresh
    per_circuit = pred.groupby("circuit_key", sort=False).first().loc[features.circuit_key]
    truth = np.log10(true_ler(features.det_fire_mean.to_numpy()))
    err = np.abs(np.log10(per_circuit.ler_q50.to_numpy()) - truth)
    assert np.median(err) < 0.2
    assert pd.Series(np.log10(per_circuit.ler_q50.to_numpy())).corr(pd.Series(truth), method="spearman") > 0.95
    above = true_ler(features.det_fire_mean.to_numpy()) > 0.5
    assert np.mean((per_circuit.p_above.to_numpy() > 0.5) == above) > 0.9


def test_regret_and_time_are_recovered(fresh):
    features, pred = fresh
    regret = pred.groupby("decoder").regret_q50.median()
    for decoder, factor in REGRET.items():
        assert abs(math.log10(regret[decoder] / factor)) < 0.08
    mechanisms = pred.circuit_key.map(features.set_index("circuit_key").mechanisms)
    for decoder in TIME:
        s = pred.decoder == decoder
        err = np.abs(np.log10(pred.ms_q50[s].to_numpy()) - np.log10(true_ms(decoder, mechanisms[s].to_numpy())))
        assert np.median(err) < 0.15
    # Tesseract is faster than BP-OSD on small circuits (true crossover at 1e4 mechanisms), slower on large ones
    wide = pred.pivot(index="circuit_key", columns="decoder", values="ms_q50")
    size = features.set_index("circuit_key").mechanisms.loc[wide.index]
    assert (wide.tesseract < wide.bposd)[size < 1e3].mean() > 0.9
    assert (wide.tesseract > wide.bposd)[size > 5e4].mean() > 0.9


def test_predict_accepts_a_dict_and_missing_columns_and_unknown_decoders(bundle):
    one = bundle.predict({"mechanisms": 1e3, "det_fire_mean": 0.01, "decomposable": 0}, decoders=["bposd", "relay_bp"])
    assert list(one.decoder) == ["bposd", "relay_bp"]
    assert one.ms_q50.iloc[0] > 0 and math.isnan(one.ms_q50.iloc[1]) and math.isnan(one.ler_decoder_q50.iloc[1])
    assert one.circuit_key.iloc[0] == 0


def test_timeouts_are_censored_not_dropped():
    bound = 300.0
    table = synthetic_table(timeout_ms=bound)
    assert (table.status == "timeout").sum() > 20
    censored = train(table)
    dropped = train(table[table.status != "timeout"])
    assert censored.meta["training"]["time_censored"] == (table.status == "timeout").sum()
    big = synthetic_features(40, seed=11).assign(mechanisms=1e5, detectors=1e4)
    ms = {name: b.predict(big, ["tesseract"]).ms_q50.median() for name, b in (("censored", censored), ("dropped", dropped))}
    assert ms["censored"] > ms["dropped"]
    assert ms["censored"] >= 0.9 * bound


def test_save_load_round_trip(bundle, fresh, tmp_path):
    features, pred = fresh
    bundle.meta["metrics"] = {"regime": {"accuracy": np.float64(0.9)}}
    bundle.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["features"] == bundle.features and meta["quantiles"] == [0.1, 0.5, 0.9]
    assert meta["metrics"]["regime"]["accuracy"] == 0.9 and "sklearn_version" in meta
    loaded = ModelBundle.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.predict(features), pred)


def test_load_warns_on_another_sklearn_version(bundle, tmp_path):
    bundle.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["sklearn_version"] = "0.0"
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.warns(UserWarning, match="scikit-learn 0.0"):
        ModelBundle.load(tmp_path)


def test_trains_on_the_committed_dataset():
    if not os.path.exists(os.path.join(ROOT, "data", "MANIFEST.json")):
        pytest.skip("no data")
    from qec_matchmaker.dataset import load_dataset
    df, manifest = load_dataset(os.path.join(ROOT, "data"))
    bundle = train(df, manifest=manifest)
    assert bundle.meta["manifest"]["rows"] == len(df)
    for column in ("code", "family", "backend", "slice", "error_type"):
        assert column not in bundle.features
    pred = bundle.predict(circuit_features(df).reset_index())
    labels = circuit_labels(df)
    p_above = pred.groupby("circuit_key").p_above.first().loc[labels.index]
    assert np.mean((p_above > 0.5) == labels.above) > 0.8
    ler = pred.groupby("circuit_key").ler_q50.first().loc[labels.index[labels.best_status == "measured"]]
    truth = np.log10(labels.best_ler[labels.best_status == "measured"].clip(1e-8, 1))
    assert pd.Series(np.log10(ler.to_numpy())).corr(pd.Series(truth.to_numpy()), method="spearman") > 0.8


def test_conformal_calibration_widens_the_intervals(bundle):
    calibrated = train(synthetic_table(), calibrate=True, calibration_folds=3)
    conformal = calibrated.meta["conformal"]
    assert conformal["ler"] >= 0 and conformal["time"] and all(v >= 0 for v in conformal["time"].values())
    features = synthetic_features(40, seed=3)
    plain, wide = bundle.predict(features), calibrated.predict(features)
    assert (wide.ler_q10 <= plain.ler_q10 * (1 + 1e-9)).all() and (wide.ler_q90 >= plain.ler_q90 * (1 - 1e-9)).all()
    assert (wide.ler_q50 == plain.ler_q50).all()


def test_predictions_name_features_outside_the_training_range(bundle):
    inside = synthetic_features(1, seed=5).iloc[0].to_dict()
    assert bundle.outside_training(inside) == []
    far = {**inside, "mechanisms": 1e12}
    assert bundle.outside_training(far)[0].startswith("mechanisms 1e+12")
    assert bundle.predict(far).outside_training.iloc[0].startswith("mechanisms")
