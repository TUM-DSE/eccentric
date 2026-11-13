import json

import numpy as np
import pytest

from qec_matchmaker.evaluate import cross_val_predictions, evaluate, held_out_beyond, report_markdown
from test_models import synthetic_table


@pytest.fixture(scope="module")
def table():
    return synthetic_table(circuits=120, codes=6, seed=3, timeout_ms=300.0)


def test_cross_val_predictions_hold_out_whole_codes(table):
    pred = cross_val_predictions(table, folds=3)
    assert pred.fold.nunique() == 3
    assert set(pred.circuit_key) == set(table.circuit_key)
    assert len(pred) == table.circuit_key.nunique() * table.decoder.nunique()
    code = table.groupby("circuit_key").code.first()
    assert (pred.groupby(pred.circuit_key.map(code)).fold.nunique() == 1).all()


def test_evaluate_and_report(table):
    metrics = evaluate(table, folds=3)
    json.dumps(metrics)
    assert metrics["setup"] == {"folds": 3, "group_by": "code", "groups": 6, "codes": 6, "circuits": 120,
                                "decodes": len(table), "split": "GroupKFold over code", "calibrated": False}
    assert metrics["per_group"] == {}
    assert 0 <= metrics["regime"]["accuracy"] <= 1 and 0 < metrics["regime"]["base_rate"] < 1
    assert metrics["ler"]["spearman"] > 0.8
    assert set(metrics["time"]["decoders"]) == {"all", "mwpm", "bposd", "tesseract"}
    assert metrics["time"]["timeouts"]["decodes"] == (table.status == "timeout").sum()
    policies = metrics["recommendation"]["policies"]
    assert set(policies) == {"recommend, accurate", "recommend, cheapest", "always bposd", "always tesseract"}
    for p in policies.values():
        assert 0 <= p["good_enough"] <= 1 and p["median_regret"] >= 1
    # Tesseract timed out on the large circuits, so always choosing it falls back there
    assert policies["always tesseract"]["fallbacks"] > 0
    report = report_markdown(metrics)
    assert report.startswith("# Matchmaker evaluation") and "| always bposd |" in report and "| tesseract |" in report


def test_hold_out_whole_families(table):
    # Three families of two codes each: a fold holds out a family, i.e. both of its codes
    families = table.assign(family=table.code.str[-1].astype(int).mod(3).map(lambda i: f"family{i}"))
    pred = cross_val_predictions(families, folds=10, group_by="family", n_jobs=2)
    family = families.groupby("circuit_key").family.first()
    assert pred.fold.nunique() == 3 and (pred.groupby(pred.circuit_key.map(family)).fold.nunique() == 1).all()
    metrics = evaluate(families, folds=10, group_by="family")
    assert metrics["setup"]["split"] == "leave-one-family-out" and set(metrics["per_group"]) == {"family0", "family1", "family2"}
    assert sum(m["circuits"] for m in metrics["per_group"].values()) == 120
    assert "## Per held-out code family" in report_markdown(metrics)
    with pytest.raises(ValueError):
        cross_val_predictions(families, group_by="decoder")


def test_extrapolation_splits_hold_out_beyond_the_training_range(table):
    first = table.groupby("circuit_key").first()
    largest = held_out_beyond(table, "largest")
    assert first.detectors[largest].min() > first.detectors[~largest].max()
    assert 0.15 < largest.mean() < 0.25
    quiet = table.assign(error_mass_per_detector=table.error_mass / table.detectors)
    quietest = held_out_beyond(quiet, "quietest")
    em = quiet.groupby("circuit_key").error_mass_per_detector.first()
    assert em[quietest].max() < em[~quietest].min()
    # Series of distances: each code's circuits at three distances, two slices; the largest of each is held out
    series = table.assign(distance=table.circuit_key.str.split("c").str[-1].astype(int) // 6 % 3 * 2 + 3, slice="s",
                          backend="b", error_type="e", error_probability=float("nan"))
    series = series.assign(cycles=series.distance, slice=np.where(series.circuit_key.str[-1].isin(list("02468")), "s0", "s1"))
    held = held_out_beyond(series, "next_distance")
    assert (series.groupby("circuit_key").distance.first()[held] == 7).all() and held.sum() == (series.groupby("circuit_key").distance.first() == 7).sum()
    pred = cross_val_predictions(table, group_by="largest")
    assert pred.fold.nunique() == 1 and set(pred.circuit_key) == set(largest.index[largest])
    metrics = evaluate(table, group_by="largest")
    assert metrics["setup"]["circuits"] == largest.sum() and metrics["setup"]["split"].startswith("the largest 20%")
    assert report_markdown(metrics).count("Extrapolation:") == 1
    trees = evaluate(table, group_by="largest", parametric=False)
    assert trees["setup"]["train_options"] == {"parametric": False}
    # The synthetic times are power laws in the size, which the time law extrapolates and the trees cannot
    assert metrics["time"]["decoders"]["all"]["within_3x"] > trees["time"]["decoders"]["all"]["within_3x"]


def test_operations_split_holds_out_the_logical_operations(table):
    keys = table.circuit_key.unique()
    ops = {k: ("transversal_cnot" if i % 5 == 0 else "ls_merge" if i % 7 == 0 else "memory") for i, k in enumerate(keys)}
    with_ops = table.assign(operation=table.circuit_key.map(ops))
    held = held_out_beyond(with_ops, "operations")
    assert set(held.index[held]) == {k for k, o in ops.items() if o != "memory"}
    assert not held_out_beyond(table, "operations").any()
    with pytest.raises(ValueError):
        cross_val_predictions(table, group_by="operations")
    pred = cross_val_predictions(with_ops, group_by="operations")
    assert pred.fold.nunique() == 1 and set(pred.circuit_key) == set(held.index[held])
    metrics = evaluate(with_ops, group_by="operations")
    assert set(metrics["per_group"]) == {"transversal_cnot", "ls_merge"}
    assert "memory experiments" in report_markdown(metrics)
