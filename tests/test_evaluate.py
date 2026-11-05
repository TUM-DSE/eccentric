import json

import pytest

from qec_matchmaker.evaluate import cross_val_predictions, evaluate, report_markdown
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
