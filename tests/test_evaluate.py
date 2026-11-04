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
    assert metrics["setup"] == {"folds": 3, "codes": 6, "circuits": 120, "decodes": len(table),
                                "split": "GroupKFold over codes", "calibrated": False}
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
