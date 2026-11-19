"""The dataset contract (eccentric.dataset) holds for the committed sweep table and its documentation"""
import json
import re
from pathlib import Path

import pytest

from eccentric import dataset

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def table():
    return dataset.load()


def test_committed_table_follows_the_schema(table):
    undocumented = [c for c in table.columns if c not in dataset.COLUMNS]
    assert not undocumented, f"document these columns in eccentric/dataset.py and docs/dataset.md: {undocumented}"


def test_rows_are_unique_per_configuration_and_decoder(table):
    assert not table.duplicated(subset=dataset.KEY).any()


def test_rows_with_rates_have_intervals(table):
    rates = dataset.with_rates(table)
    assert rates["ler_high"].notna().all()
    measured = rates[rates["status"] != "upper bound"]
    eps = 1e-9  # the CSV rounds an interval end of 1.0 (every shot a logical error)
    assert ((measured["ler_low"] <= measured["logical_error_rate"] + eps)
            & (measured["logical_error_rate"] <= measured["ler_high"] + eps)).all()


def test_docs_list_every_column_and_status():
    doc = (ROOT / "docs" / "dataset.md").read_text()
    documented = set(re.findall(r"^\| `([a-z_ ]+)` \|", doc, re.M))
    assert set(dataset.COLUMNS) <= documented
    assert set(dataset.STATUSES) <= documented
    assert f"Schema version: {dataset.SCHEMA_VERSION}" in doc


def test_stats_match_the_table(table):
    stats = json.loads((ROOT / "experiment_results" / "Sweep" / "stats.json").read_text())
    assert stats["schema_version"] == dataset.SCHEMA_VERSION
    assert stats["results"] == len(dataset.with_rates(table)), "run python -m sweep.stats"
