"""The figures (python -m eccentric.figures) build from the committed dataset, each with its data and description"""
import json

import pandas as pd

from eccentric.figures.__main__ import main


def test_every_figure_builds_with_data_and_text(tmp_path):
    main(["--out", str(tmp_path)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["source"]["schema_version"] == 1
    figures = manifest["figures"]
    assert len(figures) >= 12 and len({f["id"] for f in figures}) == len(figures)
    for f in figures:
        assert f["group"] in ("featured", "benchmark")
        assert f["title"] and f["finding"] and f["caption"]
        for name in f["files"].values():
            assert (tmp_path / name).stat().st_size > 0
        assert len(pd.read_csv(tmp_path / f["files"]["csv"])) > 0
