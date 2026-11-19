"""Headline numbers of the sweep table, for the README badges and the website.

    python -m sweep.stats [results.csv] [stats.json]

Writes experiment_results/Sweep/stats.json by default. The README's badges read it through
shields.io, so they change whenever a new table is committed to main. Run it after every
`sweep.collect`.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from eccentric.codes.catalog import CODES

PAPER_CODES = ["surface", "hh", "gross", "color", "bacon", "steane"]
# Rows with a logical error rate (the others did not fit their device or the decoder refused them)
RESULT_STATUSES = {"measured", "low statistics", "upper bound"}


def _short(x: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"{x / div:.1f}{suffix}"
    return f"{x:.0f}"


def stats(table: pd.DataFrame) -> dict:
    res = table[table["status"].isin(RESULT_STATUSES)]
    framework_codes = len(set(PAPER_CODES) | set(CODES))
    s = {
        "codes": framework_codes,
        "codes_with_results": int(res["code"].nunique()),
        "devices": int(res["backend"].nunique()),
        "noise_models": int(res["error_type"].nunique()),
        "decoders": int(res["decoder"].nunique()),
        "results": int(len(res)),
        "configurations": int(res.drop_duplicates(
            ["code", "schedule", "distance", "cycles", "backend", "backend_size", "error_type",
             "error_probability", "basis", "operation"]).shape[0]),
        "shots": int(table["shots"].sum()),
        "cpu_hours": round(float(table["decode_seconds"].sum()) / 3600),
    }
    s["display"] = {
        "results": f"{s['results']:,}",
        "shots": _short(s["shots"]),
        "cpu_hours": _short(s["cpu_hours"]),
    }
    return s


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0] if argv else "experiment_results/Sweep/results.csv")
    out = Path(argv[1] if len(argv) > 1 else src.with_name("stats.json"))
    s = stats(pd.read_csv(src, low_memory=False))
    out.write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
