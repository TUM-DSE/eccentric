"""The training table: ECCentric's measurements joined with the features of their circuits.

One row per circuit and decoder (see docs/DESIGN.md for the columns). ECCentric's collected sweep table
(`experiment_results/Sweep/results.csv`) names, for each row, the run and job it came from; the circuit
a decode used is the run's `circuits/<circuit job id>.stim`, whose features the feature store holds.
"""
import datetime
import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, FEATURE_VERSION

KEYS = ["circuit_key", "code", "family", "distance", "cycles", "backend", "error_type", "error_probability", "slice", "basis",
        "decoder"]
LABELS = ["status", "shots", "errors", "ler", "ler_low", "ler_high", "ms_per_shot", "timeout_s"]
SKIPPED = ("does not fit", "pending")
# ECCentric's configuration columns (sweep/collect.py CONFIG): one circuit per configuration
ECCENTRIC_CONFIG = ["slice", "code", "schedule", "distance", "cycles", "backend", "backend_size", "error_type",
                    "error_probability", "layout_method", "routing_method", "translating_method", "basis"]

# Code families by name prefix, as ECCentric's website groups them (first match wins)
FAMILIES = [
    ("cbb", "Coprime bivariate bicycle"), ("tb", "Trivariate bicycle"), ("bb", "Bivariate bicycle"),
    ("gross", "Bivariate bicycle"), ("bpc", "Balanced product cyclic"), ("gb", "Generalized bicycle"),
    ("hgp", "Hypergraph product"), ("lcs", "Lift-connected surface"), ("lp", "Lifted product"),
    ("qlp", "Quasi-cyclic lifted product"), ("qt", "Quantum Tanner"), ("mitten", "Mitten"),
    ("surface", "Surface"), ("xzzx", "XZZX surface"), ("toric", "Toric"), ("repetition", "Repetition"),
    ("color", "Color"), ("rm15", "Quantum Reed-Muller"), ("planar_honeycomb", "Planar honeycomb"),
    ("honeycomb", "Honeycomb"), ("semi_hyperbolic", "Semi-hyperbolic Floquet"), ("hyperbolic", "Hyperbolic Floquet"),
    ("bacon", "Bacon-Shor"), ("steane", "Concatenated Steane"), ("hh", "Heavy-hex"), ("mhc", "Many-hypercube"),
    ("golay", "Quantum Golay"),
]


def family(code: str) -> str:
    return next((name for prefix, name in FAMILIES if code.startswith(prefix)), code)


def circuit_key(run: str, job_id: str) -> str:
    """The feature-store key of the circuit an ECCentric decode job used"""
    return f"{run}/" + re.sub(r"\.decoder=.*$", "", re.sub(r"^decode\.", "circuit.", job_id))


def _timeout_seconds(note) -> float:
    m = re.match(r"timeout after ([\d.]+) (h|s)", str(note))
    return float(m.group(1)) * (3600 if m.group(2) == "h" else 1) if m else math.nan


def from_eccentric(results: pd.DataFrame, store: dict) -> tuple[pd.DataFrame, dict]:
    """The training table from ECCentric's collected results and a feature store {circuit key: features};
    also returns counts of rows left out and why"""
    rows = results[~results.status.isin(SKIPPED) & results.result_file.notna()].copy()
    rows["circuit_key"] = [circuit_key(r, j) for r, j in zip(rows.run, rows.job_id)]
    # A configuration is one circuit, but its decodes can come from several runs (a rerun after a fix, a decoder
    # added later decode the original run's circuit file under their own run's name): one key per configuration
    # Memory basis (Z or X): tables from before ECCentric had the setting are all Z
    rows["basis"] = rows["basis"].fillna("Z") if "basis" in rows.columns else "Z"
    config = [c for c in ECCENTRIC_CONFIG if c in rows.columns]
    if config:
        group = rows[config].apply(lambda c: c.map(str)).agg("|".join, axis=1)
        rows["circuit_key"] = rows.groupby(group).circuit_key.transform("min")
    rows["family"] = rows.code.map(family)
    rows["ler"] = np.where(rows.shots > 0, rows.errors / rows.shots, math.nan)
    rows["ms_per_shot"] = np.where(rows.shots > 0, 1000 * rows.decode_seconds / rows.shots, math.nan)
    rows["timeout_s"] = [_timeout_seconds(n) if s == "timeout" else math.nan for s, n in zip(rows.status, rows.note)]
    known = rows.circuit_key.isin(store.keys())
    features = pd.DataFrame([store[k] for k in rows.circuit_key[known]], index=rows.index[known])
    table = rows[known][KEYS + LABELS].join(features.reindex(columns=FEATURE_NAMES))
    left_out = {"skipped (no circuit)": int(results.status.isin(SKIPPED).sum()), "no features yet": int((~known).sum())}
    return table.reset_index(drop=True), left_out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_dataset(table: pd.DataFrame, out_dir: str, sources: dict, left_out: dict) -> dict:
    """Write the table and its manifest; sources maps a name to {"path": ..., plus any provenance}"""
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
    manifest = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "features": FEATURE_NAMES,
        "rows": len(table),
        "circuits": int(table.circuit_key.nunique()),
        "codes": int(table.code.nunique()),
        "decoders": sorted(table.decoder.unique()),
        "status": table.status.value_counts().to_dict(),
        "left_out": left_out,
        "sources": {name: {**info, "sha256": _sha256(info["path"]) if os.path.exists(info.get("path", "")) else None}
                    for name, info in sources.items()},
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest


def load_dataset(data_dir: str) -> tuple[pd.DataFrame, dict]:
    with open(os.path.join(data_dir, "MANIFEST.json")) as f:
        manifest = json.load(f)
    return pd.read_csv(os.path.join(data_dir, "dataset.csv"), low_memory=False), manifest
