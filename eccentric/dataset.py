"""The ECCentric dataset: the sweep table `experiment_results/Sweep/results.csv`, one row per configuration and
decoder. This module is its contract for tools built on it (see docs/dataset.md): the columns, what they mean,
the statuses a row can have, and a schema version that changes whenever a column is renamed, removed or
changes meaning. Adding a column does not change the version.

    from eccentric import dataset
    table = dataset.load()              # every row
    rates = dataset.with_rates(table)   # rows with a logical error rate or an upper bound
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

# Re-timing of the decoders that take microseconds per shot (mwpm, mwpm_corr, chromobius) with 10^5 shots per configuration on one host
# (sweep/retime_matching.py): the sweep's own times for decoders this fast are mostly per-batch overhead
MATCHING_TIMING_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "experiment_results" / "Sweep" / "matching_timing.csv"

DEFAULT_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "experiment_results" / "Sweep" / "results.csv"

# Configuration: together with `decoder`, these identify a row
CONFIG = {
    "slice": "Part of the sweep the row belongs to (sweep/slices.py)",
    "code": "Code name (docs/codes.md)",
    "schedule": "Requested syndrome-extraction schedule; empty for the code's default",
    "distance": "Code distance",
    "cycles": "Requested syndrome-extraction rounds; empty for the default (usually the distance)",
    "backend": "Device (docs/devices.md)",
    "backend_size": "Qubits of a synthetic (`custom_*`) device",
    "error_type": "Noise model (docs/noise-models.md)",
    "error_probability": "Physical error rate of an artificial noise model",
    "layout_method": "Qiskit layout method",
    "routing_method": "Qiskit routing method",
    "translating_method": "Gate-set translation method",
    "basis": "Memory basis, Z or X",
    "operation": "`memory`, or the fault-tolerant operation (docs/codes.md)",
}
DECODER = {"decoder": "Decoder (docs/decoders.md)"}
# The compiled circuit
RESOURCES = {
    "schedule_used": "Schedule the circuit was built with",
    "rounds": "Syndrome-extraction rounds the circuit runs",
    "code_qubits": "Qubits of the code's circuit before compilation",
    "device_qubits": "Qubits of the device",
    "circuit_qubits": "Qubits of the compiled circuit",
    "detectors": "Detectors of the compiled circuit",
    "observables": "Logical observables of the compiled circuit",
    "two_qubit_gates": "Two-qubit gates of the compiled circuit, SWAPs included",
    "swaps": "SWAPs inserted by routing",
    "layers": "Layers (moments) of the compiled circuit",
    "measurements": "Measurements of the compiled circuit",
    "undetectable_faults": "Single faults that flip an observable without firing a detector",
    "undetectable_probability": "Total probability of those faults",
}
# Where the row came from
PROVENANCE = {
    "commit": "ECCentric commit that built the circuit or decoded it",
    "run": "Sweep run directory",
    "job_id": "Job in that run",
    "result_file": "Result file of the decode",
}
# The decode
RESULT = {
    "shots": "Shots decoded",
    "errors": "Logical errors among them",
    "decode_seconds": "CPU seconds of the decode, decoder construction included",
    "refined_chunks": "Extra chunks of shots added by refinement runs",
    "status": "How far the rate can be trusted (STATUSES)",
    "logical_error_rate": "Logical errors / shots over all rounds; empty for an upper bound",
    "ler_low": "Lower end of the 95% Wilson interval",
    "ler_high": "Upper end of the 95% Wilson interval; for an upper bound, the bound",
    "ler_per_round": "Per-round rate (1 - (1 - 2 LER)^(1/rounds)) / 2; empty at LER >= 0.5",
    "ler_per_round_high": "Per-round rate of ler_high",
    "note": "Why a row has no rate (error message, timeout)",
}
COLUMNS = {**CONFIG, **DECODER, **RESOURCES, **PROVENANCE, **RESULT}

STATUSES = {
    "measured": "At least the target number of logical errors (100) seen",
    "low statistics": "Stopped by the shot or time limit before the target number of errors",
    "upper bound": "No logical error seen; ler_high is the 95% upper bound",
    "does not fit": "The code does not fit on the device",
    "decoder not applicable": "The decoder refused the circuit (e.g. matching on non-decomposable errors)",
    "timeout": "Not a single batch of shots decoded within the time limit",
    "failed": "The decode failed for another reason",
    "pending": "Not decoded yet",
}
WITH_RATE = ("measured", "low statistics", "upper bound")

KEY = list(CONFIG) + list(DECODER)

# Code families by name prefix (first match wins): (prefix, family, kind)
FAMILIES = [
    ("ls_", "Lattice surgery (surface-code patches)", "Topological"),
    ("bb", "Bivariate bicycle", "QLDPC"), ("gross", "Bivariate bicycle", "QLDPC"),
    ("cbb", "Coprime bivariate bicycle", "QLDPC"), ("tb", "Trivariate bicycle", "QLDPC"),
    ("bpc", "Balanced product cyclic", "QLDPC"), ("gb", "Generalized bicycle", "QLDPC"),
    ("hgp", "Hypergraph product", "QLDPC"), ("lcs", "Lift-connected surface", "QLDPC"),
    ("lp", "Lifted product", "QLDPC"), ("qlp", "Quasi-cyclic lifted product", "QLDPC"),
    ("qt", "Quantum Tanner", "QLDPC"), ("mitten", "Mitten", "QLDPC"),
    ("surface", "Surface", "Topological"), ("xzzx", "XZZX surface", "Topological"),
    ("toric", "Toric", "Topological"), ("repetition", "Repetition", "Topological"),
    ("color", "Color", "Topological"), ("rm15", "Quantum Reed-Muller", "Topological"),
    ("planar_honeycomb", "Planar honeycomb", "Floquet"), ("honeycomb", "Honeycomb", "Floquet"),
    ("semi_hyperbolic", "Semi-hyperbolic Floquet", "Floquet"), ("hyperbolic", "Hyperbolic Floquet", "Floquet"),
    ("bacon", "Bacon-Shor", "Subsystem"), ("steane", "Concatenated Steane", "Concatenated"),
    ("mhc", "Many-hypercube", "Concatenated"), ("golay", "Quantum Golay", "Algebraic"),
    ("hh", "Heavy-hex", "Hybrid"),
]


def family(code: str):
    """(family, kind) of a code name, e.g. ("Bivariate bicycle", "QLDPC") for bb72"""
    for prefix, name, kind in FAMILIES:
        if code.startswith(prefix):
            return name, kind
    return code, "Other"


def load(path=DEFAULT_PATH, check: bool = True) -> pd.DataFrame:
    """The sweep table. With `check`, raise if a documented column is missing or a status is unknown"""
    table = pd.read_csv(path, low_memory=False)
    if check:
        validate(table)
    return table


def validate(table: pd.DataFrame):
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"dataset is missing the columns {missing} (schema version {SCHEMA_VERSION})")
    unknown = set(table["status"].dropna()) - set(STATUSES)
    if unknown:
        raise ValueError(f"dataset has unknown statuses {sorted(unknown)}")


def with_matching_times(table: pd.DataFrame, path=MATCHING_TIMING_PATH) -> pd.DataFrame:
    """The table with the re-timed ms per shot of mwpm, mwpm_corr and chromobius in `ms_per_shot` (NaN for the other decoders
    and configurations not re-timed), if the re-timing exists"""
    table = table.copy()
    if not os.path.exists(path):
        table["ms_per_shot"] = np.nan
        return table
    times = pd.read_csv(path, low_memory=False)[KEY + ["ms_per_shot"]]
    # A column that is empty for every re-timed configuration reads as float: give each key the table's type
    for col in KEY:
        if table[col].dtype == object:
            times[col] = times[col].astype(object).where(times[col].notna(), None)
            table[col] = table[col].where(table[col].notna(), None)
        else:
            times[col] = pd.to_numeric(times[col], errors="coerce").astype(table[col].dtype)
    return table.merge(times, on=KEY, how="left")


def with_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Rows with a logical error rate, or an upper bound on it"""
    return table[table["status"].isin(WITH_RATE)]
