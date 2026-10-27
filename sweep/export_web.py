"""Export a collected sweep for the website's result explorer:
python -m sweep.export_web <results.csv> <out.json>

The JSON holds labels for codes, devices, noise models and decoders, and the rows as arrays under
"columns" to keep the file small.
"""
import datetime
import json
import math
import sys

import pandas as pd

FAMILIES = [
    ("bb", "Bivariate bicycle", "QLDPC"), ("gross", "Bivariate bicycle", "QLDPC"),
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
    ("hh", "Heavy-hex", "Hybrid"),
]

DEVICES = {
    "real_torino": "IBM Heron (133 qubits)",
    "real_willow": "Google Willow (extended, 315 qubits)",
    "real_loon_1_qpu": "IBM Loon/Nighthawk, 1 QPU (120 qubits)",
    "real_loon": "IBM Loon/Nighthawk, 2 QPUs (242 qubits)",
    "real_flamingo_1_qpu": "IBM Flamingo, 1 QPU (133 qubits)",
    "real_flamingo": "IBM Flamingo, 3 QPUs (399 qubits)",
    "custom_full": "All-to-all connectivity",
    "native": "Native layout: a device built for the code, IBM's planned layout for bivariate bicycle codes",
    "native_lr": "Native layout with long-range couplers as noisy as IBM's remote couplers",
}

NOISE = {
    "real_heron": "IBM Heron",
    "real_flamingo": "IBM Flamingo",
    "real_flamingo_downscaled": "IBM Flamingo, 10x lower",
    "sd6": "SD6 (uniform depolarizing)",
    "real_willow": "Google Willow (paper model)",
    "real_willow_spec": "Google Willow (spec sheet)",
}

DECODERS = {
    "bposd": "BP-OSD", "bplsd": "BP-LSD", "relay_bp": "Relay-BP", "belief_find": "BeliefFind",
    "tesseract": "Tesseract", "mwpm": "MWPM", "chromobius": "Chromobius", "belief_matching": "BeliefMatching",
    "union_find": "Union-Find",
}

SLICES = {
    "qldpc_ibm": "QLDPC codes on IBM devices",
    "qldpc_ibm_projected": "QLDPC codes on projected larger IBM devices",
    "bb_native": "Bivariate bicycle codes in IBM's native layout",
    "paper_dqc": "Paper: one QPU vs distributed QPUs, re-run with the fixes",
    "willow_surface": "Surface code on Google Willow, against Google's experiment",
    "reference": "Reference: all codes, all-to-all, SD6 noise",
}

COLUMNS = ["slice", "code", "family", "taxonomy", "schedule", "distance", "rounds", "k", "code_qubits", "backend",
           "error_type", "error_probability", "decoder", "status", "logical_error_rate", "ler_low", "ler_high",
           "ler_per_round", "ler_per_round_high", "shots", "errors", "circuit_qubits", "swaps", "two_qubit_gates",
           "layers", "device_qubits", "note"]


def device_label(backend):
    # Projected devices: real_loon_5_qpus is a chain of five Loon QPUs
    parts = backend.split("_")
    if len(parts) == 4 and parts[3] == "qpus":
        name, size = {"loon": ("IBM Loon/Nighthawk", 121), "flamingo": ("IBM Flamingo", 133)}[parts[1]]
        return f"{name}, {parts[2]} QPUs ({int(parts[2]) * size} qubits, projected)"
    return DEVICES.get(backend, backend)


def family(code):
    for prefix, name, taxonomy in FAMILIES:
        if code.startswith(prefix):
            return name, taxonomy
    return code, "Other"


def clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float):
        return float(f"{value:.6g}")
    return value.item() if hasattr(value, "item") else value


def main():
    df = pd.read_csv(sys.argv[1])
    df["family"], df["taxonomy"] = zip(*df.code.map(family))
    df["schedule"] = df.schedule_used.fillna(df.schedule)
    df["k"] = df.observables
    rows = [[clean(r.get(c)) for c in COLUMNS] for r in df.to_dict("records")]
    out = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commits": sorted(df.commit.dropna().unique().tolist()),
        "complete": bool((df.status != "pending").all()),
        "labels": {"devices": {b: device_label(b) for b in df.backend.unique()}, "noise": NOISE, "decoders": DECODERS, "slices": SLICES},
        "columns": COLUMNS,
        "rows": rows,
    }
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(rows)} rows, {sum(df.status == 'pending')} pending")


if __name__ == "__main__":
    main()
