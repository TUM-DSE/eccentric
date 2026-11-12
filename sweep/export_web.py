"""Export a collected sweep for the website's result explorer:
python -m sweep.export_web <results.csv> <out.json>

The JSON holds labels for codes, devices, noise models and decoders, and the rows as arrays under
"columns" to keep the file small.
"""
import datetime
import json
import math
import os
import sys

import pandas as pd

FAMILIES = [
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

DEVICES = {
    "real_torino": "IBM Heron (133 qubits)",
    "real_willow": "Google Willow (extended, 315 qubits)",
    "real_loon_1_qpu": "IBM Loon/Nighthawk, 1 QPU (121 qubits)",
    "real_loon": "IBM Loon/Nighthawk, 2 QPUs (242 qubits)",
    "real_flamingo_1_qpu": "IBM Flamingo, 1 QPU (133 qubits)",
    "real_flamingo": "IBM Flamingo, 3 QPUs (399 qubits)",
    "real_bicycle": "IBM bicycle module: gross-code toric layout with long-range c-couplers (288 qubits)",
    "real_bicycle_lr": "IBM bicycle module with c-couplers as noisy as remote couplers (288 qubits)",
    "real_apollo": "Quantinuum Apollo (trapped ions, shuttling, 768 qubits)",
    "real_nsapollo": "Quantinuum Apollo without shuttling (768 qubits)",
    "real_infleqtion": "Infleqtion (neutral atoms, shuttling, 384 qubits)",
    "real_nsinfleqtion": "Infleqtion without shuttling (384 qubits)",
    "real_harvard": "Harvard/MIT neutral-atom array (moves, 448 qubits)",
    "custom_grid": "Square grid (300 qubits)",
    "custom_cube": "Cube (300 qubits)",
    "custom_full": "All-to-all connectivity",
    "native": "Native layout: a device built for the code, IBM's planned layout for bivariate bicycle codes",
    "native_lr": "Native layout with long-range couplers as noisy as IBM's remote couplers",
}

NOISE = {
    "real_heron": "IBM Heron",
    "real_flamingo": "IBM Flamingo",
    "real_flamingo_downscaled": "IBM Flamingo, 10x lower",
    "real_flamingo_m700": "IBM Flamingo, 700 ns readout",
    "real_flamingo_downscaled_m700": "IBM Flamingo, 10x lower, 700 ns readout",
    "sd6": "SD6 (uniform depolarizing)",
    "real_willow": "Google Willow (paper model)",
    "real_willow_spec": "Google Willow (spec sheet)",
    "real_apollo": "Quantinuum Apollo",
    "real_infleqtion": "Infleqtion",
    "real_harvard": "Harvard/MIT array",
    "modsi1000": "SI1000 (superconducting-inspired)",
}

def noise_label(error_type):
    """Label of a noise model, also of a scaled one (<noise>_x<factor>: every error probability times factor)"""
    base, _, factor = error_type.rpartition("_x")
    if base in NOISE and factor.replace(".", "", 1).isdigit():
        return f"{NOISE[base]}, errors x{factor}"
    return NOISE.get(error_type, error_type)


DECODERS = {
    "bposd": "BP-OSD", "bplsd": "BP-LSD", "relay_bp": "Relay-BP", "belief_find": "BeliefFind",
    "tesseract": "Tesseract", "mwpf": "MWPF", "gari": "GARI", "gari_pp": "GARI++ (beta reimplementation)", "bposd_tuned": "BP-OSD (tuned BP)", "mwpm": "MWPM", "chromobius": "Chromobius", "belief_matching": "BeliefMatching",
    "union_find": "Union-Find",
}

SLICES = {
    "qldpc_ibm": "qLDPC codes on IBM devices",
    "qldpc_ibm_projected": "qLDPC codes on larger IBM devices (projected)",
    "qldpc_bicycle": "IBM bicycle architecture module",
    "qldpc_bicycle_chain": "Chains of IBM bicycle modules",
    "bb_native": "Bivariate bicycle codes in IBM's native layout",
    "qldpc_atoms": "qLDPC codes on neutral atoms and trapped ions",
    "readout_700": "IBM devices with 700 ns readout",
    "willow_surface": "Surface code on Google Willow",
    "floquet_devices": "Floquet codes on devices",
    "noise_scaling": "Device noise scaled down",
    "basis_x": "Memory in the X basis",
    "reference": "Reference: all codes on an all-to-all device",
    "paper_dqc": "Paper experiment: one QPU vs distributed QPUs",
    "paper_topology": "Paper experiment: codes on sparse devices",
}
# One or two sentences per dataset, shown on the website under the dataset menu
SLICE_DESCRIPTIONS = {
    "qldpc_ibm": "qLDPC codes compiled onto IBM Heron, Loon/Nighthawk (one or two QPUs) and Flamingo (one or three "
                 "QPUs), at the devices' noise and, for Loon and Flamingo, at ten times lower noise.",
    "qldpc_ibm_projected": "qLDPC codes too large for any IBM device, on projected chains of as many Loon or Flamingo "
                           "QPUs as they need, at ten times lower noise.",
    "qldpc_bicycle": "qLDPC codes and the paper's codes on the 288-qubit module of IBM's bicycle architecture, with "
                     "its long-range couplers as good as local ones or as noisy as remote ones, at IBM noise and ten "
                     "times lower.",
    "qldpc_bicycle_chain": "qLDPC codes too large for one bicycle module, on as many modules as they need, linked by "
                           "remote couplers, at ten times lower noise.",
    "bb_native": "Bivariate bicycle codes placed one to one on IBM's planned layout for them, with long-range "
                 "couplers as good as local ones or as noisy as remote ones, to compare with the same codes "
                 "compiled onto Loon and Flamingo.",
    "qldpc_atoms": "qLDPC codes on neutral atoms (the Harvard/MIT array, Infleqtion) and Quantinuum's trapped ions, "
                   "at the devices' own noise. Any two qubits can interact after a move, so no SWAPs are needed.",
    "readout_700": "The codes of the IBM datasets again, with 700 ns readout instead of the 1.56 us of IBM's "
                   "calibration data, to see how much faster readout would help.",
    "willow_surface": "The surface code memory on Google Willow at distances 3 to 7, compiled by Qiskit and placed "
                      "on the grid as in Google's below-threshold experiment (Nature 638, 920, 2025), to compare "
                      "with its measurements.",
    "floquet_devices": "Floquet codes (honeycomb, planar honeycomb, hyperbolic and semi-hyperbolic) on IBM, Google, "
                       "trapped-ion and neutral-atom devices.",
    "noise_scaling": "Codes at several distances on IBM, Google and neutral-atom devices, with every error "
                     "probability of the device's noise multiplied by 0.1, 0.25, 0.5 or 1: where each device "
                     "crosses threshold, and how much better it must get.",
    "basis_x": "The same memory experiments with the logical |+> state prepared and measured in the X basis, "
               "paired with their Z-basis results: the Willow surface code, codes on IBM, Google and neutral-atom "
               "devices at their own noise, Bacon-Shor and XZZX in both bases, and asymmetric codes under uniform "
               "SD6 noise.",
    "reference": "Every code at its default schedule on an all-to-all device with uniform SD6 circuit noise: the "
                 "common baseline that makes the device datasets comparable.",
    "paper_dqc": "The configurations of the ECCentric paper's (SIGMETRICS 2026) comparison of one QPU with "
                 "distributed QPUs, rerun with the current, corrected simulator (noise model, Steane readout, "
                 "single-QPU devices) and decoded with BP-OSD as in the paper.",
    "paper_topology": "The configurations of the ECCentric paper's (SIGMETRICS 2026) experiments on device "
                      "topologies (grid, cube and all-to-all connectivity; Willow, Apollo and Infleqtion), rerun "
                      "with the current simulator, which places codes on a device instead of qubit by qubit and so "
                      "avoids needless SWAPs; decoded with BP-OSD as in the paper.",
}

COLUMNS = ["slice", "code", "family", "taxonomy", "schedule", "basis", "distance", "rounds", "k", "code_qubits", "backend",
           "error_type", "error_probability", "decoder", "status", "logical_error_rate", "ler_low", "ler_high",
           "ler_per_round", "ler_per_round_high", "shots", "errors", "circuit_qubits", "swaps", "two_qubit_gates",
           "layers", "device_qubits", "note", "undetectable_faults", "undetectable_probability"]


def device_label(backend):
    # Projected devices: real_loon_5_qpus is a chain of five Loon QPUs
    parts = backend.split("_")
    if len(parts) >= 4 and parts[-1] == "qpus":
        if parts[1] == "bicycle":
            noisy = ", noisy c-couplers" if parts[2] == "lr" else ""
            return f"IBM bicycle modules, {parts[-2]} chained{noisy} ({int(parts[-2]) * 288} qubits)"
        name, size = {"loon": ("IBM Loon/Nighthawk", 121), "flamingo": ("IBM Flamingo", 133)}[parts[1]]
        return f"{name}, {parts[2]} QPUs ({int(parts[2]) * size} qubits, projected)"
    return DEVICES.get(backend, backend)


def family(code):
    for prefix, name, taxonomy in FAMILIES:
        if code.startswith(prefix):
            return name, taxonomy
    return code, "Other"


def latest_commit(commits):
    """The most recent of the commits that produced results, by the repository's history"""
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history = subprocess.run(["git", "log", "--all", "--format=%H"], cwd=repo, capture_output=True, text=True).stdout.split()
    for full in history:
        for short in commits:
            if full.startswith(short):
                return short
    return sorted(commits)[-1] if len(commits) else None


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
    # Tables from before the memory basis was a column are all Z-basis memories
    df["basis"] = df["basis"].fillna("Z") if "basis" in df else "Z"
    rows = [[clean(r.get(c)) for c in COLUMNS] for r in df.to_dict("records")]
    out = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": latest_commit(df.commit.dropna().unique()),
        "complete": bool((df.status != "pending").all()),
        "labels": {"devices": {b: device_label(b) for b in df.backend.unique()}, "noise": {**NOISE, **{e: noise_label(e) for e in df.error_type.dropna().unique()}}, "decoders": DECODERS, "slices": SLICES,
                   "slice_descriptions": SLICE_DESCRIPTIONS},
        "columns": COLUMNS,
        "rows": rows,
    }
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(rows)} rows, {sum(df.status == 'pending')} pending")


if __name__ == "__main__":
    main()
