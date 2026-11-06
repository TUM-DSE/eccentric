"""Export the codes ECCentric benchmarks for the website's code page:
python -m sweep.export_codes <results.csv> <out.json>

For every code of the sweep table: its family, class and a description; at each distance the sweep
runs, the qubits of its syndrome circuit (data and ancilla), its logical qubits and check weights;
its schedules; the decoders that decoded it; its best results per dataset and on the reference
(all-to-all device, SD6 noise); and, at one distance, a drawing of the circuit's qubits with the
pairs its two-qubit gates (or two-qubit measurements) entangle.

Data and ancilla qubits are told apart in the circuit: an ancilla is measured on its own more than
once (every round), a data qubit once, at the end, or only in multi-qubit measurements (Floquet
codes measure pairs of data qubits directly). A check's weight is the number of data qubits its
ancilla entangles with, or the size of a multi-qubit measurement.
"""
import datetime
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import stim

from sweep.export_web import FAMILIES, DEVICES, NOISE, DECODERS, SLICES, device_label, noise_label

# One or two sentences per family
FAMILY_DESCRIPTIONS = {
    "Surface": "The rotated surface code: data qubits on a d by d square lattice, weight-4 checks (weight 2 on the "
               "boundary), one logical qubit. The standard other codes are measured against.",
    "XZZX surface": "The surface code with every check of the form XZZX, which suits noise biased towards one Pauli "
                    "error; the same lattice and check weights as the surface code.",
    "Toric": "The surface code on a torus: weight-4 checks and no boundary, two logical qubits.",
    "Repetition": "The classical repetition code, which protects against bit flips only: the simplest test of a "
                  "device and a decoder.",
    "Color": "The triangular color code on a hexagonal lattice: every face carries an X and a Z check of weight 6 "
             "(4 at the boundary); its logical Clifford gates are transversal.",
    "Quantum Reed-Muller": "The [[15,1,3]] quantum Reed-Muller code, the smallest code with a transversal T gate.",
    "Planar honeycomb": "A Floquet code with boundaries: its checks are two-qubit measurements on the edges of a "
                        "honeycomb lattice, in three alternating rounds.",
    "Honeycomb": "The honeycomb Floquet code (Hastings and Haah) on a torus: two-qubit checks measured in three "
                 "alternating rounds, which together measure weight-6 plaquettes.",
    "Hyperbolic Floquet": "Floquet codes on hyperbolic tilings: two-qubit checks like the honeycomb code, but many "
                          "logical qubits.",
    "Semi-hyperbolic Floquet": "Floquet codes on semi-hyperbolic tilings, between the honeycomb code and the "
                               "hyperbolic ones.",
    "Bacon-Shor": "A subsystem code on a square grid, measured through two-qubit gauge checks; its distance grows "
                  "with the grid, but it has no threshold.",
    "Concatenated Steane": "The [[7,1,3]] Steane code concatenated with itself: [[49,1,9]] at two levels and "
                           "[[343,1,27]] at three.",
    "Heavy-hex": "A hybrid of the surface and Bacon-Shor codes designed for IBM's heavy-hex lattice, where every "
                 "qubit has at most three neighbours.",
    "Bivariate bicycle": "Quasi-cyclic qLDPC codes from two polynomials in two commuting cyclic shifts (Bravyi et al., "
                         "Nature 2024): weight-6 checks, many logical qubits. The gross code [[144,12,12]] is the code "
                         "of IBM's bicycle architecture.",
    "Coprime bivariate bicycle": "Bivariate bicycle codes on cyclic groups of coprime sizes, which gives other code "
                                 "lengths than the original construction (built with the qLDPC library).",
    "Trivariate bicycle": "A generalization of the bivariate bicycle codes to three cyclic shifts (built with the "
                          "qLDPC library).",
    "Balanced product cyclic": "Codes from the balanced product of cyclic codes (built with the QUITS library).",
    "Generalized bicycle": "Codes from two commuting circulant matrices, the family the bivariate bicycle codes "
                           "generalize further.",
    "Hypergraph product": "The hypergraph product of two classical codes (Tillich and Zémor), the first qLDPC family "
                          "with a constant encoding rate.",
    "Lift-connected surface": "Copies of the surface code connected through a lifted product (built with the QUITS "
                              "library).",
    "Lifted product": "Lifted products of matrices over a group algebra (Panteleev and Kalachev), a qLDPC family that "
                      "includes good codes.",
    "Quasi-cyclic lifted product": "Lifted product codes with a quasi-cyclic structure (built with the QUITS library).",
    "Quantum Tanner": "Quantum Tanner codes (Leverrier and Zémor): checks from small local codes on a square complex, "
                      "a family of good qLDPC codes.",
    "Mitten": "A qLDPC code from a group algebra construction of the QUITS library.",
    "Many-hypercube": "High-rate concatenated codes built from small error-detecting codes on hypercubes (Goto).",
    "Quantum Golay": "The [[23,1,7]] quantum Golay code, the CSS code of the classical Golay code.",
}
DRAW_LIMIT = 700  # circuits with more qubits are not drawn


def family_of(code):
    for prefix, name, taxonomy in FAMILIES:
        if code.startswith(prefix):
            return name, taxonomy
    return code, "Other"


def circuit_of(code, d):
    from eccentric.codes.utils import get_code
    c = get_code(code, d, d)
    circuit = getattr(c, "stim_circuit", None)
    if circuit is None:
        circuit = c.circuit if isinstance(getattr(c, "circuit", None), stim.Circuit) else None
    return circuit.flattened() if circuit is not None else None


def anatomy(circuit):
    """Roles of the qubits, entangled pairs, and check weights of a memory circuit"""
    single = Counter()
    in_products = set()
    pairs, products = Counter(), []
    for inst in circuit:
        data = stim.gate_data(inst.name)
        targets = inst.targets_copy()
        if inst.name == "MPP":
            # Each product is a combiner-separated group of targets
            products.extend(_mpp_groups(targets))
            continue
        if data.produces_measurements and not data.is_two_qubit_gate:
            for t in targets:
                if t.is_qubit_target:
                    single[t.value] += 1
        elif data.is_two_qubit_gate and data.is_unitary:
            q = [t.value for t in targets if t.is_qubit_target]
            for a, b in zip(q[::2], q[1::2]):
                pairs[tuple(sorted((a, b)))] += 1
        elif data.is_two_qubit_gate and data.produces_measurements:  # MXX, MZZ, ...
            q = [t.value for t in targets if t.is_qubit_target]
            products.extend([list(p) for p in zip(q[::2], q[1::2])])
    for group in products:
        in_products.update(group)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs[tuple(sorted((group[i], group[j])))] += 1
    used = set(single) | in_products | {q for p in pairs for q in p}
    ancilla = {q for q in used if single[q] > 1 and q not in in_products}
    data_q = used - ancilla
    weights = []
    neighbours = defaultdict(set)
    for a, b in pairs:
        neighbours[a].add(b)
        neighbours[b].add(a)
    for q in ancilla:
        weights.append(len(neighbours[q] & data_q))
    weights += [len(g) for g in products]
    return {"used": sorted(used), "ancilla": ancilla, "data": data_q, "pairs": pairs, "weights": weights,
            "products": bool(products)}


def _mpp_groups(targets):
    groups, current = [], []
    for i, t in enumerate(targets):
        if t.is_combiner:
            continue
        current.append(t.value)
        if not (i + 1 < len(targets) and targets[i + 1].is_combiner):
            groups.append(current)
            current = []
    return groups


def positions(circuit, used, pairs):
    """2D positions: the circuit's qubit coordinates where every qubit has them, else a force-directed layout"""
    coords = circuit.get_final_qubit_coordinates()
    if all(q in coords and len(coords[q]) >= 2 for q in used):
        pts = {q: (coords[q][0], coords[q][1]) for q in used}
        xs = sorted({p[0] for p in pts.values()})
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 1e-9]
        unit = min(gaps) if gaps else 1.0
        return {q: (x / unit, y / unit) for q, (x, y) in pts.items()}, "coordinates"
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(used)
    g.add_edges_from(pairs)
    pos = nx.spring_layout(g, seed=1, iterations=300)
    scale = math.sqrt(len(used)) * 1.6
    return {q: (float(x) * scale, float(y) * scale) for q, (x, y) in pos.items()}, "layout"


def results_of(df, code):
    """Best results of a code: per dataset, and on the reference slice by distance and physical error rate"""
    rows = df[(df.code == code) & df.status.isin(["measured", "low statistics", "upper bound"])].copy()
    rows["rate"] = np.where(rows.errors > 0, rows.errors / rows.shots, rows.ler_high)
    rows["bound"] = rows.errors == 0
    out = {"datasets": [], "reference": []}
    for s, g in rows.groupby("slice"):
        configs = g.drop_duplicates(["distance", "cycles", "backend", "error_type", "error_probability", "schedule"])
        rated = g[g.ler_per_round.notna() & (g.errors > 0)]
        best = rated.loc[rated.ler_per_round.idxmin()] if len(rated) else None
        out["datasets"].append({
            "slice": s, "label": SLICES.get(s, s), "configurations": int(len(configs)),
            "best": None if best is None else {
                "ler_per_round": float(best.ler_per_round), "ler": float(best.rate), "distance": int(best.distance),
                "rounds": int(best.cycles), "device": device_label(best.backend), "noise": NOISE.get(best.error_type) or noise_label(best.error_type),
                "error_probability": None if pd.isna(best.error_probability) else float(best.error_probability),
                "decoder": DECODERS.get(best.decoder, best.decoder)}})
    ref = rows[rows.slice == "reference"]
    for (d, p), g in ref.groupby(["distance", "error_probability"]):
        b = g.loc[g.rate.idxmin()]
        out["reference"].append({"distance": int(d), "p": float(p), "ler": float(b.rate), "bound": bool(b.bound),
                                 "decoder": DECODERS.get(b.decoder, b.decoder),
                                 "ler_per_round": None if pd.isna(b.ler_per_round) else float(b.ler_per_round)})
    decoders = sorted(rows.decoder.unique(), key=lambda x: list(DECODERS).index(x) if x in DECODERS else 99)
    out["decoders"] = [DECODERS.get(x, x) for x in decoders]
    na = df[(df.code == code) & (df.status == "decoder not applicable")].decoder.unique()
    out["not_applicable"] = [DECODERS.get(x, x) for x in na]
    out["devices"] = sorted({device_label(b) for b in df[(df.code == code) & (df.status != "does not fit")].backend.unique()})
    return out


def main():
    from sweep.slices import code_distances, schedules
    results_csv, out = sys.argv[1], sys.argv[2]
    df = pd.read_csv(results_csv, low_memory=False)
    df["k"] = df.observables  # the logical qubits a memory experiment tracks, as the website export counts them
    codes = sorted(df.code.unique(), key=lambda c: (family_of(c)[1], family_of(c)[0], len(c), c))
    entries = []
    for code in codes:
        name, taxonomy = family_of(code)
        try:
            distances = code_distances(code)
        except KeyError:
            distances = sorted(int(x) for x in df[df.code == code].distance.dropna().unique())
        rows = df[df.code == code]
        k = rows.k.dropna()
        sizes, drawing = [], None
        draw_d = distances[len(distances) // 2]
        for d in distances:
            try:
                circuit = circuit_of(code, d)
            except Exception as e:  # noqa: BLE001
                print(f"{code} d={d}: {type(e).__name__}: {e}")
                continue
            if circuit is None:
                continue
            a = anatomy(circuit)
            w = a["weights"]
            kd = rows[rows.distance == d].k.dropna()
            sizes.append({"distance": d, "data": len(a["data"]), "ancilla": len(a["ancilla"]),
                          "k": int(kd.iloc[0]) if len(kd) else (int(k.iloc[0]) if len(k) else None),
                          "check_weight_max": max(w) if w else None, "check_weight_mean": round(float(np.mean(w)), 2) if w else None,
                          "checks": len(w), "pair_measurements": a["products"]})
            if d == draw_d and len(a["used"]) <= DRAW_LIMIT:
                pos, how = positions(circuit, a["used"], a["pairs"])
                index = {q: i for i, q in enumerate(a["used"])}
                drawing = {"distance": d, "how": how,
                           "positions": [[round(pos[q][0], 3), round(pos[q][1], 3)] for q in a["used"]],
                           "roles": ["ancilla" if q in a["ancilla"] else "data" for q in a["used"]],
                           "edges": [[index[x], index[y]] for (x, y) in sorted(a["pairs"])]}
        entries.append({"id": code, "family": name, "taxonomy": taxonomy, "description": FAMILY_DESCRIPTIONS.get(name, ""),
                        "schedules": [s for s in schedules(code) if s], "sizes": sizes, "drawing": drawing,
                        **results_of(df, code)})
        print(f"{code}: {[(s['distance'], s['data'], s['ancilla'], s['k']) for s in sizes]}, drawing {'yes' if drawing else 'no'}")
    with open(out, "w") as f:
        json.dump({"generated": datetime.date.today().isoformat(), "codes": entries}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
