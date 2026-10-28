"""Export the devices ECCentric supports for the website's device page:
python -m sweep.export_devices <out.json>

For every device: where its qubits sit (in units of the qubit pitch), its couplers by kind (local,
long-range within a QPU, between QPUs, wrapping around a torus), the T1 and T2 of every qubit, and
the parameters of the noise models it is simulated with.
"""
import datetime
import json
import sys
from collections import defaultdict

import numpy as np

from eccentric.backends.utils import get_backend
from eccentric.noise.utils import get_noise_model
from sweep.export_web import DEVICES

# (backend, size of a custom backend, label, maker, noise models, description)
CATALOG = [
    ("real_torino", None, "IBM Heron r1 (Torino)", "IBM", ["real_heron"],
     "IBM's 133-qubit Heron processor: a heavy-hex lattice, every qubit coupled to two or three neighbours."),
    ("real_flamingo_1_qpu", None, "IBM Flamingo, 1 QPU", "IBM", ["real_flamingo", "real_flamingo_downscaled"],
     "A Flamingo QPU as modelled here: the 127-qubit Eagle heavy-hex lattice extended to 133 qubits."),
    ("real_flamingo", None, "IBM Flamingo, 3 QPUs", "IBM", ["real_flamingo", "real_flamingo_downscaled"],
     "Three Flamingo QPUs in a chain, each linked to the next by four l-couplers (remote, 3% error, 300 ns)."),
    ("real_loon_1_qpu", None, "IBM Loon/Nighthawk, 1 QPU", "IBM", ["real_flamingo", "real_flamingo_downscaled"],
     "An 11 by 11 square lattice with long-range c-couplers (IBM's large-scale fault-tolerance roadmap). "
     "As modelled, the c-couplers start from the first row of the lattice only."),
    ("real_loon", None, "IBM Loon/Nighthawk, 2 QPUs", "IBM", ["real_flamingo", "real_flamingo_downscaled"],
     "Two Loon QPUs linked by four l-couplers."),
    ("real_bicycle", None, "IBM bicycle architecture module", "IBM", ["real_flamingo", "real_flamingo_downscaled"],
     "The 288 qubits of the gross code in its toric layout: four couplers to neighbours on a torus and two "
     "long-range c-couplers per qubit, all as good as local couplers."),
    ("real_bicycle_lr", None, "IBM bicycle module, noisy c-couplers", "IBM", ["real_flamingo", "real_flamingo_downscaled"],
     "The bicycle architecture module with its c-couplers as noisy and slow as remote couplers."),
    ("real_bicycle_2_qpus", None, "IBM bicycle modules, 2 chained", "IBM", ["real_flamingo_downscaled"],
     "Two bicycle architecture modules linked by four remote couplers, for codes larger than one module."),
    ("real_willow", None, "Google Willow (extended)", "Google", ["real_willow", "real_willow_spec"],
     "Willow's square lattice in the shape of a diamond, extended to 315 qubits."),
    ("real_apollo", None, "Quantinuum Apollo", "Quantinuum", ["real_apollo"],
     "Trapped ions: neighbours on the grid interact in place, any other pair after moving an ion (shuttling)."),
    ("real_nsapollo", None, "Quantinuum Apollo, no shuttling", "Quantinuum", ["real_apollo"],
     "Apollo restricted to nearest-neighbour gates."),
    ("real_infleqtion", None, "Infleqtion (neutral atoms)", "Infleqtion", ["real_infleqtion"],
     "Neutral atoms on a grid: neighbours interact in place, any other pair after moving an atom."),
    ("real_nsinfleqtion", None, "Infleqtion, no shuttling", "Infleqtion", ["real_infleqtion"],
     "Infleqtion's array restricted to nearest-neighbour gates."),
    ("real_harvard", None, "Harvard/MIT neutral-atom array", "Harvard/MIT", ["real_harvard"],
     "The reconfigurable 448-atom array of Bluvstein et al. (Nature 2025): any pair interacts after a move, "
     "which costs a gate error and atom loss."),
    ("custom_grid", 289, "Square grid", "Synthetic", ["sd6", "modsi1000"], "A 17 by 17 square lattice."),
    ("custom_diamond", 313, "Diamond grid", "Synthetic", ["sd6", "modsi1000"],
     "A square lattice in the shape of a diamond, like Willow's, into which the rotated surface code fits."),
    ("custom_heavyhex", 290, "Heavy-hex", "Synthetic", ["sd6", "modsi1000"], "A heavy-hex lattice like IBM's."),
    ("custom_cube", 300, "Cube", "Synthetic", ["sd6", "modsi1000"], "A 5 by 6 by 10 cubic lattice."),
    ("custom_full", 300, "All-to-all", "Synthetic", ["sd6", "modsi1000"], "Every pair of qubits is coupled."),
]

NOISE_LABELS = {
    "real_heron": "IBM Heron (calibration averages)",
    "real_flamingo": "IBM (Flamingo/Loon targets)",
    "real_flamingo_downscaled": "IBM, tenfold lower",
    "real_willow": "Google Willow (Nature 2025)",
    "real_willow_spec": "Google Willow (spec sheet)",
    "real_apollo": "Quantinuum Apollo",
    "real_infleqtion": "Infleqtion",
    "real_harvard": "Harvard/MIT array",
    "sd6": "SD6 (standard depolarizing)",
    "modsi1000": "SI1000 (superconducting-inspired)",
}

# Noise model attribute -> label, for the parameters shown on the page
NOISE_PARAMS = [
    ("sq", "1-qubit gate"), ("tq", "2-qubit gate"), ("measure", "Measurement"), ("reset", "Reset"),
    ("idle", "Idle"), ("crosstalk", "Crosstalk"), ("leakage", "Leakage"), ("remote", "Remote gate"),
    ("shuttle", "Move (shuttling)"), ("loss", "Loss per gate"), ("loss_remote", "Loss per move"),
    ("loss_measure", "Loss per measurement"),
]
GATE_TIMES = [("SQ", "1-qubit gate"), ("TQ", "2-qubit gate"), ("M", "Measurement"), ("R", "Reset"), ("REMOTE", "Remote gate / move")]


def heavy_hex_positions(edges, n):
    """Coordinates of a heavy-hex lattice numbered row by row as IBM numbers its devices: rows of
    consecutive qubits, joined by single bridge qubits"""
    adjacent = defaultdict(set)
    for a, b in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    runs, run = [], [0]
    for q in range(1, n):
        if q - 1 in adjacent[q]:
            run.append(q)
        else:
            runs.append(run)
            run = [q]
    runs.append(run)
    pos, row, offset = {}, 0, 0
    pending = []  # bridge qubits waiting for the next row
    for run in runs:
        if len(run) < 4:
            pending += run
            continue
        if pending:
            # Align the row so that each bridge sits right above its neighbour in this row
            for b in pending:
                below = [q for q in adjacent[b] if q in run]
                if below:
                    offset = pos[b][0] - run.index(below[0])
                    break
            row += 2
        for i, q in enumerate(run):
            pos[q] = (offset + i, row)
        pending = []
        # Bridges below this row hang under their upper neighbour
        for q in run:
            for b in adjacent[q]:
                if b not in pos and len(adjacent[b]) <= 2 and all(c in run or c > q for c in adjacent[b]):
                    pos[b] = (pos[q][0], row + 1)
    return pos


def heavy_hex_code_positions(edges, n):
    """Coordinates of Qiskit's heavy-hex coupling map (CouplingMap.from_heavy_hex), numbered as the
    heavy-hex code's qubits: the layout of the code, matched to the coupling map by isomorphism"""
    import rustworkx as rx

    d = int(round((2 + np.sqrt(4 + 40 * n + 20)) / 10))  # n = (5 d^2 - 2 d - 1) / 2
    coords = [(r, 2 * c) for c in range(d) for r in range(2 * d - 1)]  # data and flag qubits in rows
    index = {p: i for i, p in enumerate(coords)}
    lattice = [(index[(r, 2 * c)], index[(r + 1, 2 * c)]) for c in range(d) for r in range(2 * d - 2)]

    def syndrome(p, a, b):
        index[p] = len(coords)
        coords.append(p)
        lattice.extend([(index[p], index[a]), (index[p], index[b])])
    for r in range(1, 2 * d - 1, 2):
        for c in range(d - 1):
            if (c + r // 2) % 2 == 0:
                syndrome((r, 2 * c + 1), (r, 2 * c), (r, 2 * c + 2))
    for c in range(d - 1):
        if c % 2 == 1:
            syndrome((0, 2 * c + 1), (0, 2 * c), (0, 2 * c + 2))
        else:
            syndrome((2 * d - 2, 2 * c + 1), (2 * d - 2, 2 * c), (2 * d - 2, 2 * c + 2))
    device, model = rx.PyGraph(), rx.PyGraph()
    device.add_nodes_from(range(n))
    device.add_edges_from_no_data(edges)
    model.add_nodes_from(range(len(coords)))
    model.add_edges_from_no_data(lattice)
    mapping = next(iter(rx.vf2_mapping(device, model)))
    return {q: coords[m] for q, m in mapping.items()}


def place_leftovers(pos, edges, n):
    """Put qubits without coordinates next to a placed neighbour"""
    adjacent = defaultdict(set)
    for a, b in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    taken = set(pos.values())
    for _ in range(n):
        missing = [q for q in range(n) if q not in pos]
        if not missing:
            break
        for q in missing:
            placed = [pos[p] for p in adjacent[q] if p in pos]
            if not placed:
                continue
            x, y = placed[0]
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                if (x + dx, y + dy) not in taken:
                    pos[q] = (x + dx, y + dy)
                    taken.add(pos[q])
                    break
    return pos


def bicycle_layout():
    """Positions of the 288 qubits of FakeIBMBicycle (X checks, data, Z checks) in the toric layout,
    found from the torus couplers so that each one joins neighbouring sites"""
    from eccentric.codes.gross_code import create_bivariate_bicycle_codes

    code, A, B = create_bivariate_bicycle_codes(12, 6, [3], [1, 2], [1, 2], [3])
    half = code.N // 2
    N = code.N
    a = [m.toarray() for m in A]
    b = [m.toarray() for m in B]
    width, height = 24, 12

    def nz(m, i):
        return int(np.nonzero(m[i])[0][0])

    def nz_col(m, j):
        return int(np.nonzero(m[:, j])[0][0])

    # X check i at (x, y): L data A2(i) at (x, y + 1), L data A3(i) at (x, y - 1),
    # R data B2(i) at (x + 1, y), R data B3(i) at (x - 1, y); X checks two apart share a data qubit
    xpos = {0: (0, 0)}
    queue = [0]
    while queue:
        i = queue.pop()
        x, y = xpos[i]
        steps = [(nz_col(a[2], nz(a[1], i)), (0, 2)), (nz_col(a[1], nz(a[2], i)), (0, -2)),
                 (nz_col(b[2], nz(b[1], i)), (2, 0)), (nz_col(b[1], nz(b[2], i)), (-2, 0))]
        for j, (dx, dy) in steps:
            if j not in xpos:
                xpos[j] = ((x + dx) % width, (y + dy) % height)
                queue.append(j)
    pos = {}
    for i, (x, y) in xpos.items():
        pos[i] = (x, y)
        pos[half + nz(a[1], i)] = (x, (y + 1) % height)
        pos[N + nz(b[1], i)] = ((x + 1) % width, y)
    # Z checks sit on the remaining sites; hz = [B^T | A^T], so Z check j meets its data qubits
    # through B^T (L data) and A^T (R data)
    free = {(x, y) for x in range(width) for y in range(height)} - set(pos.values())
    z0 = N + half
    for j in range(half):
        neighbours = [half + nz(b[1].T, j), half + nz(b[2].T, j), N + nz(a[1].T, j), N + nz(a[2].T, j)]
        sites = [pos[q] for q in neighbours]
        for site in free:
            if all(torus_distance(site, s, width, height) == 1 for s in sites):
                pos[z0 + j] = site
                free.discard(site)
                break
    c_couplers = {(i, half + nz(a[0], i)) for i in range(half)} | {(i, N + nz(b[0], i)) for i in range(half)}
    c_couplers |= {(z0 + j, half + nz(b[0].T, j)) for j in range(half)} | {(z0 + j, N + nz(a[0].T, j)) for j in range(half)}
    return pos, (width, height), {tuple(sorted(e)) for e in c_couplers}


def torus_distance(p, q, width, height):
    dx = abs(p[0] - q[0])
    dy = abs(p[1] - q[1])
    return min(dx, width - dx) + min(dy, height - dy)


def layout(name, size, backend):
    """Positions of the qubits, and the kind of each coupler"""
    n = backend.num_qubits
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    remote = {tuple(sorted(e)) for e in getattr(backend, "get_remote_gates", [])}
    moves = len(remote) > 4 * len(set(edges) - remote)
    kinds = {}
    qpus = getattr(backend, "num_qpus", None) or 1
    qpu_size = n // qpus
    pos = None
    wrap = None

    if name.startswith("real_bicycle"):
        one, (width, height), c_couplers = bicycle_layout()
        qpus = n // 288
        qpu_size = 288
        pos = {q + k * 288: (x + k * (width + 3), y) for k in range(qpus) for q, (x, y) in one.items()}
        for a, b in edges:
            ka, kb = a // 288, b // 288
            if ka != kb:
                kinds[(a, b)] = "link"
            elif (a % 288, b % 288) in c_couplers or (b % 288, a % 288) in c_couplers:
                kinds[(a, b)] = "c" if (a, b) not in remote else "c_remote"
            elif torus_distance(one[a % 288], one[b % 288], width, height) == 1 and \
                    abs(one[a % 288][0] - one[b % 288][0]) + abs(one[a % 288][1] - one[b % 288][1]) > 1:
                kinds[(a, b)] = "wrap"
            else:
                kinds[(a, b)] = "local"
        wrap = {"width": width, "height": height}
    elif name == "custom_heavyhex":
        pos = heavy_hex_code_positions(edges, n)
    elif name.startswith(("real_torino", "real_flamingo")):
        if name.startswith("real_flamingo"):
            qpu_size = 133
            qpus = n // qpu_size
            one_edges = [(a, b) for a, b in edges if a < qpu_size and b < qpu_size]
            one = place_leftovers(heavy_hex_positions(one_edges, 127), one_edges, qpu_size)
            width = max(x for x, _ in one.values()) + 1
            pos = {q + k * qpu_size: (x + k * (width + 3), y) for k in range(qpus) for q, (x, y) in one.items()}
        else:
            pos = place_leftovers(heavy_hex_positions(edges, n), edges, n)
    elif name.startswith("real_loon"):
        qpu_size = 121
        qpus = n // qpu_size
        pos = {q: ((q % qpu_size) % 11 + (q // qpu_size) * 14, (q % qpu_size) // 11) for q in range(n)}
    elif name == "real_willow":
        rows = [r for r, _ in backend.qubit_positions]
        pos = {q: (c, max(rows) - r) for q, (r, c) in enumerate(backend.qubit_positions)}
    elif name in ("real_apollo", "real_nsapollo", "real_infleqtion", "real_nsinfleqtion", "real_harvard"):
        pos = {q: (q % backend.columns, q // backend.columns) for q in range(n)}
    elif name == "custom_grid":
        side = int(round(n ** 0.5))
        pos = {q: (q % side, q // side) for q in range(n)}
    elif name == "custom_diamond":
        r = int((np.sqrt(2 * n - 1) - 1) / 2)
        nodes = [(x, y) for x in range(-r, r + 1) for y in range(-r, r + 1) if abs(x) + abs(y) <= r]
        pos = {q: (x + r, y + r) for q, (x, y) in enumerate(nodes)}
    elif name == "custom_cube":
        # The layers side by side; a coupler between layers joins the same site of neighbouring layers
        layers, rows, cols = 5, 6, 10
        pos = {}
        for q in range(n):
            layer, rest = divmod(q, rows * cols)
            row, col = divmod(rest, cols)
            pos[q] = (col + layer * (cols + 2), row)
    elif name == "custom_full":
        angle = 2 * np.pi * np.arange(n) / n
        pos = {q: (40 + 40 * np.cos(t), 40 + 40 * np.sin(t)) for q, t in zip(range(n), angle)}

    for e in edges:
        if e not in kinds:
            if e in remote:
                kinds[e] = "link"
            elif name == "custom_cube" and abs(e[0] - e[1]) >= 60:
                kinds[e] = "layer"
            elif name.startswith("real_loon") and abs(pos[e[0]][0] - pos[e[1]][0]) + abs(pos[e[0]][1] - pos[e[1]][1]) > 1.5:
                kinds[e] = "c"
            else:
                kinds[e] = "local"
    if name == "custom_full":
        kinds = {}
    if moves:
        # Shuttling devices: every pair not coupled on the grid is a move, too many to draw
        kinds = {e: k for e, k in kinds.items() if k != "link"}
    return pos, kinds, {"qpus": qpus, "qpu_size": qpu_size, "moves": moves, "all_to_all": name == "custom_full",
                        "torus": wrap}


def noise_params(error_type, backend):
    if error_type in ("sd6", "modsi1000"):
        model = get_noise_model(error_type, None, p=1.0, backend=backend)
        scale = "p"
    else:
        model = get_noise_model(error_type, None, backend=backend)
        scale = None
    params = []
    for attr, label in NOISE_PARAMS:
        value = getattr(model, attr, 0)
        if isinstance(value, (int, float)) and value:
            params.append({"label": label, "value": float(value), "scale": scale})
    for gate, value in sorted((model.noisy_gates or {}).items()):
        if value and not isinstance(value, dict):
            params.append({"label": f"Gate {gate}", "value": float(value), "scale": scale})
    times = []
    for key, label in GATE_TIMES:
        value = (model.gate_times or {}).get(key)
        if value:
            times.append({"label": label, "seconds": float(value)})
    return {"id": error_type, "label": NOISE_LABELS.get(error_type, error_type), "params": params, "times": times}


def coherence(backend):
    t1, t2 = [], []
    for props in getattr(backend.target, "qubit_properties", None) or []:
        t1.append(None if props is None or props.t1 is None else round(float(np.ravel(props.t1)[0]) * 1e6, 1))
        t2.append(None if props is None or props.t2 is None else round(float(np.ravel(props.t2)[0]) * 1e6, 1))
    if not any(v is not None for v in t1):
        return None, None
    return t1, t2


def main():
    out = sys.argv[1]
    np.random.seed(7)  # the IBM devices draw the T1 and T2 of their qubits at random
    devices = []
    for name, size, label, maker, noises, description in CATALOG:
        backend = get_backend(name, size)
        pos, kinds, info = layout(name, size, backend)
        # Qiskit's generic backends draw T1 and T2 at random; the synthetic noise models do not use them
        t1, t2 = coherence(backend) if not name.startswith("custom") else (None, None)
        n = backend.num_qubits
        degree = [0] * n
        for a, b in kinds:
            degree[a] += 1
            degree[b] += 1
        devices.append({
            "id": name, "label": label, "maker": maker, "description": description, "qubits": n,
            "sweep_label": DEVICES.get(name),
            "positions": [[round(float(pos[q][0]), 3), round(float(pos[q][1]), 3)] for q in range(n)],
            "edges": [[a, b, k] for (a, b), k in sorted(kinds.items())],
            "t1": t1, "t2": t2,
            "noise": [noise_params(e, backend) for e in noises],
            **info,
        })
        print(f"{name}: {n} qubits, {len(kinds)} couplers drawn, {info}")
    with open(out, "w") as f:
        json.dump({"generated": datetime.date.today().isoformat(), "devices": devices}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
