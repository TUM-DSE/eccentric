import sys
import os
import stim

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

import yaml
import logging
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from qiskit.compiler import transpile
from qiskit_qec.utils import get_stim_circuits
from backends import get_backend, QubitTracking
from codes import get_code, get_max_d, get_min_n, make_idle_qubit_circuit
from noise import get_noise_model
from decoders import decode, raw_error_rate
from transpilers import run_transpiler, translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
import stim
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set Helvetica font
rcParams['font.family'] = 'Helvetica'

qiskit_stim = [
    "x", "y", "z", "cx", "cz", "cy", "h", "s", "s_dag",
    "swap", "reset", "measure", "barrier"
]

# -----------------------------
# 1. Load Steane code
# -----------------------------
code = get_code("surface", 3, 1)
detectors, logicals = code.stim_detectors()
qc = code.qc

# -----------------------------
# 2. Plotting function with gate counts and grid layout
# -----------------------------
def draw_two_qubit_graph_grid_14(circuit, filename):
    G = nx.MultiGraph()
    swap_edges = []
    cnots = 0
    swaps = 0

    # Extract edges and count
    for instr in circuit.data:
        if instr.operation.name in ["cx", "cy", "cz"]:
            q0 = instr.qubits[0]._index
            q1 = instr.qubits[1]._index
            G.add_edge(q0, q1)
            cnots += 1
        elif instr.operation.name == "swap":
            q0 = instr.qubits[0]._index
            q1 = instr.qubits[1]._index
            swap_edges.append((q0, q1))
            swaps += 1

    # Layout: 2 rows × 7 columns for 14 qubits
    all_nodes = sorted(set(list(G.nodes()) + [q for e in swap_edges for q in e]))
    pos = {}
    rows = 2
    cols = 7
    for i, node in enumerate(all_nodes):
        row = i // cols
        col = i % cols
        pos[node] = (col, -row)  # invert y so top row is row 0

    plt.figure(figsize=(10, 4))

    # Draw normal two-qubit gates (black) with a slight negative bend
    normal_edges = [e for e in G.edges() if e not in swap_edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        edge_color="black",
        width=2,
        connectionstyle="arc3,rad=-0.2"  # slight counter-clockwise bend
    )

    # Draw SWAP gates (red) with a slight positive bend
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=swap_edges,
        edge_color="red",
        width=2,
        connectionstyle="arc3,rad=0.2"   # slight clockwise bend
    )

    # Draw nodes (without labels if desired)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800)

    # Add counts underneath
    plt.text(0.5, 0.1,
             f"Number of CNOT gates: {cnots}\nNumber of SWAP gates: {swaps}",
             fontsize=20, fontname='Helvetica', ha='center', va='top', transform=plt.gca().transAxes)

    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------
# 3. Original Steane
# -----------------------------
draw_two_qubit_graph_grid_14(qc, "surface_connectivity.png")

# -----------------------------
# 4. Transpile to backends
# -----------------------------
backend_cube = get_backend("custom_cube", 36)
backend_full = get_backend("custom_full", 36)
backend_grid = get_backend("custom_grid", 36)

qc_grid = transpile(qc, backend=backend_grid, optimization_level=3, basis_gates=qiskit_stim)
qc_full = transpile(qc, backend=backend_full, optimization_level=3, basis_gates=qiskit_stim)
qc_cube = transpile(qc, backend=backend_cube, optimization_level=3, basis_gates=qiskit_stim)

# -----------------------------
# 5. Transpiled figures
# -----------------------------
draw_two_qubit_graph_grid_14(qc_grid, "surface_grid_swaps.png")
draw_two_qubit_graph_grid_14(qc_full, "surface_full_swaps.png")
draw_two_qubit_graph_grid_14(qc_cube, "surface_cube_swaps.png")