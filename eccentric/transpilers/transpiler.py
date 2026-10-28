import rustworkx as rx
from qiskit import transpile
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import SabreLayout
from .translators import qiskit_stim, qiskit_ibm_heron

# Search budget of the perfect layout search, as in Qiskit's optimization level 2
VF2_CALL_LIMIT = 5_000_000
VF2_CANDIDATES = 100


def perfect_layout(circuit, backend):
    """A placement of the circuit on the device under which every 2-qubit gate acts on coupled qubits,
    so routing adds no SWAPs (as Qiskit's VF2Layout finds), or None. It uses only local couplers if it
    can, else, of the placements found, one with the fewest gates on remote couplers."""
    pairs = {tuple(sorted(circuit.find_bit(q).index for q in inst.qubits))
             for inst in circuit.data if len(inst.qubits) == 2}
    interaction = rx.PyGraph()
    interaction.add_nodes_from(range(circuit.num_qubits))
    interaction.add_edges_from_no_data(sorted(pairs))
    coupling_map = backend.coupling_map
    edges = {tuple(sorted(e)) for e in coupling_map.get_edges()}
    remote = {tuple(sorted(p)) for p in getattr(backend, "get_remote_gates", [])}
    for allowed in (edges - remote, edges) if remote else (edges,):
        device = rx.PyGraph()
        device.add_nodes_from(range(coupling_map.size()))
        device.add_edges_from_no_data(sorted(allowed))
        best, best_cost = None, None
        mappings = rx.vf2_mapping(device, interaction, subgraph=True, induced=False, call_limit=VF2_CALL_LIMIT)
        for _, mapping in zip(range(VF2_CANDIDATES), mappings):
            # mapping: device qubit -> circuit qubit
            placement = {v: p for p, v in mapping.items()}
            cost = sum(tuple(sorted((placement[a], placement[b]))) in remote for a, b in pairs)
            if best is None or cost < best_cost:
                best, best_cost = placement, cost
            if cost == 0:
                break
        if best is not None:
            return Layout({circuit.qubits[v]: p for v, p in best.items()})
    return None


def local_layout(circuit, backend):
    """SabreLayout's placement of the circuit on the largest group of qubits joined by local couplers,
    with the device's local couplers to route on, or None if the device has no remote couplers or the
    circuit does not fit on the group"""
    coupling_map = backend.coupling_map
    remote = {tuple(sorted(p)) for p in getattr(backend, "get_remote_gates", [])}
    if not remote:
        return None
    local = [e for e in coupling_map.get_edges() if tuple(sorted(e)) not in remote]
    graph = rx.PyGraph()
    graph.add_nodes_from(range(coupling_map.size()))
    graph.add_edges_from_no_data(local)
    qubits = sorted(max(rx.connected_components(graph), key=len))
    if len(qubits) < circuit.num_qubits:
        return None
    index = {q: i for i, q in enumerate(qubits)}
    group = CouplingMap([[index[a], index[b]] for a, b in local if a in index and b in index])
    layout_pass = SabreLayout(group, skip_routing=True)
    layout_pass.run(circuit_to_dag(circuit))
    layout = layout_pass.property_set["layout"]
    return Layout({q: qubits[layout[q]] for q in circuit.qubits}), CouplingMap(local)


def run_transpiler(circuit, backend, layout_method, routing_method):
    # Without a layout method, Qiskit's level 0 places circuit qubit i on device qubit i. Place the
    # circuit without SWAPs when the device allows it, as levels 1 to 3 do, else let SabreLayout choose,
    # and routed on one QPU's local couplers if the circuit fits there
    initial_layout, coupling_map = None, None
    if layout_method is None:
        initial_layout = perfect_layout(circuit, backend)
        if initial_layout is None:
            initial_layout, coupling_map = local_layout(circuit, backend) or (None, None)
        if initial_layout is None:
            layout_method = "sabre"
    return transpile(
                circuit,
                basis_gates=qiskit_stim,
                optimization_level=0,
                backend=backend,
                coupling_map=coupling_map,
                initial_layout=initial_layout,
                layout_method=None if initial_layout else layout_method,
                routing_method=routing_method
        )
