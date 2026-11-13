"""Lattice-surgery operations on rotated surface-code patches, built with tqec (https://github.com/tqec/tqec).

Each operation is a tqec block graph compiled into a noiseless Stim circuit with tqec's own detectors and
observables, which then goes through ECCentric's pipeline like any memory circuit: compiled onto a device,
given a noise model, decoded. A patch of distance d is tqec's scale k = (d - 1) / 2, and every block of the
graph lasts d syndrome rounds (tqec's blocks scale with the patch, so the rounds per block are fixed).

- ls_memory: one patch, prepared and read out in the basis (tqec's memory; a check against the surface code).
- ls_merge: two patches merged for one block and split again: in the Z basis a ZZ measurement (patches
  joined along their Z boundaries), in the X basis an XX measurement. The observable is the joint logical
  operator of both patches, which the merge must preserve.
- ls_cnot: the lattice-surgery CNOT of tqec's gallery (control, target and an ancilla patch); its observables
  are the logical Z (Z basis) or X (X basis) correlations a CNOT maps. A shot fails if any is wrong.
"""
from functools import lru_cache

import stim

# Time layers of blocks in each operation's graph (its syndrome rounds are this many times d)
LAYERS = {"ls_memory": 1, "ls_merge": 3, "ls_cnot": 4}


def _merge_graph(basis: str):
    from tqec.computation.block_graph import BlockGraph
    from tqec.utils.position import Position3D

    # A Z-basis patch (ZXZ) has its Z boundaries facing x, an X-basis one (ZXX) its X boundaries facing y; joining
    # two patches along those boundaries for one block measures ZZ or XX
    kind, other = ("ZXZ", (1, 0)) if basis == "Z" else ("ZXX", (0, 1))
    g = BlockGraph(f"{basis}{basis} merge")
    for x, y in ((0, 0), other):
        for z in range(3):
            g.add_cube(Position3D(x, y, z), kind)
        for z in range(2):
            g.add_pipe(Position3D(x, y, z), Position3D(x, y, z + 1))
    g.add_pipe(Position3D(0, 0, 1), Position3D(other[0], other[1], 1))
    return g


def _graph(operation: str, basis: str):
    from tqec import gallery
    from tqec.utils.enums import Basis

    b = Basis.Z if basis == "Z" else Basis.X
    if operation == "ls_memory":
        return gallery.memory(b)
    if operation == "ls_merge":
        return _merge_graph(basis)
    if operation == "ls_cnot":
        return gallery.cnot(b)
    raise ValueError(f"unknown lattice-surgery operation {operation}")


@lru_cache(maxsize=8)
def _circuit(operation: str, d: int, basis: str) -> stim.Circuit:
    from tqec import compile_block_graph

    # Without tqec's detector database: it is one pickle file in the home directory, which parallel workers
    # would write at once
    circuit = compile_block_graph(_graph(operation, basis)).generate_stim_circuit((d - 1) // 2, do_not_use_database=True)
    return _without_unused_qubits(circuit)


def _without_unused_qubits(circuit: stim.Circuit) -> stim.Circuit:
    """tqec declares coordinates for qubits of its templates that no gate of the operation uses (12 of the
    merge's 53 at d = 3); leave them out so that the circuit asks a device only for the qubits it needs"""
    used = {t.value for inst in circuit.flattened() if inst.name != "QUBIT_COORDS"
            for t in inst.targets_copy() if t.is_qubit_target}
    result = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name == "QUBIT_COORDS":
            targets = [t for t in inst.targets_copy() if t.value in used]
            if targets:
                result.append("QUBIT_COORDS", targets, inst.gate_args_copy())
        else:
            result.append(inst)
    return result


def make(operation: str):
    def build(d: int, rounds: int, schedule, basis: str) -> stim.Circuit:
        if d < 3 or d % 2 == 0:
            raise ValueError(f"lattice-surgery patches have odd distance d >= 3, got {d}")
        if rounds is not None and rounds != d:
            raise ValueError(f"{operation} runs d = {d} syndrome rounds per block, not {rounds} (leave cycles unset)")
        return _circuit(operation, d, basis).copy()
    return build


# Physical qubits the operations use (data and syndrome qubits), exact for every d: a patch is the rotated
# surface code's 2d^2 - 1, a merge adds a seam of 2d + 1
NUM_QUBITS = {
    "ls_memory": lambda d: 2 * d * d - 1,
    "ls_merge": lambda d: 4 * d * d + 2 * d - 1,
    "ls_cnot": lambda d: 6 * d * d + 4 * d - 1,
}
