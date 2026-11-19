"""Fault-tolerant operations on the CSS codes of the catalog, built with the qLDPC library
(https://github.com/qLDPCOrg/qLDPC, Apache-2.0; its surgery subpackage is marked experimental by its authors).

An operation is chosen with the `operation` setting (`operations` in a config); None is the code's own memory
circuit. Every operation is a noiseless Stim circuit with detectors and observables that then goes through the
pipeline like a memory circuit (device, routing, noise model, decoders); a shot fails if any observable is
wrong. All of them use qLDPC's syndrome extraction (edge coloring of the Tanner graph, one ancilla per check),
so `memory_ec` is the baseline the others are compared with; the catalog's own memory circuits may use another
schedule (IBM's for bivariate bicycle codes, QUITS' for product codes, Stim's for the surface code).

- memory_ec: the memory experiment with qLDPC's syndrome extraction, `rounds` rounds.
- measure_z, measure_x: fault-tolerant measurement of the first logical qubit's Z (X) by code surgery: a gadget
  of ancilla qubits is attached to the logical operator's support for `rounds` rounds (Webster, Smith, Cohen,
  arXiv:2511.15989, the gadget; Cain et al., arXiv:2603.28627, the circuit), then detached. The data is
  prepared in the measured basis, so the measured value is known; the observable is that value (the parity of
  the gadget's checks). Gadgets whose boundary Cheeger constant is below 1 can lose distance (Williamson and
  Yoder, Nature Physics 2026); GADGET_BOOST asks qLDPC to augment them until it is at least 1, as far as it
  can: when the constant cannot be computed (logical operators of weight above 26) or the target is out of
  reach, the plain gadget is used and a warning logged.
- joint_zz, joint_xx: joint measurement of the first logical qubits of two blocks of the code (ZZ or XX), the
  two gadgets joined by a universal adapter (Swaroop, Jochym-O'Connor, Yoder, arXiv:2410.03628).
- transversal_cnot: two blocks of the code, `rounds` rounds each, one layer of CX gates between corresponding
  data qubits (a logical CNOT between every pair of logical qubits, control in the first block), `rounds` more
  rounds, readout in the memory basis. Errors cross the gate, so the detectors of the round after it also
  include the partner block's previous round, and decoders see hyperedges between the blocks.

measure_z and joint_zz run in the Z basis, measure_x and joint_xx in the X basis; the others take the memory
basis. Operations need a fixed distance; `rounds` defaults to the distance. The circuit-level distance is that
of qLDPC's syndrome extraction, which can be below the code's (the Steane code: 2, from two-qubit gate faults
in its Z-check layers; the surface code keeps its distance). The codes available are the catalog's
CSS codes (bivariate, coprime and trivariate bicycle, balanced product, hypergraph and lifted product,
lift-connected surface, quantum Tanner, many-hypercube, Golay, Reed-Muller, toric) and the surface code, the
gross code and the Steane code at d = 3; subsystem codes (Bacon-Shor, heavy-hex), the color code, XZZX and the
Floquet codes are not CSS stabilizer codes qLDPC can act on.
"""
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import stim

OPERATIONS = ("memory_ec", "measure_z", "measure_x", "joint_zz", "joint_xx", "transversal_cnot")
BASIS_OF = {"measure_z": "Z", "joint_zz": "Z", "measure_x": "X", "joint_xx": "X"}
# Augment surgery gadgets until their boundary Cheeger constant reaches this (None: no augmentation)
GADGET_BOOST = 1.0
MEASUREMENTS = ("M", "MX", "MR", "MRX")


def css_code(code_name: str, d: Optional[int]):
    """The qLDPC CSSCode of a catalog code (or the surface, gross or Steane code), ValueError if it has none"""
    from qldpc import codes

    from .catalog import CODES, FixedCode

    if code_name == "surface":
        return codes.SurfaceCode(d, rotated=True)
    if code_name == "gross":
        from .quits_codes import css_code as quits_css
        return quits_css("make_bb_circuit", 144)
    if code_name == "steane":
        if d != 3:
            raise ValueError("operations on the concatenated Steane code exist for d = 3 only")
        return codes.SteaneCode()
    spec = CODES.get(code_name)
    if spec is None or not hasattr(spec.make, "code"):
        raise ValueError(f"{code_name} is not a CSS code the operations can be built on")
    if isinstance(spec, FixedCode) and d is not None and d != spec.distance:
        raise ValueError(f"{code_name} only has distance {spec.distance}, got {d}")
    return spec.make.code(d)


def operation_rounds(operation: str, rounds: int) -> int:
    """Syndrome rounds the operation's circuit runs for a setting of `rounds` (the merge rounds of a
    measurement, the rounds before and after the gate of a CNOT)"""
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation {operation}; choose from {OPERATIONS}")
    return 2 * rounds if operation == "transversal_cnot" else rounds


def _edge_coloring():
    from qldpc.circuits import EdgeColoring
    return EdgeColoring()


def _gadget(code, basis):
    from qldpc.experimental import surgery
    from qldpc.objects import Pauli

    pauli = Pauli.Z if basis == "Z" else Pauli.X
    gadget = surgery.build_gadget(code, code.get_logical_ops(pauli)[0], basis=pauli)
    if GADGET_BOOST is None:
        return gadget
    try:
        if surgery.cheeger_constant(gadget) < GADGET_BOOST:
            gadget = surgery.boost_gadget(gadget, method="combinatorial", target=GADGET_BOOST, seed=0)
    except (ValueError, RuntimeError) as e:
        logging.warning("surgery gadget (%s, %d qubits) not boosted to Cheeger constant %s: %s",
                        basis, len(gadget.support), GADGET_BOOST, str(e)[:120])
    return gadget


def _init(code, basis):
    from qldpc.experimental import surgery
    return surgery.logical_state_init(code, "0" if basis == "Z" else "+", log_idx=0)


def _measurement(code, rounds, basis):
    from qldpc.experimental import surgery
    circuit = surgery.build_single_ppm_circuit(_gadget(code, basis), rounds=rounds, data_init=_init(code, basis))
    return surgery.keep_only_observable(circuit, keep_idx=0)


def _joint_measurement(code, rounds, basis):
    from qldpc import codes
    from qldpc.experimental import surgery

    # A distinct code object for the second block: gadgets over one object share its data qubits
    other = codes.CSSCode(code.matrix_x, code.matrix_z)
    left, right = _gadget(code, basis), _gadget(other, basis)
    bridge = surgery.build_bridge(left, right)
    circuit, _ = surgery.build_joint_ppm_circuit(left, right, bridge, rounds=rounds,
                                                data_init=(_init(code, basis), _init(other, basis)))
    return surgery.keep_only_observable(circuit, keep_idx=0)


def _memory(code, rounds, basis):
    from qldpc.circuits import get_memory_experiment
    from qldpc.objects import Pauli

    return get_memory_experiment(code, basis=Pauli[basis], num_rounds=rounds, syndrome_measurement_strategy=_edge_coloring())


def _transversal_cnot(code, rounds, basis):
    from qldpc import codes
    from qldpc.circuits.memory import get_memory_experiment_parts
    from qldpc.objects import Pauli

    hx, hz = np.asarray(code.matrix_x, dtype=int), np.asarray(code.matrix_z, dtype=int)
    n = hx.shape[1]
    pair = codes.CSSCode(_block_diag(hx), _block_diag(hz))
    parts = get_memory_experiment_parts(pair, Pauli[basis], num_rounds=1, syndrome_measurement_strategy=_edge_coloring())
    ids = parts.qubit_ids
    # One round of syndrome extraction without the library's detectors (its detectors assume no gate between
    # rounds); the checks are measured once per round, in the order of this instruction
    cycle = stim.Circuit()
    order = []
    for inst in parts.qec_cycle.flattened():
        if inst.name == "DETECTOR":
            continue
        cycle.append(inst)
        if inst.name in MEASUREMENTS:
            order += [t.value for t in inst.targets_copy()]
    if sorted(order) != sorted(ids.check):
        raise ValueError("expected each check qubit measured once per round")
    position = {q: i for i, q in enumerate(order)}
    mx, mz = hx.shape[0], hz.shape[0]
    # The combined code's checks are the first block's then the second's; CX (control first block, target
    # second) maps Z checks of the target onto both blocks' and X checks of the control likewise
    partner = {}
    for checks, m in ((ids.checks_x, mx), (ids.checks_z, mz)):
        for i in range(m):
            partner[checks[i]], partner[checks[i + m]] = checks[i + m], checks[i]
    crossed = set(ids.checks_z[mz:]) | set(ids.checks_x[:mx])
    first_round = ids.checks_z if basis == "Z" else ids.checks_x

    circuit = parts.initialization.copy()
    total, index = 0, {}  # measurements so far; (check, round) -> absolute measurement index

    def rec(q, r):
        return stim.target_rec(index[q, r] - total)

    for r in range(2 * rounds):
        if r == rounds:
            circuit.append("TICK")
            circuit.append("CX", [q for i in range(n) for q in (ids.data[i], ids.data[n + i])])
            circuit.append("TICK")
        circuit += cycle
        for q in order:
            index[q, r] = total + position[q]
        total += len(order)
        for q in order:
            if r == 0:
                if q in first_round:
                    circuit.append("DETECTOR", [rec(q, 0)])
                continue
            targets = [rec(q, r), rec(q, r - 1)]
            if r == rounds and q in crossed:
                targets.append(rec(partner[q], r - 1))
            circuit.append("DETECTOR", targets)
    circuit += parts.readout
    return circuit


def _block_diag(h):
    m, n = h.shape
    out = np.zeros((2 * m, 2 * n), dtype=int)
    out[:m, :n], out[m:, n:] = h, h
    return out


def make_operation_circuit(code_name: str, d: int, rounds: Optional[int], basis: str, operation: str) -> stim.Circuit:
    """The noiseless circuit of an operation on a code at distance d (ValueError if it cannot be built)"""
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation {operation}; choose from {OPERATIONS}")
    if operation in BASIS_OF and basis != BASIS_OF[operation]:
        raise ValueError(f"{operation} runs in the {BASIS_OF[operation]} basis, not {basis}")
    code = css_code(code_name, d)
    rounds = rounds or d
    if operation == "memory_ec":
        return _memory(code, rounds, basis)
    if operation.startswith("measure"):
        return _measurement(code, rounds, basis)
    if operation.startswith("joint"):
        return _joint_measurement(code, rounds, basis)
    return _transversal_cnot(code, rounds, basis)


@lru_cache(maxsize=None)
def operation_num_qubits(code_name: str, d: int, operation: str) -> int:
    from .catalog import _flatten_and_compact
    return _flatten_and_compact(make_operation_circuit(code_name, d, 2, BASIS_OF.get(operation, "Z"), operation)).num_qubits
