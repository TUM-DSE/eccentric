"""Whether a circuit's detectors split into two types, X-type (detecting Z errors on the data qubits) and Z-type
(detecting X errors), as CSS memory circuits' do, and how the error mechanisms fall on the two sides. GARI and
GARI++ need this structure (the port follows ECCentric's GARI classification, eccentric/decoders/gari/gari.py),
and no error-model feature shows it.

Detectors are typed by stim's detecting regions on the data qubits (the qubit identities measured exactly once,
followed through SWAPs), read in each data qubit's frame: compiled circuits keep data qubits in a Hadamard frame
for parts of a round. H toggles the frame, R/MR clear it, RX/MRX set it, Pauli gates leave it; any other
single-qubit Clifford (S, SQRT_X, ...) makes it unknown until the next reset, so the detectors it touches count as
detecting both types (Floquet circuits). Circuits with other gates get NaN.
"""
import math

import numpy as np
import stim

ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK", "MPAD"}
PAULIS = {"I", "X", "Y", "Z"}
TWO_QUBIT = {"CX", "CNOT", "CZ", "ZCX", "ZCZ"}
FEATURES = ["mixed_detectors", "y_mechanisms", "y_parts_unmatched", "observables_both_sides"]


class Unsupported(ValueError):
    pass


def _frames(noiseless):
    """Per TICK: the qubits holding a data qubit, those in a Hadamard frame, and those in an unknown frame"""
    holder = list(range(noiseless.num_qubits))
    measured = np.zeros(noiseless.num_qubits, dtype=int)
    positions, hadamard_at, unknown_at = [], [], []
    hadamard, unknown = set(), set()
    for inst in noiseless:
        name = inst.name
        if name == "TICK":
            positions.append(list(holder))
            hadamard_at.append(frozenset(hadamard))
            unknown_at.append(frozenset(unknown))
            continue
        if name in ANNOTATIONS or name.endswith("_ERROR") or name.startswith(("DEPOLARIZE", "PAULI_CHANNEL")):
            continue
        targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "SWAP":
            for a, b in zip(targets[::2], targets[1::2]):
                holder[a], holder[b] = holder[b], holder[a]
                for s in (hadamard, unknown):
                    ina, inb = a in s, b in s
                    s.discard(a), s.discard(b)
                    if ina:
                        s.add(b)
                    if inb:
                        s.add(a)
        elif name == "H":
            hadamard ^= set(targets)
        elif name in ("R", "RZ"):
            hadamard -= set(targets)
            unknown -= set(targets)
        elif name == "RX":
            hadamard |= set(targets)
            unknown -= set(targets)
        elif name in ("M", "MZ", "MX"):
            for q in targets:
                measured[holder[q]] += 1
        elif name in ("MR", "MRZ", "MRX"):
            for q in targets:
                measured[holder[q]] += 1
            (hadamard.update if name == "MRX" else hadamard.difference_update)(targets)
            unknown -= set(targets)
        elif name in PAULIS or name in TWO_QUBIT:
            pass
        elif stim.gate_data(name).is_single_qubit_gate and stim.gate_data(name).is_unitary:
            unknown |= set(targets)
        else:
            raise Unsupported(name)
    data = {i for i in range(noiseless.num_qubits) if measured[i] == 1}
    data_at = [frozenset(q for q, ident in enumerate(h) if ident in data) for h in positions]
    return data_at, hadamard_at, unknown_at


def detector_votes(circuit: stim.Circuit):
    """Per detector: how many (tick, data qubit) places flip it with an X error and with a Z error"""
    noiseless = circuit.flattened().without_noise()
    data_at, hadamard_at, unknown_at = _frames(noiseless)
    n = noiseless.num_detectors
    by_x, by_z = np.zeros(n), np.zeros(n)
    for target, ticks in noiseless.detecting_regions().items():
        if not target.is_relative_detector_id():
            continue
        k = target.val
        for tick, pauli in ticks.items():
            if tick >= len(data_at):
                continue
            for q in data_at[tick]:
                if q >= len(pauli) or not pauli[q]:
                    continue
                p = pauli[q]  # 1 = X, 2 = Y, 3 = Z
                if q in unknown_at[tick]:
                    by_x[k] += 1
                    by_z[k] += 1
                    continue
                if q in hadamard_at[tick]:
                    p = {1: 3, 3: 1}.get(p, p)
                by_x[k] += p in (2, 3)
                by_z[k] += p in (1, 2)
    return by_x, by_z


def detector_type_features(circuit: stim.Circuit, dem: stim.DetectorErrorModel | None = None) -> dict:
    """mixed_detectors: share of detectors detecting both X and Z errors on the data qubits; y_mechanisms: share
    of error mechanisms flipping detectors of both types; y_parts_unmatched: share of those whose part on one
    side flips a set of detectors no single-type mechanism flips; observables_both_sides: 1 if mechanisms of
    both single types flip an observable. NaN for circuits with gates the classification does not follow"""
    try:
        by_x, by_z = detector_votes(circuit)
    except Unsupported:
        return {name: math.nan for name in FEATURES}
    n = len(by_x)
    if n == 0:
        return {name: math.nan for name in FEATURES}
    z_type = by_x > by_z  # detects X errors
    dem = dem or circuit.detector_error_model(approximate_disjoint_errors=True)
    singles = {True: set(), False: set()}  # side (Z-type?) -> detector sets of single-type mechanisms
    ys, obs_side, single_count = [], {True: False, False: False}, 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        targets = inst.targets_copy()
        dets = [t.val for t in targets if t.is_relative_detector_id()]
        if not dets:
            continue
        on_z = tuple(sorted(d for d in dets if z_type[d]))
        on_x = tuple(sorted(d for d in dets if not z_type[d]))
        if on_z and on_x:
            ys.append((on_z, on_x))
        else:
            side = bool(on_z)
            singles[side].add(on_z or on_x)
            single_count += 1
            if any(t.is_logical_observable_id() for t in targets):
                obs_side[side] = True
    mechanisms = len(ys) + single_count
    unmatched = sum(1 for on_z, on_x in ys if on_z not in singles[True] or on_x not in singles[False])
    return {"mixed_detectors": float(((by_x > 0) & (by_z > 0)).mean()),
            "y_mechanisms": len(ys) / mechanisms if mechanisms else math.nan,
            "y_parts_unmatched": unmatched / len(ys) if ys else 0.0,
            "observables_both_sides": int(obs_side[True] and obs_side[False])}
