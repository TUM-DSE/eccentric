import stim

# XZZX surface code (Bonilla Ataides et al., arXiv:2009.07851): the rotated surface code with a
# Hadamard on every other data qubit, so every stabilizer becomes XZZX
_HADAMARD_CONJUGATED = {
    "R": "RX", "RX": "R",
    "M": "MX", "MX": "M",
    "MR": "MRX", "MRX": "MR",
}


def _hadamard_conjugate(circuit: stim.Circuit, qubits: set) -> stim.Circuit:
    result = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(inst.repeat_count, _hadamard_conjugate(inst.body_copy(), qubits)))
            continue
        name, targets, args = inst.name, inst.targets_copy(), inst.gate_args_copy()
        if name in _HADAMARD_CONJUGATED:
            # One target at a time to keep the measurement record order
            for t in targets:
                result.append(_HADAMARD_CONJUGATED[name] if t.value in qubits else name, [t], args)
        elif name == "CX":
            for c, t in zip(targets[::2], targets[1::2]):
                if c.value in qubits and t.value in qubits:
                    result.append("CX", [t, c])
                elif t.value in qubits:
                    result.append("CZ", [c, t])
                elif c.value in qubits:
                    result.append("XCX", [c, t])
                else:
                    result.append("CX", [c, t])
        elif name in ("H", "TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS") or not any(
            t.is_qubit_target and t.value in qubits for t in targets
        ):
            result.append(inst)
        else:
            raise NotImplementedError(f"Cannot Hadamard-conjugate {name}")
    return result


def make_xzzx_circuit(d, T, basis="Z") -> stim.Circuit:
    css = stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", rounds=T, distance=d)
    coords = css.get_final_qubit_coordinates()
    # Data qubits sit at odd (x, y); one diagonal of every plaquette gets the Hadamard
    hadamard = {
        q for q, (x, y) in coords.items()
        if x % 2 == 1 and y % 2 == 1 and ((x - 1) // 2 + (y - 1) // 2) % 2 == 1
    }
    return _hadamard_conjugate(css, hadamard)
