import math
import os

import stim

# Circuits from https://github.com/oscarhiggott/hyperbolic-floquet-data (arXiv:2308.03750),
# see data/hyperbolic_floquet/README.md

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hyperbolic_floquet")


def make_hyperbolic_floquet_circuit(filename: str, rounds: int) -> stim.Circuit:
    circuit = stim.Circuit.from_file(os.path.join(DATA_DIR, filename))
    # 4 + 2 sub-rounds around a REPEAT block of 6 sub-rounds (two X, Y, Z rounds per iteration)
    repeats = max(1, math.ceil((rounds - 2) / 2))
    result = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(repeats, inst.body_copy()))
        else:
            result.append(inst)
    return result
