import math
from .gross_code import get_gross_code
from .color_code_stim import get_color_code
from .bacon_shor import get_bacon_shor_code
from .concat_steane import get_concat_steane_code
from .surface_code import get_surface_code
from .hh_code import get_hh_code
from .catalog import CODES, MemoryCodeCircuit, make_circuit, max_distance, min_num_qubits, num_rounds, resolve_schedule

def get_code(code_name: str, d: int, cycles: int, schedule: str = None, basis: str = "Z"):
    """The noiseless memory circuit of a code, preparing and measuring the logical state in `basis` (Z or X)"""
    resolve_schedule(code_name, schedule)
    if basis not in ("Z", "X"):
        raise ValueError(f"basis must be Z or X, got {basis}")
    if code_name in CODES:
        return MemoryCodeCircuit(stim_circuit=make_circuit(code_name, d, cycles, schedule, basis))
    if cycles is None:
        # The builders' own defaults: the gross code's 12 rounds, Bacon-Shor's single round
        cycles = {"gross": 12, "bacon": 1}.get(code_name, d)
    if code_name == "hh":
        return get_hh_code(d, cycles, basis)
    elif code_name == "gross":
        return get_gross_code(T=cycles, basis=basis)
    elif code_name == "surface":
        return get_surface_code(d=d, T=cycles, basis=basis)
    elif code_name == "color":
        return get_color_code(d, rounds=cycles, basis=basis)
    elif code_name == "bacon":
        return get_bacon_shor_code(d, cycles, basis)
    elif code_name == 'steane':
        m = {3: 1, 9: 2, 27: 3}.get(d)
        if m is None:
            raise ValueError("Steane code only supports m = 1, 2, 3")
        return get_concat_steane_code(m, cycles, basis=basis)


def get_max_d(code_name: str, n: int):
    if code_name in CODES:
        return max_distance(code_name, n)
    elif code_name == "surface":
        if n < 26:
            raise ValueError(f"Surface code needs at least 26 qubits, got {n}")

        d = int((-3 + math.isqrt(9 + 8*(n+1))) // 4)
        d = d - ((1 - d) % 2)
        assert d % 2 == 1
        return d
    elif code_name == "hh":
        d = int((2 + math.sqrt(40 * n + 24)) / 10)
        d = d - ((1 - d) % 2) # https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.011022
        assert d % 2 == 1
        return d
    elif code_name == "gross":
        return 12
    elif code_name == "color":
        d = int((math.sqrt(4*n) +1)/3)
        d = d - ((1 - d) % 2)
        assert d % 2 == 1
        return d
    elif code_name == "bacon":
        d = int(math.sqrt(n))
        d = d - ((1 - d) % 2)
        return d
    elif code_name == 'steane':
        if n >= 685:
            return 27
        elif n >= 97:
            return 9
        elif n >= 13:
            return 3
    return 0


def get_min_n(code_name: str, d: int):
    if code_name in CODES:
        return min_num_qubits(code_name, d)
    elif code_name == "surface":
        return 2 * d * d + 3 * d - 1
    elif code_name == "hh":
        return int(math.ceil((5 * d * d - 2 * d - 1) / 2))
    elif code_name == "gross":
        return 288
    elif code_name == "color":
        return int(math.ceil((3*d - 1)**2 / 4))
    elif code_name == "bacon":
        return d * d
    elif code_name == 'steane':
        if d == 3:
            return 13
        elif d == 9:
            return 97
        elif d == 27:
            return 685
    return 0
