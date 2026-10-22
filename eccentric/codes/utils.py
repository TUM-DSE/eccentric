import math
from .gross_code import get_gross_code
from .color_code_stim import get_color_code
from .bacon_shor import get_bacon_shor_code
from .concat_steane import get_concat_steane_code
from .surface_code import get_surface_code
from .hh_code import get_hh_code
from .catalog import CODES, MemoryCodeCircuit, make_circuit, max_distance, min_num_qubits, num_rounds, resolve_schedule

def get_code(code_name: str, d: int, cycles: int, schedule: str = None):
    resolve_schedule(code_name, schedule)
    if code_name in CODES:
        return MemoryCodeCircuit(stim_circuit=make_circuit(code_name, d, cycles, schedule))
    elif code_name == "hh":
        if cycles == None:
            return get_hh_code(d, None)
        else:
             return get_hh_code(d, cycles)
    elif code_name == "gross":
        if cycles == None:
            return get_gross_code()
        else:
            return get_gross_code(T=cycles)
    elif code_name == "surface":
        if cycles == None:
            code = get_surface_code(d=d, T=d)
        else:
            code = get_surface_code(d=d, T=cycles)
        return code
    elif code_name == "color":
        if cycles == None:
            return get_color_code(d, rounds=d)
        else:
            return get_color_code(d, rounds=cycles)
    elif code_name == "bacon":
        if cycles == None:
            return get_bacon_shor_code(d)
        else:
            return get_bacon_shor_code(d, cycles)
    elif code_name == 'steane':
        if d == 3:
            m = 1
        elif d == 9:
            m = 2
        elif d == 27:
            m = 3
        else:
            raise ValueError("Steane code only supports m = 1, 2, 3")
        if cycles == None:
            cycles = d
        return get_concat_steane_code(m,cycles, basis="Z")


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
        if n >= 686:
            return 27
        elif n >= 98:
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
            return 14
        elif d == 9:
            return 98
        elif d == 27:
            return 686
    return 0
