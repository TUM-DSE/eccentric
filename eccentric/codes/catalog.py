"""Codes that are built as noiseless Stim memory circuits and converted with StimCodeCircuit.

A ``CodeFamily`` is parametrized by its distance, a ``FixedCode`` is a single [[n, k, d]] instance.
``make`` returns the circuit for a number of syndrome rounds (and a distance, for families) and a
syndrome-extraction schedule out of ``schedules``, the first of which is the default. Codes with
no ``schedules`` have a single, fixed circuit.
"""
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import stim
from qiskit_qec.circuits import StimCodeCircuit


QLDPC_SCHEDULES = ("edge_coloring", "edge_coloring_xz")
PRODUCT_SCHEDULES = ("cardinal", "cardinalNSmerge", "zxcoloration")
# zxcoloration keeps more circuit-level distance than cardinal for the LCS codes
LCS_SCHEDULES = ("zxcoloration", "cardinal", "cardinalNSmerge")
MITTEN_SCHEDULES = ("custom", "hook_error_free", "zxcoloration")
# "ibm" is the depth-8 circuit of Bravyi et al. (arXiv:2308.07915) also used for the gross code
BB_SCHEDULES = ("ibm", "zxcoloration")


def _same_rounds(rounds):
    return rounds


def _planar_honeycomb_rounds(rounds):
    return max(4, rounds + rounds % 2)


def _hyperbolic_rounds(rounds):
    return 2 + 2 * max(1, math.ceil((rounds - 2) / 2))


@dataclass(frozen=True)
class CodeFamily:
    make: Callable[[int, int, Optional[str]], stim.Circuit]
    num_qubits: Callable[[int], int]
    min_d: int = 3
    step: int = 2
    schedules: tuple = ()
    # Syndrome rounds the circuit actually runs when asked for a number of rounds
    rounds: Callable[[int], int] = _same_rounds


@dataclass(frozen=True)
class FixedCode:
    make: Callable[[int, Optional[str]], stim.Circuit]
    distance: int
    schedules: tuple = ()
    rounds: Callable[[int], int] = _same_rounds


def _repetition(d, rounds, schedule):
    return stim.Circuit.generated("repetition_code:memory", rounds=rounds, distance=d)


def _xzzx(d, rounds, schedule):
    from .xzzx_code import make_xzzx_circuit
    return make_xzzx_circuit(d, rounds)


def _toric(d, rounds, schedule):
    from .qldpc_codes import make_memory_circuit, toric_code
    return make_memory_circuit(toric_code(d), rounds, schedule)


def _qldpc(builder_name, *args):
    def make(rounds, schedule):
        from . import qldpc_codes
        return qldpc_codes.make_memory_circuit(getattr(qldpc_codes, builder_name)(*args), rounds, schedule)
    return make


def _bb(n):
    def make(rounds, schedule):
        if schedule == "ibm":
            from .gross_code import make_bb_circuit
            return make_bb_circuit(n, rounds)
        from .quits_codes import make_bb_circuit
        return make_bb_circuit(n, rounds, schedule)
    return make


def _quits(builder_name, *args):
    def make(rounds, schedule):
        from . import quits_codes
        return getattr(quits_codes, builder_name)(*args, rounds, schedule)
    return make


def _hyperbolic(filename):
    def make(rounds, schedule):
        from .hyperbolic_floquet import make_hyperbolic_floquet_circuit
        return make_hyperbolic_floquet_circuit(filename, rounds)
    return make


def _chromobius_color(d, rounds, schedule):
    # Triangular color code measured with MPPs, with the color-annotated detectors Chromobius needs
    from .chromobius_color_code import make_color_code_circuit
    return make_color_code_circuit(obs_basis="Z", base_data_width=d, rounds=rounds, noise_strength=0)


def _periodic_honeycomb(d, rounds, schedule):
    from .honeycomb import make_periodic_honeycomb_circuit
    return make_periodic_honeycomb_circuit(d, rounds)


def _planar_honeycomb(d, rounds, schedule):
    from .honeycomb import make_planar_honeycomb_circuit
    return make_planar_honeycomb_circuit(d, rounds)


def _periodic_honeycomb_qubits(d):
    from .honeycomb import periodic_honeycomb_num_qubits
    return periodic_honeycomb_num_qubits(d)


def _planar_honeycomb_qubits(d):
    from .honeycomb import planar_honeycomb_num_qubits
    return planar_honeycomb_num_qubits(d)


CODES = {
    "repetition": CodeFamily(_repetition, lambda d: 2 * d - 1),
    "xzzx": CodeFamily(_xzzx, lambda d: 2 * d * d - 1),
    "toric": CodeFamily(_toric, lambda d: 4 * d * d, schedules=QLDPC_SCHEDULES),
    "chromobius_color": CodeFamily(_chromobius_color, lambda d: (3 * d * d + 1) // 4),
    "honeycomb": CodeFamily(_periodic_honeycomb, _periodic_honeycomb_qubits, min_d=4),
    "planar_honeycomb": CodeFamily(_planar_honeycomb, _planar_honeycomb_qubits, rounds=_planar_honeycomb_rounds),
    "bb54": FixedCode(_bb(54), 6, BB_SCHEDULES),
    "bb72": FixedCode(_bb(72), 6, BB_SCHEDULES),
    "bb90": FixedCode(_bb(90), 10, BB_SCHEDULES),
    "bb98": FixedCode(_bb(98), 12, BB_SCHEDULES),
    "bb108": FixedCode(_bb(108), 10, BB_SCHEDULES),
    "bb162": FixedCode(_bb(162), 14, BB_SCHEDULES),
    "bb180": FixedCode(_bb(180), 16, BB_SCHEDULES),
    "bb288": FixedCode(_bb(288), 18, BB_SCHEDULES),
    "bb756": FixedCode(_bb(756), 34, BB_SCHEDULES),
    "hgp_hamming": FixedCode(_qldpc("hgp_hamming_code"), 3, QLDPC_SCHEDULES),
    "rm15": FixedCode(_qldpc("reed_muller_15_code"), 3, QLDPC_SCHEDULES),
    "lp": FixedCode(_qldpc("lifted_product_code"), 3, QLDPC_SCHEDULES),
    "gb46": FixedCode(_qldpc("generalized_bicycle_code", 46), 9, QLDPC_SCHEDULES),
    "gb48": FixedCode(_qldpc("generalized_bicycle_code", 48), 8, QLDPC_SCHEDULES),
    "gb126": FixedCode(_qldpc("generalized_bicycle_code", 126), 8, QLDPC_SCHEDULES),
    "qt36": FixedCode(_qldpc("quantum_tanner_code", 1), 3, QLDPC_SCHEDULES),
    "qt72": FixedCode(_qldpc("quantum_tanner_code", 0), 4, QLDPC_SCHEDULES),
    "bpc36": FixedCode(_quits("make_bpc_circuit", 36), 4, PRODUCT_SCHEDULES),
    "bpc54": FixedCode(_quits("make_bpc_circuit", 54), 4, PRODUCT_SCHEDULES),
    "bpc72": FixedCode(_quits("make_bpc_circuit", 72), 8, PRODUCT_SCHEDULES),
    "bpc90": FixedCode(_quits("make_bpc_circuit", 90), 10, PRODUCT_SCHEDULES),
    "bpc108": FixedCode(_quits("make_bpc_circuit", 108), 8, PRODUCT_SCHEDULES),
    "bpc126": FixedCode(_quits("make_bpc_circuit", 126), 10, PRODUCT_SCHEDULES),
    "bpc144": FixedCode(_quits("make_bpc_circuit", 144), 12, PRODUCT_SCHEDULES),
    "bpc162": FixedCode(_quits("make_bpc_circuit", 162), 12, PRODUCT_SCHEDULES),
    "bpc180": FixedCode(_quits("make_bpc_circuit", 180), 16, PRODUCT_SCHEDULES),
    "hgp225": FixedCode(_quits("make_hgp_circuit", 225), 6, PRODUCT_SCHEDULES),
    "hgp625": FixedCode(_quits("make_hgp_circuit", 625), 8, PRODUCT_SCHEDULES),
    "hgp900": FixedCode(_quits("make_hgp_circuit", 900), 10, PRODUCT_SCHEDULES),
    "lcs15": FixedCode(_quits("make_lcs_circuit", 15), 3, LCS_SCHEDULES),
    "lcs52": FixedCode(_quits("make_lcs_circuit", 52), 4, LCS_SCHEDULES),
    "lcs65": FixedCode(_quits("make_lcs_circuit", 65), 5, LCS_SCHEDULES),
    "lcs150": FixedCode(_quits("make_lcs_circuit", 150), 6, LCS_SCHEDULES),
    "lcs175": FixedCode(_quits("make_lcs_circuit", 175), 7, LCS_SCHEDULES),
    "qlp544": FixedCode(_quits("make_qlp_circuit"), 12, PRODUCT_SCHEDULES),
    "mitten150": FixedCode(_quits("make_mitten_circuit"), 8, MITTEN_SCHEDULES),
    "hyperbolic_floquet_60": FixedCode(_hyperbolic("4.10.10_n=60_k=5_d=3.stim"), 3, rounds=_hyperbolic_rounds),
    "hyperbolic_floquet_120": FixedCode(_hyperbolic("4.10.10_n=120_k=8_d=3.stim"), 3, rounds=_hyperbolic_rounds),
    "hyperbolic_floquet_160": FixedCode(_hyperbolic("4.10.10_n=160_k=10_d=4.stim"), 4, rounds=_hyperbolic_rounds),
    "hyperbolic_floquet_240": FixedCode(_hyperbolic("4.10.8_n=240_k=8_d=4.stim"), 4, rounds=_hyperbolic_rounds),
    "hyperbolic_floquet_256": FixedCode(_hyperbolic("8.8.8_n=256_k=34_d=4.stim"), 4, rounds=_hyperbolic_rounds),
    "hyperbolic_floquet_320": FixedCode(_hyperbolic("4.10.10_n=320_k=18_d=5.stim"), 5, rounds=_hyperbolic_rounds),
    "semi_hyperbolic_floquet_64": FixedCode(_hyperbolic("semi_l=2_8.8.8_n=64_k=4_d=3.stim"), 3, rounds=_hyperbolic_rounds),
    "semi_hyperbolic_floquet_144": FixedCode(_hyperbolic("semi_l=3_8.8.8_n=144_k=4_d=4.stim"), 4, rounds=_hyperbolic_rounds),
    "semi_hyperbolic_floquet_256": FixedCode(_hyperbolic("semi_l=4_8.8.8_n=256_k=4_d=6.stim"), 6, rounds=_hyperbolic_rounds),
    "semi_hyperbolic_floquet_400": FixedCode(_hyperbolic("semi_l=5_8.8.8_n=400_k=4_d=7.stim"), 7, rounds=_hyperbolic_rounds),
    "semi_hyperbolic_floquet_576": FixedCode(_hyperbolic("semi_l=6_8.8.8_n=576_k=4_d=8.stim"), 8, rounds=_hyperbolic_rounds),
}


class MemoryCodeCircuit(StimCodeCircuit):
    def stim_detectors(self):
        # StimCodeCircuit makes every OBSERVABLE_INCLUDE its own logical; Floquet circuits build
        # one observable from several of them, so merge them back by observable index
        indices = [int(inst.gate_args_copy()[0]) for inst in self.stim_circuit.flattened()
                   if inst.name == "OBSERVABLE_INCLUDE"]
        merged = {}
        for index, logical in zip(indices, self.logicals):
            merged.setdefault(index, Counter()).update(logical["clbits"])
        logicals = [{"clbits": [c for c, n in merged[i].items() if n % 2]} for i in sorted(merged)]
        return self.detectors, logicals


def _remap_target(t: stim.GateTarget, index: dict) -> stim.GateTarget:
    if t.is_combiner or t.is_measurement_record_target or t.is_sweep_bit_target:
        return t
    q = index[t.value]
    if t.is_x_target:
        return stim.target_x(q, t.is_inverted_result_target)
    if t.is_y_target:
        return stim.target_y(q, t.is_inverted_result_target)
    if t.is_z_target:
        return stim.target_z(q, t.is_inverted_result_target)
    return stim.target_inv(q) if t.is_inverted_result_target else q


def _flatten_and_compact(circuit: stim.Circuit) -> stim.Circuit:
    """Unroll REPEAT blocks (StimCodeCircuit cannot name their registers uniquely) and renumber
    the used qubits 0..n-1, since some generators index qubits by sparse coordinates"""
    circuit = circuit.flattened()
    used = sorted({t.value for inst in circuit for t in inst.targets_copy() if t.is_qubit_target
                   or t.is_x_target or t.is_y_target or t.is_z_target})
    index = {q: i for i, q in enumerate(used)}
    result = stim.Circuit()
    for inst in circuit:
        targets = [_remap_target(t, index) for t in inst.targets_copy()]
        result.append(stim.CircuitInstruction(inst.name, targets, inst.gate_args_copy()))
    return result


@lru_cache(maxsize=None)
def _fixed_num_qubits(code_name: str) -> int:
    return make_circuit(code_name, CODES[code_name].distance, 2).num_qubits


def num_rounds(code_name: str, rounds: int) -> int:
    return CODES[code_name].rounds(rounds) if code_name in CODES else rounds


def resolve_schedule(code_name: str, schedule: Optional[str]) -> Optional[str]:
    """The schedule a code is built with: the requested one, or the code's default"""
    schedules = CODES[code_name].schedules if code_name in CODES else ()
    if schedule is None:
        return schedules[0] if schedules else None
    if schedule not in schedules:
        raise ValueError(f"{code_name} supports the schedules {list(schedules)}, not {schedule}")
    return schedule


def make_circuit(code_name: str, d: int, rounds: Optional[int], schedule: Optional[str] = None) -> stim.Circuit:
    code = CODES[code_name]
    schedule = resolve_schedule(code_name, schedule)
    if isinstance(code, FixedCode):
        if d is not None and d != code.distance:
            raise ValueError(f"{code_name} only has distance {code.distance}, got {d}")
        return _flatten_and_compact(code.make(rounds or code.distance, schedule))
    return _flatten_and_compact(code.make(d, rounds or d, schedule))


def min_num_qubits(code_name: str, d: int) -> int:
    code = CODES[code_name]
    if isinstance(code, FixedCode):
        if d != code.distance:
            raise ValueError(f"{code_name} only has distance {code.distance}, got {d}")
        return _fixed_num_qubits(code_name)
    return code.num_qubits(d)


def max_distance(code_name: str, n: int) -> int:
    code = CODES[code_name]
    if isinstance(code, FixedCode):
        return code.distance if _fixed_num_qubits(code_name) <= n else 0
    d = code.min_d
    if code.num_qubits(d) > n:
        return 0
    while code.num_qubits(d + code.step) <= n:
        d += code.step
    return d
