"""Codes that are built as noiseless Stim memory circuits and converted with StimCodeCircuit.

A ``CodeFamily`` is parametrized by its distance, a ``FixedCode`` is a single [[n, k, d]] instance.
``make`` returns the circuit for a number of syndrome rounds (and a distance, for families).
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import stim
from qiskit_qec.circuits import StimCodeCircuit


@dataclass(frozen=True)
class CodeFamily:
    make: Callable[[int, int], stim.Circuit]
    num_qubits: Callable[[int], int]
    min_d: int = 3
    step: int = 2


@dataclass(frozen=True)
class FixedCode:
    make: Callable[[int], stim.Circuit]
    distance: int


def _repetition(d, rounds):
    return stim.Circuit.generated("repetition_code:memory", rounds=rounds, distance=d)


def _xzzx(d, rounds):
    from .xzzx_code import make_xzzx_circuit
    return make_xzzx_circuit(d, rounds)


def _toric(d, rounds):
    from .qldpc_codes import make_memory_circuit, toric_code
    return make_memory_circuit(toric_code(d), rounds)


def _qldpc(builder_name, *args):
    def make(rounds):
        from . import qldpc_codes
        return qldpc_codes.make_memory_circuit(getattr(qldpc_codes, builder_name)(*args), rounds)
    return make


def _bb(n):
    def make(rounds):
        from .gross_code import make_bb_circuit
        return make_bb_circuit(n, rounds)
    return make


def _quits(builder_name, *args):
    def make(rounds):
        from . import quits_codes
        return getattr(quits_codes, builder_name)(*args, rounds)
    return make


def _hyperbolic(filename):
    def make(rounds):
        from .hyperbolic_floquet import make_hyperbolic_floquet_circuit
        return make_hyperbolic_floquet_circuit(filename, rounds)
    return make


def _periodic_honeycomb(d, rounds):
    from .honeycomb import make_periodic_honeycomb_circuit
    return make_periodic_honeycomb_circuit(d, rounds)


def _planar_honeycomb(d, rounds):
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
    "toric": CodeFamily(_toric, lambda d: 4 * d * d),
    "honeycomb": CodeFamily(_periodic_honeycomb, _periodic_honeycomb_qubits, min_d=4),
    "planar_honeycomb": CodeFamily(_planar_honeycomb, _planar_honeycomb_qubits),
    "bb54": FixedCode(_bb(54), 6),
    "bb72": FixedCode(_bb(72), 6),
    "bb90": FixedCode(_bb(90), 10),
    "bb98": FixedCode(_bb(98), 12),
    "bb108": FixedCode(_bb(108), 10),
    "bb162": FixedCode(_bb(162), 14),
    "bb180": FixedCode(_bb(180), 16),
    "bb288": FixedCode(_bb(288), 18),
    "bb756": FixedCode(_bb(756), 34),
    "hgp_hamming": FixedCode(_qldpc("hgp_hamming_code"), 3),
    "rm15": FixedCode(_qldpc("reed_muller_15_code"), 3),
    "lp": FixedCode(_qldpc("lifted_product_code"), 3),
    "gb46": FixedCode(_qldpc("generalized_bicycle_code", 46), 9),
    "gb48": FixedCode(_qldpc("generalized_bicycle_code", 48), 8),
    "gb126": FixedCode(_qldpc("generalized_bicycle_code", 126), 8),
    "qt36": FixedCode(_qldpc("quantum_tanner_code", 1), 3),
    "qt72": FixedCode(_qldpc("quantum_tanner_code", 0), 4),
    "bpc36": FixedCode(_quits("make_bpc_circuit", 36), 4),
    "bpc54": FixedCode(_quits("make_bpc_circuit", 54), 4),
    "bpc72": FixedCode(_quits("make_bpc_circuit", 72), 8),
    "bpc90": FixedCode(_quits("make_bpc_circuit", 90), 10),
    "bpc108": FixedCode(_quits("make_bpc_circuit", 108), 8),
    "bpc126": FixedCode(_quits("make_bpc_circuit", 126), 10),
    "bpc144": FixedCode(_quits("make_bpc_circuit", 144), 12),
    "bpc162": FixedCode(_quits("make_bpc_circuit", 162), 12),
    "bpc180": FixedCode(_quits("make_bpc_circuit", 180), 16),
    "hgp225": FixedCode(_quits("make_hgp_circuit", 225), 6),
    "hgp625": FixedCode(_quits("make_hgp_circuit", 625), 8),
    "hgp900": FixedCode(_quits("make_hgp_circuit", 900), 10),
    "lcs15": FixedCode(_quits("make_lcs_circuit", 15), 3),
    "lcs52": FixedCode(_quits("make_lcs_circuit", 52), 4),
    "lcs65": FixedCode(_quits("make_lcs_circuit", 65), 5),
    "lcs150": FixedCode(_quits("make_lcs_circuit", 150), 6),
    "lcs175": FixedCode(_quits("make_lcs_circuit", 175), 7),
    "qlp544": FixedCode(_quits("make_qlp_circuit"), 12),
    "mitten150": FixedCode(_quits("make_mitten_circuit"), 8),
    "hyperbolic_floquet_60": FixedCode(_hyperbolic("4.10.10_n=60_k=5_d=3.stim"), 3),
    "hyperbolic_floquet_120": FixedCode(_hyperbolic("4.10.10_n=120_k=8_d=3.stim"), 3),
    "hyperbolic_floquet_160": FixedCode(_hyperbolic("4.10.10_n=160_k=10_d=4.stim"), 4),
    "hyperbolic_floquet_240": FixedCode(_hyperbolic("4.10.8_n=240_k=8_d=4.stim"), 4),
    "hyperbolic_floquet_256": FixedCode(_hyperbolic("8.8.8_n=256_k=34_d=4.stim"), 4),
    "hyperbolic_floquet_320": FixedCode(_hyperbolic("4.10.10_n=320_k=18_d=5.stim"), 5),
    "semi_hyperbolic_floquet_64": FixedCode(_hyperbolic("semi_l=2_8.8.8_n=64_k=4_d=3.stim"), 3),
    "semi_hyperbolic_floquet_144": FixedCode(_hyperbolic("semi_l=3_8.8.8_n=144_k=4_d=4.stim"), 4),
    "semi_hyperbolic_floquet_256": FixedCode(_hyperbolic("semi_l=4_8.8.8_n=256_k=4_d=6.stim"), 6),
    "semi_hyperbolic_floquet_400": FixedCode(_hyperbolic("semi_l=5_8.8.8_n=400_k=4_d=7.stim"), 7),
    "semi_hyperbolic_floquet_576": FixedCode(_hyperbolic("semi_l=6_8.8.8_n=576_k=4_d=8.stim"), 8),
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


def make_circuit(code_name: str, d: int, rounds: Optional[int]) -> stim.Circuit:
    code = CODES[code_name]
    if isinstance(code, FixedCode):
        if d is not None and d != code.distance:
            raise ValueError(f"{code_name} only has distance {code.distance}, got {d}")
        return _flatten_and_compact(code.make(rounds or code.distance))
    return _flatten_and_compact(code.make(d, rounds or d))


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
