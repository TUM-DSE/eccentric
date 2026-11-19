import math

import scipy.stats._stats_mstats_common
import scipy.stats._stats_py
import stim

# Based on https://github.com/Strilanc/honeycomb_threshold (periodic) and
# https://github.com/Strilanc/honeycomb-boundaries (planar). Both use the SD6 circuits, which
# measure the weight-2 edge checks with ancilla qubits and CNOTs.

# Both repositories import LinregressResult from its pre-scipy-1.11 location
if not hasattr(scipy.stats._stats_mstats_common, "LinregressResult"):
    scipy.stats._stats_mstats_common.LinregressResult = scipy.stats._stats_py.LinregressResult


def periodic_honeycomb_size(d: int):
    if d % 2 or d < 4:
        raise ValueError(f"Periodic honeycomb code needs an even distance >= 4, got {d}")
    return d, 6 * math.ceil(d / 4)


def make_periodic_honeycomb_circuit(d: int, rounds: int) -> stim.Circuit:
    from honeycomb_layout import HoneycombLayout

    width, height = periodic_honeycomb_size(d)
    layout = HoneycombLayout(
        data_width=width, data_height=height, sub_rounds=3 * rounds, style="SD6", obs="H", noise=0
    )
    return layout.make_circuit().without_noise()


def periodic_honeycomb_num_qubits(d: int) -> int:
    width, height = periodic_honeycomb_size(d)
    return int(width * height * 2.5)


def planar_honeycomb_layout(d: int, rounds: int):
    from hcb.codes.honeycomb.layout import HoneycombLayout

    width, height = HoneycombLayout.unsheared_size_for_code_distance(d, "SD6")
    return HoneycombLayout(
        data_width=width,
        data_height=height,
        rounds=max(4, rounds + rounds % 2),
        noise_level=0,
        noisy_gate_set="SD6",
        tested_observable="H",
    )


def make_planar_honeycomb_circuit(d: int, rounds: int) -> stim.Circuit:
    from hcb.codes.honeycomb.circuit_maker import HoneycombCircuitMaker

    maker = HoneycombCircuitMaker(layout=planar_honeycomb_layout(d, rounds))
    maker.process()
    return maker.final_ideal_circuit()


def planar_honeycomb_num_qubits(d: int) -> int:
    return planar_honeycomb_layout(d, 4).num_used_qubits()
