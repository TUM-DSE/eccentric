import stim

from qec_matchmaker.api import advise


def test_code_distance_lets_the_ler_follow_the_distance_scaling():
    # Below threshold, without its distance the LER of a distance-21 surface code is that of an average
    # training code at this noise, and advise asks for the distance; with it, far lower
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=21, rounds=21,
                                     after_clifford_depolarization=0.001, before_measure_flip_probability=0.001,
                                     after_reset_flip_probability=0.001, before_round_data_depolarization=0.001)
    without, given = advise(circuit), advise(circuit, code_distance=21)
    assert any("code distance" in w for w in without.warnings)
    assert not any("code distance" in w for w in given.warnings)
    assert given.ler[1] < without.ler[1] / 1000
