"""Correlated matching (mwpm_corr) and level-by-level decoding of concatenated codes (hier_bposd)"""
import collections

import numpy as np
import pytest
import stim

from eccentric.decoders import decode
from eccentric.experiments.base import Task, load_backend, noisy_circuit


def _noisy(code, d, operation=None, p=1e-3):
    from eccentric.codes import get_min_n
    size = get_min_n(code, d, operation) if operation else get_min_n(code, d)
    t = Task(experiment="test", num_samples=1, backend="custom_full", backend_size=size,
             code=code, distance=d, cycles=d, error_type="sd6", error_probability=p, operation=operation)
    return noisy_circuit(t, load_backend(t), d, d)


def _raw(circuit, shots=3000):
    return circuit.compile_detector_sampler().sample(shots, separate_observables=True)[1].any(axis=1).mean()


def test_correlated_matching_is_at_least_as_good_as_plain_matching_on_surface_memory():
    from eccentric.decoders.correlated_matching import matching_decode
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
                                     after_clifford_depolarization=0.006, before_measure_flip_probability=0.006)
    plain = matching_decode(circuit, 20000, False, correlated=False)
    correlated = matching_decode(circuit, 20000, False, correlated=True)
    assert correlated < 1.1 * plain


def test_correlated_matching_decodes_what_stim_cannot_decompose():
    pytest.importorskip("qldpc.experimental.surgery")
    from eccentric.decoders.correlated_matching import decomposed_dem, _targets
    circuit = _noisy("surface", 5, "transversal_cnot", 2e-3)
    with pytest.raises(ValueError):
        circuit.detector_error_model(decompose_errors=True)
    assert decode("surface", circuit, 200, "mwpm", "custom_full", "sd6") is None
    dem, split, mismatched, _ = decomposed_dem(circuit, False)
    assert split > 0 and mismatched <= split
    components = [c for inst in dem.flattened() if inst.type == "error" for c in _targets(inst)]
    assert all(1 <= len(dets) <= 2 for dets, _ in components)
    ler = decode("surface", circuit, 3000, "mwpm_corr", "custom_full", "sd6")
    assert ler < 0.1 and ler < _raw(circuit)


def test_errors_more_likely_than_not_are_flipped_exactly():
    from eccentric.decoders.correlated_matching import matching_decode
    # A repetition code whose first data qubit flips with probability 0.9 before every round: plain and
    # correlated matching must agree with each other and stay far below the raw error rate
    base = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5,
                                  after_clifford_depolarization=0.01, before_round_data_depolarization=0.01)
    text = str(base).replace("DEPOLARIZE1(0.01) 0 2 4 6 8", "DEPOLARIZE1(0.01) 0 2 4 6 8\nX_ERROR(0.9) 0", 1)
    circuit = stim.Circuit(text)
    assert any(i.type == "error" and i.args_copy()[0] > 0.5 for i in circuit.detector_error_model(decompose_errors=True).flattened())
    plain = matching_decode(circuit, 5000, False, correlated=False)
    correlated = matching_decode(circuit, 5000, False, correlated=True)
    assert correlated < 0.05 and abs(correlated - plain) < 0.02


def test_check_levels_of_the_concatenated_codes():
    from eccentric.decoders.hierarchical import check_levels
    expected = {("mhc36", 4): {1: 6, 2: 4}, ("mhc216", 8): {1: 36, 2: 24, 3: 16},
                ("steane", 9): {1: 21, 2: 3}, ("steane", 27): {1: 147, 2: 21, 3: 3}}
    for (code, d), counts in expected.items():
        levels, top = check_levels(code, d)
        assert dict(collections.Counter(levels.tolist())) == counts and top == max(counts)
    assert check_levels("steane", 3) is None and check_levels("surface", 3) is None


def test_hierarchical_decodes_mhc36_and_steane9():
    from eccentric.decoders.hierarchical import detector_levels
    for code, d in (("mhc36", 4), ("steane", 9)):
        circuit = _noisy(code, d)
        levels = detector_levels(circuit, code, d)
        assert set(levels.tolist()) == {1, 2} and len(levels) == circuit.num_detectors
        raw = _raw(circuit, 400)
        for decoder in ("hier_bposd", "hier_bposd_hard"):
            ler = decode(code, circuit, 200, decoder, "custom_full", "sd6", distance=d)
            assert ler is not None and ler <= raw


def test_hierarchical_refuses_codes_without_levels():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                                     after_clifford_depolarization=0.001)
    assert decode("surface", circuit, 10, "hier_bposd", "custom_full", "sd6", distance=3) is None
