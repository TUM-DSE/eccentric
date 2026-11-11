import math

import numpy as np
import pytest
import stim

from qec_matchmaker.features import FEATURE_NAMES, features_from_circuit, features_from_dem, features_from_matrices


def surface(p=0.003, d=3):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p)


def matrices(dem):
    probs, H, L = [], [], []
    for inst in dem.flattened():
        if inst.type == "error":
            probs.append(inst.args_copy()[0])
            H.append([t.val for t in inst.targets_copy() if t.is_relative_detector_id()])
            L.append([t.val for t in inst.targets_copy() if t.is_logical_observable_id()])
    h = np.zeros((dem.num_detectors, len(probs)), dtype=np.uint8)
    l = np.zeros((dem.num_observables, len(probs)), dtype=np.uint8)
    for j, (dets, obs) in enumerate(zip(H, L)):
        h[dets, j] = 1
        l[obs, j] = 1
    return h, np.array(probs), l


def test_all_features_present_and_circuit_counts():
    f = features_from_circuit(surface())
    assert set(f) == set(FEATURE_NAMES)
    assert f["detectors"] == 24 and f["observables"] == 1 and f["qubits"] == 26
    assert f["graphlike"] == 0 and f["decomposable"] == 1 and f["rounds"] == 4
    assert 0 < f["det_fire_mean"] < f["det_fire_max"] < 0.5


def test_dem_and_matrices_agree():
    dem = surface().detector_error_model(approximate_disjoint_errors=True)
    a = features_from_dem(dem)
    b = features_from_matrices(*matrices(dem))
    for name in ("detectors", "mechanisms", "mechanism_weight_mean", "error_mass", "det_fire_mean", "det_fire_p90",
                 "logical_mass", "graphlike", "hyperedge_fraction"):
        assert a[name] == pytest.approx(b[name]), name
    assert math.isnan(b["rounds"]) and math.isnan(b["qubits"])


def test_firing_probability_is_exact_for_independent_errors():
    # Two independent errors on one detector: it fires when exactly one happens
    dem = stim.DetectorErrorModel("error(0.1) D0\nerror(0.2) D0 L0\nerror(0.05) L0")
    f = features_from_dem(dem)
    assert f["det_fire_mean"] == pytest.approx(0.1 * 0.8 + 0.9 * 0.2)
    assert (f["undetectable_faults"], f["undetectable_probability"]) == (1, pytest.approx(0.05))
    assert f["logical_mass"] == pytest.approx(0.25)


def test_hyperedges_and_decomposition():
    circuit = stim.Circuit("""
        R 0 1 2
        X_ERROR(0.1) 0
        CX 0 1 0 2
        M 0 1 2
        DETECTOR rec[-1]
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """)
    f = features_from_circuit(circuit)
    assert f["mechanism_weight_max"] == 3 and f["hyperedge_fraction"] == 1 and f["graphlike"] == 0
    assert f["decomposable"] == 0 and f["two_qubit_gates"] == 2


def test_detector_types_tell_css_circuits_from_others():
    from qec_matchmaker.detector_types import detector_type_features
    for basis in "zx":
        css = stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=3, rounds=3,
                                     after_clifford_depolarization=0.001, before_measure_flip_probability=0.001)
        f = detector_type_features(css)
        assert f["mixed_detectors"] == 0 and f["y_parts_unmatched"] == 0 and f["observables_both_sides"] == 0
    xyz = stim.Circuit.generated("color_code:memory_xyz", distance=3, rounds=3, after_clifford_depolarization=0.001)
    assert detector_type_features(xyz)["mixed_detectors"] > 0.5


def test_compiled_hadamard_frames_keep_a_css_circuit_css():
    # The X-basis memory written with R + H and H + M instead of RX and MX, as compilers emit it: the data qubits
    # sit in a Hadamard frame, and their regions read as X there
    from qec_matchmaker.detector_types import detector_type_features
    css = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3,
                                 after_clifford_depolarization=0.001).flattened()
    compiled = stim.Circuit()
    for inst in css:
        targets = inst.targets_copy()
        if inst.name == "RX":
            compiled.append("R", targets)
            compiled.append("H", targets)
        elif inst.name == "MX":
            compiled.append("H", targets)
            compiled.append("M", targets, inst.gate_args_copy())
        else:
            compiled.append(inst)
    assert compiled.detector_error_model().num_errors == css.detector_error_model().num_errors
    assert detector_type_features(compiled)["mixed_detectors"] == 0
