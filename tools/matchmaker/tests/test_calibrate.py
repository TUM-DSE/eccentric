import json

import numpy as np
import pytest

from qec_matchmaker import calibrate
from qec_matchmaker.features import features_from_circuit


def installed():
    return [name for name in calibrate.DECODERS if calibrate._load(name) is not None]


def test_speed_factors_only_where_both_exist():
    measured = {"mwpm": 2.0, "bposd": 3.0, "tesseract": 1.0, "chromobius": None}
    reference = {"mwpm": 1.0, "bposd": None, "relay_bp": 4.0, "tesseract": 4.0, "chromobius": 1.0}
    assert calibrate.speed_factors(measured, reference) == {"mwpm": 2.0, "tesseract": 0.25}


def test_fill_reference_feeds_default_reference(monkeypatch):
    for name in calibrate.REFERENCE:
        monkeypatch.setitem(calibrate.REFERENCE, name, None)
    assert calibrate.speed_factors({"mwpm": 1.0}) == {}
    calibrate.fill_reference({"mwpm": 0.5, "not_a_decoder": 1.0})
    assert "not_a_decoder" not in calibrate.REFERENCE
    assert calibrate.speed_factors({"mwpm": 1.0, "bposd": 2.0}) == {"mwpm": 2.0}


def test_save_load_round_trip(tmp_path):
    factors = {"mwpm": 1.5, "bposd": 0.8, "relay_bp": 2.25}
    path = tmp_path / "factors.json"
    calibrate.save(path, factors)
    assert calibrate.load(path) == factors
    assert set(json.loads(path.read_text())) >= {"speed_factors", "platform", "date"}


def test_benchmark_circuits():
    circuits = calibrate.benchmark_circuits()
    names = [name for name, _ in circuits]
    assert len(set(names)) == len(names) and {"surface_d3", "surface_d5", "color_annotated_d3"} <= set(names)
    features = {name: features_from_circuit(circuit) for name, circuit in circuits}
    assert features["surface_d3"]["decomposable"] == 1 and features["surface_d3"]["color_annotated"] == 0
    assert features["color_annotated_d3"]["color_annotated"] == 1
    for name, circuit in circuits:
        # Deterministic detectors and observables, and no fault that slips past every detector
        circuit.detector_error_model(approximate_disjoint_errors=True)
        assert features[name]["undetectable_faults"] == 0, name


def test_check_matrices_merge_equal_mechanisms():
    import stim
    pytest.importorskip("scipy")
    circuit = stim.Circuit("X_ERROR(0.1) 0\nX_ERROR(0.2) 0\nX_ERROR(0.05) 1\nM 0 1\nDETECTOR rec[-2]\n"
                           "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    chk, obs, priors = calibrate._check_matrices(circuit)
    assert chk.shape == (2, len(priors)) and obs.shape == (1, len(priors))
    columns = {(tuple(chk[:, j].nonzero()[0]), tuple(obs[:, j].nonzero()[0])): p for j, p in enumerate(priors)}
    assert columns[((0,), ())] == pytest.approx(0.1 * 0.8 + 0.9 * 0.2)
    assert columns[((1,), (0,))] == pytest.approx(0.05)


@pytest.mark.parametrize("name", installed() or [pytest.param("none", marks=pytest.mark.skip("no decoder installed"))])
def test_installed_decoders_decode(name):
    entry = calibrate._load(name)
    # The small circuits every decoder decodes well (matching does poorly on Stim's XYZ color code)
    for circuit_name, circuit in calibrate.benchmark_circuits():
        if circuit_name not in ("surface_d3", "color_annotated_d3"):
            continue
        if not calibrate._applies(name, features_from_circuit(circuit)):
            continue
        dets, obs = circuit.compile_detector_sampler(seed=1).sample(500, separate_observables=True)
        predicted = calibrate._DECODERS[name][2](entry, circuit)(dets)
        assert predicted.shape == obs.shape
        ler = np.any(predicted.astype(bool) != obs, axis=1).mean()
        assert ler < 0.5 * obs.any(axis=1).mean(), (name, circuit_name)


def test_measure_installed_only():
    if not installed():
        pytest.skip("no decoder installed")
    name = installed()[0]
    measured = calibrate.measure([name], shots=100)
    assert set(measured) == {name} and measured[name] > 0
    missing = [n for n in calibrate.DECODERS if n not in installed()]
    assert set(calibrate.measure([name, *missing], shots=50)) == {name}
    with pytest.raises(ValueError):
        calibrate.measure(["not_a_decoder"])
