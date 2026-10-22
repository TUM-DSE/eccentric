import csv

import pytest

from eccentric.codes import resolve_schedule
from eccentric.codes.catalog import CODES, make_circuit
from eccentric.experiments import Skip, Task, logical_error
from eccentric.experiments.base import load_backend, noisy_circuit, per_round_error_rate
from eccentric.runner import expand
from eccentric.utils import save_results_to_csv


def task(**overrides):
    params = dict(experiment="T", num_samples=200, backend="custom_full", backend_size=100, code="surface",
                  distance=3, cycles=2, decoder="bposd", error_type="sd6", error_probability=0.001)
    params.update(overrides)
    return Task(**params)


# Syndrome schedules

def test_default_schedule_is_first():
    assert resolve_schedule("bpc36", None) == "cardinal"
    assert resolve_schedule("lcs52", None) == "zxcoloration"
    assert resolve_schedule("surface", None) is None


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        resolve_schedule("bpc36", "edge_coloring")
    with pytest.raises(ValueError):
        resolve_schedule("surface", "cardinal")


@pytest.mark.parametrize("code,schedule", [
    (code, schedule)
    for code in ("toric", "bpc36", "bb72", "gb48")
    for schedule in CODES[code].schedules
])
def test_every_schedule_builds_a_valid_circuit(code, schedule):
    d = getattr(CODES[code], "distance", 3)
    circuit = make_circuit(code, d, 2, schedule)
    circuit.detector_error_model()
    assert circuit.num_qubits == make_circuit(code, d, 2).num_qubits


def test_schedules_are_a_sweep_axis():
    _, tasks = expand({
        "name": "T", "num_samples": 1, "backends": ["custom_full"], "backends_sizes": [200],
        "codes": ["bpc36"], "decoders": ["bposd"], "error_types": ["sd6"], "error_probabilities": [0.001],
        "schedules": ["cardinal", "zxcoloration"],
    })
    assert sorted(t.schedule for t in tasks) == ["cardinal", "zxcoloration"]


def test_unsupported_schedule_skips_the_configuration():
    with pytest.raises(Skip):
        logical_error.run(task(schedule="cardinal"))


def test_schedule_is_reported():
    (row,) = logical_error.run(task(code="toric", backend_size=36, schedule="edge_coloring_xz"))
    assert row["schedule"] == "edge_coloring_xz"


# Per-round error rates

def test_per_round_error_rate():
    assert per_round_error_rate(0.1, 1) == pytest.approx(0.1)
    p = 0.01
    total = (1 - (1 - 2 * p) ** 5) / 2
    assert per_round_error_rate(total, 5) == pytest.approx(p)
    assert per_round_error_rate(0.6, 3) != per_round_error_rate(0.6, 3)


def test_rows_use_the_rounds_the_circuit_runs():
    (row,) = logical_error.run(task(code="planar_honeycomb", backend_size=68, cycles=3))
    assert row["cycles"] == 3 and row["rounds"] == 4
    assert "logical_error_rate_per_round" in row


# Detector coordinates

def test_detector_coordinates_survive_the_pipeline():
    t = task(code="chromobius_color", decoder="chromobius", distance=5, backend_size=19, cycles=3)
    circuit = noisy_circuit(t, load_backend(t), 5, 3)
    coords = [inst.gate_args_copy() for inst in circuit.flattened() if inst.name == "DETECTOR"]
    assert coords and all(len(c) == 4 for c in coords)
    (row,) = logical_error.run(t)
    assert 0.0 <= float(row["logical_error_rate"]) < 0.5


# Loss with detection

@pytest.mark.parametrize("error_type,heralded", [("erasure", True), ("erasure_undetected", False)])
def test_erasure_keeps_detectors_deterministic(error_type, heralded):
    t = task(error_type=error_type, error_probability=0.01)
    circuit = noisy_circuit(t, load_backend(t), 3, 2)
    heralds = sum(1 for inst in circuit.flattened() if inst.name == "DETECTOR" and inst.tag == "herald")
    assert (heralds > 0) == heralded
    # Raises if remapping the measurement records broke a detector
    circuit.detector_error_model()


def test_atom_loss_is_detected_at_measurement():
    t = task(backend="real_harvard", backend_size=None, error_type="real_harvard", error_probability=None)
    circuit = noisy_circuit(t, load_backend(t), 3, 2)
    circuit.detector_error_model()
    heralds = [inst for inst in circuit.flattened() if inst.name == "DETECTOR" and inst.tag == "herald"]
    measured = sum(len(inst.targets_copy()) for inst in circuit.flattened() if inst.name == "M" and inst.tag != "herald")
    assert len(heralds) == measured


def test_detected_erasures_are_corrected_better():
    detected = float(logical_error.run(task(error_type="erasure", error_probability=0.03))[0]["logical_error_rate"])
    undetected = float(logical_error.run(task(error_type="erasure_undetected", error_probability=0.03))[0]["logical_error_rate"])
    assert detected < undetected


# Results

def test_new_columns_extend_existing_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_results_to_csv({"a": 1, "b": 2}, "X")
    save_results_to_csv({"b": 4, "a": 3, "c": 5}, "X")
    with open("experiment_results/X/results.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"a": "1", "b": "2", "c": ""}, {"a": "3", "b": "4", "c": "5"}]
