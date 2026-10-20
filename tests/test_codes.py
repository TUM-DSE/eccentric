import pytest

from eccentric.codes import get_code, get_max_d, get_min_n
from eccentric.codes.catalog import CODES, CodeFamily, FixedCode, make_circuit
from eccentric.experiments import Task, logical_error
from qiskit_qec.utils import get_stim_circuits

NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1",
             "PAULI_CHANNEL_2", "E", "ELSE_CORRELATED_ERROR", "HERALDED_ERASE"}


def smallest_distance(name):
    code = CODES[name]
    return code.distance if isinstance(code, FixedCode) else code.min_d


@pytest.mark.parametrize("name", sorted(CODES))
def test_circuit_is_noiseless_with_deterministic_detectors(name):
    d = smallest_distance(name)
    circuit = make_circuit(name, d, 2)
    ops = {inst.name for inst in circuit.flattened()}
    assert not ops & NOISE_OPS
    assert circuit.num_detectors > 0 and circuit.num_observables > 0
    # Raises if a detector or observable is not deterministic
    circuit.detector_error_model()
    assert get_min_n(name, d) == circuit.num_qubits


@pytest.mark.parametrize("name", sorted(n for n, c in CODES.items() if isinstance(c, CodeFamily)))
def test_family_size_formula(name):
    code = CODES[name]
    for d in (code.min_d, code.min_d + code.step):
        assert code.num_qubits(d) == make_circuit(name, d, 2).num_qubits


@pytest.mark.parametrize("name", sorted(CODES))
def test_max_distance_matches_min_size(name):
    d = smallest_distance(name)
    n = get_min_n(name, d)
    assert get_max_d(name, n) == d
    assert get_max_d(name, n - 1) < d


def test_fixed_code_rejects_other_distances():
    with pytest.raises(ValueError):
        get_min_n("bb72", 5)


def test_rounds_change_circuit_length():
    short, long = make_circuit("toric", 3, 2), make_circuit("toric", 3, 4)
    assert long.num_detectors > short.num_detectors


@pytest.mark.parametrize("name", sorted(CODES))
def test_qiskit_round_trip_keeps_detectors(name):
    code = get_code(name, smallest_distance(name), 2)
    detectors, logicals = code.stim_detectors()
    assert code.qc.num_qubits == code.stim_circuit.num_qubits
    circuit = get_stim_circuits(code.qc, detectors=detectors, logicals=logicals)[0][0]
    assert circuit.num_detectors == code.stim_circuit.num_detectors
    used = {inst.gate_args_copy()[0] for inst in code.stim_circuit.flattened() if inst.name == "OBSERVABLE_INCLUDE"}
    assert circuit.num_observables == len(used)
    circuit.detector_error_model()


@pytest.mark.parametrize("name,decoder", [
    ("repetition", "mwpm"),
    ("xzzx", "bposd"),
    ("toric", "mwpm"),
    ("bb72", "bposd"),
    ("bpc36", "bposd"),
    ("hgp_hamming", "bposd"),
    ("rm15", "bposd"),
    ("lp", "bposd"),
    ("qt36", "bposd"),
    ("honeycomb", "bposd"),
    ("planar_honeycomb", "bposd"),
    ("hyperbolic_floquet_60", "bposd"),
    ("semi_hyperbolic_floquet_64", "mwpm"),
    ("bb54", "bposd"),
    ("bpc54", "bposd"),
    ("gb48", "bposd"),
])
def test_logical_error_end_to_end(name, decoder):
    d = smallest_distance(name)
    task = Task(
        experiment="T",
        num_samples=100,
        backend="custom_full",
        backend_size=get_min_n(name, d),
        code=name,
        distance=d,
        cycles=2,
        decoder=decoder,
        error_type="sd6",
        error_probability=0.001,
    )
    (row,) = logical_error.run(task)
    assert row["distance"] == d and row["cycles"] == 2
    assert 0.0 <= float(row["logical_error_rate"]) < 0.5
