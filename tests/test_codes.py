import stim
import pytest

from eccentric.codes import get_code, get_max_d, get_min_n
from eccentric.codes.catalog import CODES, CodeFamily, FixedCode, make_circuit


def few_rounds(name, d):
    """Two rounds keep the tests fast; codes with a fixed number of rounds (lattice surgery) run d"""
    return d if getattr(CODES.get(name), "rounds_fixed", False) else 2
from eccentric.experiments import Task, logical_error
from qiskit_qec.utils import get_stim_circuits

NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1",
             "PAULI_CHANNEL_2", "E", "ELSE_CORRELATED_ERROR", "HERALDED_ERASE"}


# QUITS takes up to a minute to schedule these, run them with `pytest -m slow`
SLOW = {"hgp625", "hgp900", "qlp544"}


def all_codes(names=CODES):
    return [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in sorted(names)]


def smallest_distance(name):
    code = CODES[name]
    return code.distance if isinstance(code, FixedCode) else code.min_d


@pytest.mark.parametrize("name", all_codes())
def test_circuit_is_noiseless_with_deterministic_detectors(name):
    d = smallest_distance(name)
    circuit = make_circuit(name, d, few_rounds(name, d))
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
        assert code.num_qubits(d) == make_circuit(name, d, few_rounds(name, d)).num_qubits


@pytest.mark.parametrize("name", all_codes())
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


@pytest.mark.parametrize("name", all_codes())
def test_qiskit_round_trip_keeps_detectors(name):
    code = get_code(name, smallest_distance(name), few_rounds(name, smallest_distance(name)))
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
    ("cbb30", "bposd"),
    ("tb30", "relay_bp"),
    ("mhc36", "bposd"),
    ("hgp225", "bposd"),
    ("lcs52", "bposd"),
    ("mitten150", "bposd"),
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


# Codes whose memory circuits exist in both bases; the others must refuse the X basis
Z_ONLY = {"repetition", "honeycomb", "planar_honeycomb"} | {n for n in CODES if "hyperbolic_floquet" in n}


@pytest.mark.parametrize("name", all_codes(sorted(set(CODES) - Z_ONLY)) + ["surface", "hh", "color", "bacon", "gross"])
def test_x_basis_circuit_matches_z_basis(name):
    d = smallest_distance(name) if name in CODES else {"gross": 12}.get(name, 3)
    z = get_code(name, d, few_rounds(name, d), None, "Z").stim_circuit
    x = get_code(name, d, few_rounds(name, d), None, "X").stim_circuit
    # Raises if a detector or observable is not deterministic
    x.detector_error_model()
    assert x.num_qubits == z.num_qubits
    assert x.num_observables == z.num_observables and x.num_detectors > 0
    # The observable is sensitive to other errors: its detecting region (the Pauli errors that flip it) differs
    # (circuits may measure X as H then M, and Bacon-Shor toggles the data qubits' frame every half-round)
    zr, xr = (logical_region(c) for c in (z, x))
    assert any(zr[t] != xr[t] for t in zr)
    if name in ("surface", "hh", "color", "gross", "bb72", "toric", "bpc36"):
        # Standard CSS circuits: mid-circuit, X errors flip the Z-basis observable and Z errors the X-basis one
        middle = sorted(zr)[len(zr) // 2]
        assert set(str(zr[middle])[1:]) <= {"_", "Z"} and set(str(xr[middle])[1:]) <= {"_", "X"}


def logical_region(circuit):
    return circuit.detecting_regions(targets=["L0"])[stim.DemTarget.logical_observable_id(0)]


@pytest.mark.parametrize("name", sorted(Z_ONLY) + ["steane"])
def test_z_only_codes_refuse_x_basis(name):
    with pytest.raises((ValueError, NotImplementedError)):
        get_code(name, smallest_distance(name) if name in CODES else 3, 2, None, "X")


@pytest.mark.parametrize("name,d", [("surface", 3), ("hh", 3), ("color", 3), ("bacon", 3), ("xzzx", 3)])
def test_x_basis_keeps_the_distance(name, d):
    # Under uniform depolarizing noise, the shortest graphlike logical error is as long in X as in Z
    def noisy(c, p=1e-3):
        out = stim.Circuit()
        for inst in c.flattened():
            if inst.name in ("M", "MX", "MR", "MRX", "MZ", "MRZ"):
                out.append("Z_ERROR" if "X" in inst.name else "X_ERROR", [t.value for t in inst.targets_copy()], p)
            out.append(inst)
            if inst.name in ("CX", "CZ", "XCX"):
                out.append("DEPOLARIZE2", [t.value for t in inst.targets_copy()], p)
        return out
    lengths = [len(noisy(get_code(name, d, d, None, b).stim_circuit).shortest_graphlike_error()) for b in ("Z", "X")]
    assert lengths[0] == lengths[1]
