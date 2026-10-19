import pytest
from qiskit import QuantumCircuit

from eccentric.transpilers import translate
from eccentric.transpilers.translators import qiskit_h2, qiskit_ibm_heron


def decomposable_circuit():
    qc = QuantumCircuit(3)
    qc.u(0.5, 1.2, 2.3, 0)
    qc.t(1)
    qc.swap(0, 2)
    qc.ccx(0, 1, 2)
    return qc


@pytest.mark.parametrize("gate_set,allowed", [("h2", qiskit_h2), ("ibm_heron", qiskit_ibm_heron)])
@pytest.mark.parametrize("level", [0, 2, 3])
def test_qiskit_translation_uses_target_gate_set(gate_set, allowed, level):
    translated = translate(decomposable_circuit(), "qiskit", gate_set, level)
    used = {inst.operation.name for inst in translated.data}
    assert used <= set(allowed)


@pytest.mark.parametrize("gate_set", ["h2", "ibm_heron"])
def test_tket_translation_returns_circuit(gate_set):
    translated = translate(decomposable_circuit(), "tket", gate_set, 0)
    assert translated.n_qubits == 3


def test_unknown_gate_set():
    with pytest.raises(ValueError):
        translate(QuantumCircuit(1), "qiskit", "unknown")


def test_unknown_method():
    with pytest.raises(ValueError):
        translate(QuantumCircuit(1), "unknown_method")
