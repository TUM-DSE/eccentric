"""CHANGELOG fix 2: the concatenated Steane circuits read the logical qubit out of the data qubits"""
import pytest
import stim

from eccentric.codes.utils import get_code
from validation.run_job import with_sd6_noise


def steane(d):
    return get_code("steane", d, d).stim_circuit


def observable_qubits(circuit):
    """The qubits whose measurements make up the logical observable"""
    measured, qubits = [], set()
    for inst in circuit.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            measured += [t.value for t in inst.targets_copy()]
        elif inst.name == "OBSERVABLE_INCLUDE":
            qubits |= {measured[len(measured) + t.value] for t in inst.targets_copy()}
    return sorted(qubits)


def with_error_after_preparation(circuit, qubits):
    """The circuit with an X on the given qubits right after they are first reset"""
    flat = circuit.flattened()
    out, done = stim.Circuit(), False
    for inst in flat:
        if not done and inst.name not in ("R", "RZ", "QUBIT_COORDS", "TICK"):
            out.append("X_ERROR", qubits, 1.0)
            done = True
        out.append(inst)
    return out


@pytest.mark.parametrize("d, qubits", [(3, 13), (9, 97), (27, 685)])
def test_no_readout_ancilla(d, qubits):
    assert steane(d).num_qubits == qubits


@pytest.mark.parametrize("d", [3, 9])
def test_logical_x_flips_the_result_without_detection(d):
    circuit = steane(d)
    data = observable_qubits(circuit)
    assert len(data) == (7 if d == 3 else 49)
    dets, obs = with_error_after_preparation(circuit, data).compile_detector_sampler().sample(1, separate_observables=True)
    assert obs.all() and not dets.any()
    dets, obs = with_error_after_preparation(circuit, data[:1]).compile_detector_sampler().sample(1, separate_observables=True)
    # One data error flips the raw parity too, but fires detectors, so the decoder corrects it
    assert dets.any()


@pytest.mark.parametrize("d", [3, 9])
def test_circuit_distance_is_the_code_distance(d):
    dem = with_sd6_noise(steane(d), 1e-3).detector_error_model(approximate_disjoint_errors=True)
    undetectable = [e for e in dem.flattened() if e.type == "error" and all(not t.is_relative_detector_id() for t in e.targets_copy())
                    and any(t.is_logical_observable_id() for t in e.targets_copy())]
    assert not undetectable
    assert len(dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)) == d
