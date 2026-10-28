import math

import numpy as np
import pytest
import stim
from qiskit.providers import QubitProperties
from qiskit.transpiler import CouplingMap

from eccentric.backends import QubitTracking
from eccentric.codes import get_min_n
from eccentric.experiments import Task, logical_error
from eccentric.noise.noise import NoiseModel

T1, T2 = 100e-6, 80e-6
TQ_TIME = 70e-9


class FakeBackend:
    """A line of qubits with uniform T1/T2"""

    def __init__(self, n):
        self.num_qubits = n
        self.coupling_map = CouplingMap.from_line(n)
        self.name = "Fake"
        self.get_remote_gates = set()

    def qubit_properties(self, q):
        return QubitProperties(t1=T1, t2=T2)


def decoherence(duration):
    p_x = 0.25 * (1 - math.exp(-duration / T1))
    p_z = (1 - math.exp(-duration / T2)) / 2 - (1 - math.exp(-duration / T1)) / 4
    return [p_x, p_x, p_z]


def model(n, **kwargs):
    backend = FakeBackend(n)
    params = dict(tq=1e-3, gate_times={"SQ": 0, "TQ": TQ_TIME, "M": 0, "R": 0}, backend=backend)
    params.update(kwargs)
    return NoiseModel(**params)


def channels(circuit: stim.Circuit, name: str):
    return [inst for inst in circuit.flattened() if inst.name == name]


@pytest.mark.parametrize("pairs", [1, 4, 20])
def test_parallel_gates_decohere_each_qubit_once(pairs):
    # Regression test: every qubit of a layer of parallel two-qubit gates decoheres for one gate
    # duration, not once per pair in the layer
    n = 2 * pairs
    circuit = stim.Circuit(f"CX {' '.join(map(str, range(n)))}")
    noisy = model(n).noisy_circuit(circuit)
    paulis = channels(noisy, "PAULI_CHANNEL_1")
    hits = [t.value for inst in paulis for t in inst.targets_copy()]
    assert sorted(hits) == list(range(n))
    for inst in paulis:
        assert inst.gate_args_copy() == pytest.approx(decoherence(TQ_TIME))


def test_two_qubit_gates_get_their_gate_error():
    circuit = stim.Circuit("CX 0 1 2 3")
    noisy = model(4, backend=None, gate_times={}).noisy_circuit(circuit)
    (dep,) = channels(noisy, "DEPOLARIZE2")
    assert dep.gate_args_copy() == pytest.approx([1e-3])
    assert [t.value for t in dep.targets_copy()] == [0, 1, 2, 3]


def test_idle_noise_hits_only_idle_qubits():
    # Every TICK closes a time step, so a trailing TICK would add an empty, fully idle step
    circuit = stim.Circuit("H 0\nTICK\nH 1 2")
    noisy = NoiseModel(sq=0, idle=1e-3).noisy_circuit(circuit)
    idles = [sorted(t.value for t in inst.targets_copy()) for inst in channels(noisy, "DEPOLARIZE1")]
    assert idles == [[1, 2], [0]]


def test_measurement_and_reset_errors():
    circuit = stim.Circuit("R 0 1\nTICK\nM 0 1")
    noisy = NoiseModel(reset=2e-3, measure=3e-3).noisy_circuit(circuit)
    flips = [(inst.gate_args_copy()[0], [t.value for t in inst.targets_copy()]) for inst in channels(noisy, "X_ERROR")]
    assert flips == [(2e-3, [0, 1]), (3e-3, [0, 1])]
    # The measurement flip comes before the measurement
    names = [inst.name for inst in noisy.flattened() if inst.name in ("X_ERROR", "M")]
    assert names[-2:] == ["X_ERROR", "M"]


def test_remote_gates_get_the_remote_error():
    backend = FakeBackend(4)
    backend.get_remote_gates = {(0, 3)}
    qc = __import__("qiskit").QuantumCircuit(4)
    qc.cx(0, 3)
    qc = __import__("qiskit").transpile(qc, coupling_map=CouplingMap([[0, 3], [3, 0], [0, 1], [1, 2], [2, 3]]),
                                        initial_layout=[0, 1, 2, 3], optimization_level=0)
    qt = QubitTracking(backend, qc)
    noisy = NoiseModel(tq=1e-3, remote=3e-2, qt=qt, backend=backend).noisy_circuit(stim.Circuit("CX 0 3 1 2"))
    errors = {tuple(t.value for t in inst.targets_copy()): inst.gate_args_copy()[0] for inst in channels(noisy, "DEPOLARIZE2")}
    assert errors == {(0, 3): 3e-2, (1, 2): 1e-3}


@pytest.mark.parametrize("code,decoder", [("surface", "mwpm"), ("surface", "bposd"), ("bb72", "bposd"),
                                          ("bb72", "relay_bp"), ("color", "chromobius")])
@pytest.mark.parametrize("error_type", ["sd6", "si1000", "constant"])
def test_zero_noise_gives_no_logical_errors(code, decoder, error_type):
    d = {"surface": 3, "bb72": 6, "color": 3}[code]
    task = Task(experiment="T", num_samples=200, backend="custom_full", backend_size=max(26, get_min_n(code, d)),
                code=code, distance=d, cycles=2, decoder=decoder, error_type=error_type, error_probability=1e-12)
    (row,) = logical_error.run(task)
    assert float(row["logical_error_rate"]) == 0.0


@pytest.mark.parametrize("window,expected", [("layer", 3), ("measurement", 1)])
def test_idle_window(window, expected):
    # Qubit 0 is reset, then waits through three layers: a gate, a measurement of qubit 1, another gate
    circuit = stim.Circuit("R 0\nTICK\nH 1\nTICK\nM 1\nTICK\nH 1")
    noisy = NoiseModel(idle=0.01, idle_window=window).noisy_circuit(circuit)
    idles = [inst for inst in noisy.flattened() if inst.name == "DEPOLARIZE1" and 0 in [t.value for t in inst.targets_copy()]]
    assert len(idles) == expected


def test_waiting_qubits_decay():
    # On a device with T1 and T2, a qubit that waits while another is measured decays for the
    # duration of the measurement, and with the coherence times scaled it decays less
    from eccentric.backends.utils import get_backend
    from eccentric.noise.flamingo_noise import FlamingoNoise
    from eccentric.noise.flamingo_noise_downscaled import FlamingoNoiseDownscaled

    backend = get_backend("real_loon_1_qpu", None)
    circuit = stim.Circuit("R 0 1\nTICK\nM 1")

    def waiting(model):
        noisy = model.noisy_circuit(circuit)
        layer = str(noisy).split("TICK")[1]
        channels = [inst for inst in stim.Circuit(layer).flattened() if inst.name == "PAULI_CHANNEL_1"]
        return sum(sum(i.gate_args_copy()) for i in channels if i.targets_copy()[0].value == 0)

    t1, t2 = backend.qubit_properties(0).t1, backend.qubit_properties(0).t2
    expected = 0.5 * (1 - math.exp(-1.56e-6 / t1)) + (1 - math.exp(-1.56e-6 / t2)) / 2 - (1 - math.exp(-1.56e-6 / t1)) / 4
    assert waiting(FlamingoNoise.get_noise(None, backend)) == pytest.approx(expected, rel=1e-4)
    assert 0 < waiting(FlamingoNoiseDownscaled.get_noise(None, backend)) < waiting(FlamingoNoise.get_noise(None, backend))


def test_waiting_in_a_layer_with_gates_in_sequence():
    # A layer can measure a qubit and then reset it; the others wait for both, 1.56 us + 1.29 us
    from eccentric.backends.utils import get_backend
    from eccentric.noise.flamingo_noise import FlamingoNoise

    backend = get_backend("real_loon_1_qpu", None)
    model = FlamingoNoise.get_noise(None, backend)
    noisy = model.noisy_circuit(stim.Circuit("R 0 1\nTICK\nM 1\nR 1"))
    layer = stim.Circuit(str(noisy).split("TICK")[1])
    waited = [i for i in layer.flattened() if i.name == "PAULI_CHANNEL_1" and i.targets_copy()[0].value == 0]
    t1, t2 = backend.qubit_properties(0).t1, backend.qubit_properties(0).t2
    t = 1.56e-6 + model.gate_times["R"]
    px = 0.25 * (1 - math.exp(-t / t1))
    assert len(waited) == 1 and waited[0].gate_args_copy()[0] == pytest.approx(px, rel=1e-4)
