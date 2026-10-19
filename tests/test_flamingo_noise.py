import stim
import pytest
from noise.flamingo_noise import FlamingoNoise, flamingo_err_prob


class DummyQubitTracking:
    def get_layout_postion(self, q):
        return q

    def swap_qubits(self, q1, q2):
        pass

class QubitProperties:
    def __init__(self, t1=None, t2=None, frequency=None):
        self.t1 = t1
        self.t2 = t2
        self.frequency = frequency


class DummyBackend:
    @property
    def get_remote_gates(self):
        return {(2, 3)}

    def qubit_properties(self, qubit):
        return QubitProperties(t1=0.1, t2=0.1, frequency=None)

@pytest.fixture
def noise_model():
    flamingo_err_prob.update({
        "P_CZ": 0.01,
        "P_CZ_CROSSTALK": 0.0,
        "P_CZ_LEAKAGE": 0.0,
        "P_IDLE": 0.0,
        "P_READOUT": 0.0,
        "P_RESET": 0.0,
        "P_SQ": 0.0,
        "P_LEAKAGE": 0.0
    })

    qt = DummyQubitTracking()
    backend = DummyBackend()
    return FlamingoNoise.get_noise(qt, backend)

def test_noisy_op_distinguishes_remote_gates(noise_model):
    op = stim.CircuitInstruction("CX", [
        stim.GateTarget(0), stim.GateTarget(1),
        stim.GateTarget(2), stim.GateTarget(3),
        stim.GateTarget(4), stim.GateTarget(5),
    ])

    pre, mid, post = noise_model.noisy_op(op, base_p=0.01, ancilla=10)

    post_str = str(post)
    mid_str = str(mid)
