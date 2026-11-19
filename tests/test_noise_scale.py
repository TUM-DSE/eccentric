"""Device noise scaled by a factor: <error_type>_x<factor> multiplies every error probability"""
import copy
import random

import pytest
import stim

import eccentric.experiments.base as base
from eccentric.experiments import Task
from eccentric.noise.noise import PROBABILITY_ARGS, ScaledNoise, scale_noise
from eccentric.noise.utils import get_noise_model
from eccentric.noise.willow_noise import WillowNoise

CIRCUIT = """
R 0 1 2
X_ERROR(0.01) 0
REPEAT 3 {
    DEPOLARIZE2(0.002) 0 1
    PAULI_CHANNEL_1(0.001, 0.002, 0.003) 2
    E(0.004) X0 Z1
    M(0.02) 0
    DETECTOR(1, 2, 0) rec[-1]
    TICK
}
MPP(0.008) X0*X1
HERALDED_ERASE(0.01) 2
Y_ERROR(1) 1
M 1 2
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def assert_scaled(original: stim.Circuit, scaled: stim.Circuit, factor: float):
    a, b = original.flattened(), scaled.flattened()
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.name, x.targets_copy(), x.tag) == (y.name, y.targets_copy(), y.tag)
        args = x.gate_args_copy()
        expected = [p * factor for p in args] if x.name in PROBABILITY_ARGS and args != [1.0] else args
        assert y.gate_args_copy() == pytest.approx(expected, rel=1e-12, abs=0)


def test_scale_noise_multiplies_every_error_probability():
    circuit = stim.Circuit(CIRCUIT)
    circuit.append(stim.CircuitInstruction("Z_ERROR", [stim.GateTarget(2)], [0.03], tag="kept"))
    scaled = scale_noise(circuit, 0.25)
    assert_scaled(circuit, scaled, 0.25)
    assert scaled.flattened()[3].gate_args_copy() == [0.001 * 0.25, 0.002 * 0.25, 0.003 * 0.25]
    assert any(isinstance(op, stim.CircuitRepeatBlock) and op.repeat_count == 3 for op in scaled)
    assert scale_noise(circuit, 1.0) == circuit
    # Coordinates, observable indices and leakage's probability-1 errors are not probabilities to scale
    assert str(scaled).count("DETECTOR(1, 2, 0)") == 1 and "Y_ERROR(1) 1" in str(scaled)


def test_error_type_suffix():
    model = get_noise_model("real_willow_x0.25", None)
    assert isinstance(model, ScaledNoise) and model.factor == 0.25
    assert get_noise_model("real_willow_x1", None).factor == 1.0
    assert not isinstance(get_noise_model("real_willow", None), ScaledNoise)
    assert not isinstance(get_noise_model("real_willow_spec", None), ScaledNoise)


def test_leakage_rate_is_scaled():
    base_model = WillowNoise.get_noise(None)
    scaled = ScaledNoise(WillowNoise.get_noise(None), 0.1)
    assert scaled.leakage == pytest.approx(base_model.leakage * 0.1)
    assert scaled.leakage_propagation == pytest.approx(base_model.leakage_propagation * 0.1)


@pytest.mark.parametrize("backend, error_type", [("real_flamingo_1_qpu", "real_flamingo"), ("real_torino", "real_heron"),
                                                 ("real_harvard", "real_harvard"), ("real_infleqtion", "real_infleqtion"),
                                                 ("real_willow", "real_willow")])
@pytest.mark.parametrize("factor", [0.1, 0.25, 1.0])
def test_scaled_device_noise_on_compiled_circuits(monkeypatch, backend, error_type, factor):
    """The sweep's own pipeline: the same routed circuit with the device noise and with it scaled"""
    task = Task(experiment="test", num_samples=1, backend=backend, code="surface", distance=3, cycles=3,
                error_type=error_type)
    built = {}
    real = base.get_noise_model

    class Both:
        def __init__(self, error_type, qt, p, device):
            self.base = real(error_type, qt, p, device)
            self.scaled = real(f"{error_type}_x{factor}", copy.deepcopy(qt), p, device)

        def noisy_circuit(self, circuit):
            random.seed(7)
            built["base"] = self.base.noisy_circuit(circuit)
            random.seed(7)
            built["scaled"] = self.scaled.noisy_circuit(circuit)
            return built["base"]

    monkeypatch.setattr(base, "get_noise_model", Both)
    base.noisy_circuit(task, base.load_backend(task), 3, 3)
    leaked = any(i.gate_args_copy() == [1.0] for c in built.values() for i in c.flattened())
    if leaked and factor != 1.0:
        pytest.skip("a qubit leaked in one of the circuits, so their random draws differ")
    assert_scaled(built["base"], built["scaled"], factor)
    mass = lambda c: sum(e.args_copy()[0] for e in c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True).flattened() if e.type == "error")
    assert mass(built["scaled"]) == pytest.approx(factor * mass(built["base"]), rel=0.1)
