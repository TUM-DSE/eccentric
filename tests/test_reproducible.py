"""A configuration compiles to the same circuit every time: seeded routing, noise choices and T1/T2 draws"""
import numpy as np
import pytest

from eccentric.backends.utils import get_backend
from eccentric.experiments import Task
from eccentric.experiments.base import load_backend, noisy_circuit
from test_noise_scale import assert_scaled


def build(backend, error_type, code="surface", d=3, cycles=3):
    task = Task(experiment="test", num_samples=1, backend=backend, code=code, distance=d, cycles=cycles, error_type=error_type)
    np.random.seed(None)  # whatever state the global generators are in must not matter
    return noisy_circuit(task, load_backend(task), d, cycles)


@pytest.mark.parametrize("backend, error_type", [("real_flamingo_1_qpu", "real_flamingo"), ("real_loon", "real_flamingo"),
                                                 ("real_willow", "real_willow")])
def test_same_configuration_same_circuit(backend, error_type):
    assert build(backend, error_type) == build(backend, error_type)


@pytest.mark.parametrize("name", ["real_flamingo_1_qpu", "real_loon", "real_bicycle"])
def test_one_calibration_per_device(name):
    t = lambda: [(p.t1, p.t2) for p in get_backend(name, None).target.qubit_properties]
    assert t() == t()


def test_noise_scales_share_the_circuit():
    # Built separately, the device noise and a quarter of it give the same routed circuit
    assert_scaled(build("real_flamingo_1_qpu", "real_flamingo"), build("real_flamingo_1_qpu", "real_flamingo_x0.25"), 0.25)
