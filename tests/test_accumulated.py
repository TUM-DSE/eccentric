import numpy as np
import pytest

from eccentric.codes import get_code
from eccentric.experiments.accumulated_error import _compiled, _noisy, with_idle_qubits
from eccentric.experiments.base import Task, load_backend


@pytest.mark.parametrize("p", [1e-3, 1e-2])
def test_idle_reference_idles_as_long_as_the_code(p):
    # The idle qubit is reset, idles through every layer of the code's circuit and is measured: under modified
    # SI1000 (reset error 2p, measurement error 5p, idling p/10 per layer) it flips with about 7p + 2/3 p/10 per
    # idle layer
    task = Task(experiment="accumulated", num_samples=1, backend="custom_full", backend_size=300, code="surface",
                distance=3, cycles=3, decoder="bposd", error_type="modsi1000", error_probability=p)
    backend = load_backend(task)
    qt, compiled = _compiled(task, backend, get_code("surface", 3, 3))
    idle = _noisy(task, backend, qt, with_idle_qubits(compiled, 1))
    assert idle.num_observables == 1 and idle.num_detectors == 0
    dem = idle.detector_error_model()
    flips = [e.args_copy()[0] for e in dem.flattened() if e.type == "error"]
    # Probability that an odd number of the independent mechanisms happen
    flip = (1 - np.prod([1 - 2 * q for q in flips])) / 2
    layers = sum(1 for inst in compiled.flattened() if inst.name == "TICK") + 1
    expected = 7 * p + (2 / 3) * (p / 10) * (layers - 2)
    assert 0.9 * expected < flip < 1.1 * expected
