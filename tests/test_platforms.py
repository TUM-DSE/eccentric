import pytest

from eccentric.backends import get_backend
from eccentric.experiments import Task, logical_error


def test_harvard_backend_size_and_shuttling():
    backend = get_backend("real_harvard", None)
    assert backend.num_qubits == 448
    # Grid neighbours interact in place, every other pair through a move
    grid_edges = 16 * 27 + 28 * 15
    assert len(backend.get_remote_gates) == 448 * 447 // 2 - grid_edges
    assert (0, 1) not in backend.get_remote_gates and (0, 447) in backend.get_remote_gates


@pytest.mark.parametrize("backend,error_type", [
    ("real_willow", "real_willow_spec"),
    ("real_harvard", "real_harvard"),
])
def test_platform_end_to_end(backend, error_type):
    task = Task(
        experiment="T",
        num_samples=200,
        backend=backend,
        code="surface",
        distance=3,
        decoder="mwpm",
        error_type=error_type,
    )
    (row,) = logical_error.run(task)
    assert 0.0 <= float(row["logical_error_rate"]) < 0.5
