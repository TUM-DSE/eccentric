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


def test_single_and_projected_ibm_qpus():
    assert get_backend("real_loon_1_qpu", None).num_qubits == 121
    assert get_backend("real_flamingo_1_qpu", None).num_qubits == 133
    assert get_backend("real_loon_1_qpu", None).get_remote_gates == []
    projected = get_backend("real_flamingo_5_qpus", None)
    assert projected.num_qubits == 5 * 133 and len(projected.get_remote_gates) == 4 * 4


@pytest.mark.parametrize("schedule", ["ibm", "zxcoloration"])
def test_native_layout_of_bb_code(schedule):
    from collections import Counter
    from eccentric.experiments.base import load_backend, noisy_circuit

    for backend, remote_per_qubit in [("native", set()), ("native_lr", {2})]:
        task = Task(experiment="T", num_samples=1, backend=backend, code="bb72", distance=6, cycles=2,
                    error_type="real_flamingo", schedule=schedule)
        device = load_backend(task)
        # Six couplers per qubit, two of them long-range in IBM's toric layout, and no SWAPs
        assert set(Counter(q for e in device.coupling_map.get_edges() for q in e).values()) == {12}
        assert set(Counter(q for e in device.get_remote_gates for q in e).values()) == remote_per_qubit
        circuit = noisy_circuit(task, device, 6, 2)
        assert not any(inst.name == "SWAP" for inst in circuit.flattened())
