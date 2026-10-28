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


def test_layout_of_codes_that_fit_the_device():
    from eccentric.codes import get_code
    from eccentric.transpilers import run_transpiler

    def compile_code(code, d, backend, layout_method=None, size=None):
        device = get_backend(backend, size)
        qc = run_transpiler(get_code(code, d, 2).qc, device, layout_method, None)
        remote = {tuple(sorted(p)) for p in getattr(device, "get_remote_gates", [])}
        gates = [tuple(sorted(qc.find_bit(q).index for q in inst.qubits)) for inst in qc.data if len(inst.qubits) == 2]
        return qc.count_ops().get("swap", 0), sum(g in remote for g in gates)

    # The surface code is a subgraph of Willow's grid and needs no SWAPs, unless placed qubit by qubit
    assert compile_code("surface", 7, "real_willow") == (0, 0)
    assert compile_code("surface", 7, "real_willow", "trivial")[0] > 0
    # A code that fits one QPU stays off the remote couplers between QPUs
    assert compile_code("surface", 7, "real_flamingo")[1] == 0
    # The rotated surface code fits a grid in the shape of a diamond
    assert get_backend("custom_diamond", 85).num_qubits == 85
    assert compile_code("surface", 5, "custom_diamond", size=85)[0] == 0


def test_bicycle_module():
    from collections import Counter
    from eccentric.codes import get_code
    from eccentric.transpilers import run_transpiler

    for backend, noisy in [("real_bicycle", False), ("real_bicycle_lr", True), ("real_bicycle_2_qpus", False)]:
        device = get_backend(backend, None)
        # Six couplers per qubit (both directions), plus the links between modules
        degrees = Counter(q for e in device.coupling_map.get_edges() for q in e)
        assert Counter(degrees.values())[12] == device.num_qubits - 2 * len(device.LINKS) * (device.num_qubits // 288 - 1)
        assert len(device.get_remote_gates) == noisy * 288 + len(device.LINKS) * (device.num_qubits // 288 - 1)
    # The gross code is placed without SWAPs, with two gates per qubit and cycle on the c-couplers
    device = get_backend("real_bicycle_lr", None)
    qc = run_transpiler(get_code("gross", 12, 2).qc, device, device.layout_method, None)
    remote = {tuple(sorted(p)) for p in device.get_remote_gates}
    gates = [tuple(sorted(qc.find_bit(q).index for q in inst.qubits)) for inst in qc.data if len(inst.qubits) == 2]
    assert qc.count_ops().get("swap", 0) == 0
    assert sum(g in remote for g in gates) == 2 * 288
