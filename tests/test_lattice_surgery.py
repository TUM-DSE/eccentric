import pytest
import stim

pytest.importorskip("tqec")

from eccentric.codes import get_code  # noqa: E402
from eccentric.codes.catalog import CODES, num_rounds  # noqa: E402
from eccentric.codes.lattice_surgery import LAYERS, NUM_QUBITS  # noqa: E402
from eccentric.decoders import decode  # noqa: E402
from eccentric.experiments.base import Task, load_backend, noisy_circuit  # noqa: E402

OBSERVABLES = {"ls_memory": 1, "ls_merge": 1, "ls_cnot": 2}


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("operation", sorted(LAYERS))
def test_noiseless_operations_are_deterministic(operation, basis):
    circuit = get_code(operation, 3, 3, basis=basis).stim_circuit
    assert circuit.num_qubits == NUM_QUBITS[operation](3)
    assert circuit.num_observables == OBSERVABLES[operation]
    # d^2 - 1 stabilizers per patch and round: the operations run d rounds per block
    assert num_rounds(operation, 3) == 3 * LAYERS[operation]
    dets, obs = circuit.compile_detector_sampler().sample(64, separate_observables=True)
    assert not dets.any() and not obs.any()


def test_rounds_are_fixed_per_block():
    with pytest.raises(ValueError):
        get_code("ls_merge", 3, 5)
    with pytest.raises(ValueError):
        get_code("ls_memory", 4, 4)
    assert CODES["ls_cnot"].num_qubits(5) == get_code("ls_cnot", 5, 5).stim_circuit.num_qubits


def sd6(operation, backend, size, basis="Z", p=1e-3):
    t = Task(experiment="test", num_samples=1, backend=backend, backend_size=size, code=operation, distance=3,
             cycles=3, error_type="sd6", error_probability=p, basis=basis)
    return noisy_circuit(t, load_backend(t), 3, 3)


@pytest.mark.parametrize("operation", sorted(LAYERS))
def test_operations_keep_their_distance_through_the_pipeline(operation):
    # Compiled onto an all-to-all device with SD6 noise: the circuit distance stays the patches' distance and
    # matching corrects most errors
    circuit = sd6(operation, "custom_full", NUM_QUBITS[operation](3))
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    assert len(dem.shortest_graphlike_error()) == 3
    ler = decode(operation, circuit, 4000, "mwpm", "custom_full", "sd6")
    raw = circuit.compile_detector_sampler().sample(4000, separate_observables=True)[1].any(axis=1).mean()
    assert ler < 0.05 and ler < raw / 3


def test_patches_are_placed_on_a_grid_without_swaps():
    # tqec's layout is a square grid with nearest-neighbour gates, so a grid device holds it exactly
    circuit = sd6("ls_memory", "custom_grid", 49)
    assert not any(inst.name == "SWAP" for inst in circuit.flattened())

