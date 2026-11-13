import pytest
import stim

pytest.importorskip("qldpc.experimental.surgery")

from eccentric.codes import get_code, get_min_n, num_rounds  # noqa: E402
from eccentric.codes.operations import BASIS_OF, OPERATIONS, css_code, make_operation_circuit  # noqa: E402
from eccentric.decoders import decode  # noqa: E402
from eccentric.experiments.base import Task, load_backend, noisy_circuit  # noqa: E402

SMALL = [("steane", 3), ("hgp_hamming", 3), ("surface", 3), ("cbb30", 6)]


def sd6(circuit, p=1e-3):
    out = stim.Circuit()
    for inst in circuit.flattened():
        n = inst.name
        if n in ("R", "RX"):
            out.append(inst)
            out.append("X_ERROR" if n == "R" else "Z_ERROR", inst.targets_copy(), p)
        elif n in ("M", "MX"):
            out.append("X_ERROR" if n == "M" else "Z_ERROR", inst.targets_copy(), p)
            out.append(inst)
        elif n in ("CX", "CZ"):
            out.append(inst)
            out.append("DEPOLARIZE2", inst.targets_copy(), p)
        else:
            out.append(inst)
    return out


@pytest.mark.parametrize("code,d", SMALL)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_operations_are_deterministic_without_noise(code, d, operation):
    basis = BASIS_OF.get(operation, "Z")
    circuit = get_code(code, d, d, basis=basis, operation=operation).stim_circuit
    assert circuit.num_qubits == get_min_n(code, d, operation)
    assert circuit.num_observables >= 1
    dets, obs = circuit.compile_detector_sampler().sample(64, separate_observables=True)
    assert not dets.any() and not obs.any()
    assert num_rounds(code, d, operation) == (2 * d if operation == "transversal_cnot" else d)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_transversal_cnot_entangles_the_blocks(basis):
    # Without the CNOT's detector correction, the round after the gate would not be deterministic
    code = css_code("steane", 3)
    from eccentric.codes import operations
    circuit = operations._transversal_cnot(code, 2, basis)
    assert circuit.num_observables == 2
    # A single data error on the control before the gate reaches the target: the error model has mechanisms
    # flipping both blocks' observables
    dem = sd6(circuit).detector_error_model(approximate_disjoint_errors=True)
    both = [e for e in dem.flattened() if e.type == "error"
            and sum(t.is_logical_observable_id() for t in e.targets_copy()) == 2]
    assert both


def circuit_distance(circuit):
    return len(sd6(circuit).search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=3, dont_explore_edges_with_degree_above=3,
        dont_explore_edges_increasing_symptom_degree=False))


@pytest.mark.parametrize("operation", ["memory_ec", "measure_z", "measure_x", "transversal_cnot"])
def test_operations_keep_the_distance_of_the_surface_code(operation):
    assert circuit_distance(make_operation_circuit("surface", 3, 3, BASIS_OF.get(operation, "Z"), operation)) == 3


def test_steane_loses_distance_to_hook_errors_of_the_edge_coloring_schedule():
    # qLDPC's syndrome extraction of the Steane code has circuit distance 2 (two-qubit gate faults in the Z-check
    # layers), with or without an operation; the catalog's own Steane circuit keeps 3
    assert circuit_distance(make_operation_circuit("steane", 3, 3, "Z", "memory_ec")) == 2
    assert circuit_distance(make_operation_circuit("steane", 3, 3, "Z", "measure_z")) == 2
    assert circuit_distance(make_operation_circuit("steane", 3, 3, "X", "measure_x")) == 3


def test_operations_refuse_non_css_codes_and_wrong_bases():
    with pytest.raises(ValueError):
        css_code("bacon", 3)
    with pytest.raises(ValueError):
        css_code("planar_honeycomb", 3)
    with pytest.raises(ValueError):
        make_operation_circuit("steane", 3, 3, "X", "measure_z")
    with pytest.raises(ValueError):
        make_operation_circuit("steane", 3, 3, "Z", "no_such_operation")


def test_operation_through_the_pipeline_decodes():
    t = Task(experiment="test", num_samples=1, backend="custom_full", backend_size=get_min_n("hgp_hamming", 3, "measure_z"),
             code="hgp_hamming", distance=3, cycles=3, error_type="sd6", error_probability=1e-3, operation="measure_z")
    circuit = noisy_circuit(t, load_backend(t), 3, 3)
    ler = decode("hgp_hamming", circuit, 3000, "bposd", "custom_full", "sd6")
    raw = circuit.compile_detector_sampler().sample(3000, separate_observables=True)[1].any(axis=1).mean()
    assert ler < 0.05 and ler < raw


def test_runner_sweeps_operations():
    from eccentric.runner import expand
    _, tasks = expand({"name": "O", "num_samples": 10, "backends": ["custom_full"], "codes": ["steane"], "distances": [3],
                       "decoders": ["bposd"], "error_types": ["sd6"], "error_probabilities": [1e-3],
                       "operations": ["memory_ec", "transversal_cnot"]})
    assert {t.operation for t in tasks} == {"memory_ec", "transversal_cnot"}
    assert {t.backend_size for t in tasks} == {get_min_n("steane", 3, "memory_ec"), get_min_n("steane", 3, "transversal_cnot")}
