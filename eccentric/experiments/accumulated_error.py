from qiskit_qec.utils import get_stim_circuits

from ..backends import QubitTracking
from ..codes import get_code, make_idle_qubit_circuit
from ..decoders import decode, raw_error_rate
from ..noise import get_noise_model
from ..transpilers import run_transpiler
from .base import base_row, load_backend, resolve_cycles, resolve_distance

REQUIRED = ("backends", "codes", "decoders", "error_types")


def _noisy(task, backend, code):
    detectors, logicals = code.stim_detectors()
    code.qc = run_transpiler(code.qc, backend, task.layout_method, task.routing_method)
    qt = QubitTracking(backend, code.qc)
    circuit = get_stim_circuits(code.qc, detectors=detectors, logicals=logicals)[0][0]
    return get_noise_model(task.error_type, qt, task.error_probability, backend).noisy_circuit(circuit)


def run(task):
    backend = load_backend(task)
    d = resolve_distance(task, backend)
    cycles = resolve_cycles(task, d)
    logical_qubits = 12 if task.code == "gross" else 1
    circuit = _noisy(task, backend, get_code(task.code, d, cycles))
    idle_circuit = _noisy(task, backend, make_idle_qubit_circuit(cycles, logical_qubits))
    raw = raw_error_rate(circuit, task.num_samples) / task.num_samples
    idle = raw_error_rate(idle_circuit, task.num_samples) / task.num_samples
    corrected = decode(task.code, circuit, task.num_samples, task.decoder, task.backend, task.error_type)
    return [{
        **base_row(task, d, cycles),
        "raw_error_rate": f"{raw:.5f}",
        "idle_error_rate": f"{idle:.5f}",
        "corrected_error_rate": f"{corrected:.5f}",
    }]
