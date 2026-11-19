import stim
from qiskit_qec.utils import get_stim_circuits

from ..backends import QubitTracking
from ..codes import get_code
from ..decoders import decode, raw_error_rate
from ..noise import get_noise_model
from ..transpilers import run_transpiler
from .base import base_row, load_backend, resolve_code_schedule, resolve_cycles, resolve_distance

REQUIRED = ("backends", "codes", "decoders", "error_types")

# Where a code's measurement results are not annotated (detectors, observables), only its gates matter
ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}


def _compiled(task, backend, code):
    """The code's circuit compiled onto the device, before noise, and its qubit tracking"""
    detectors, logicals = code.stim_detectors()
    code.qc = run_transpiler(code.qc, backend, task.layout_method, task.routing_method)
    qt = QubitTracking(backend, code.qc)
    return qt, get_stim_circuits(code.qc, detectors=detectors, logicals=logicals)[0][0]


def _noisy(task, backend, qt, circuit):
    return get_noise_model(task.error_type, qt, task.error_probability, backend).noisy_circuit(circuit)


def with_idle_qubits(circuit: stim.Circuit, n: int) -> stim.Circuit:
    """The idle reference: n extra qubits that are reset with the code's first layer, idle through every
    later layer of the code's circuit, and are measured with its last layer, each its own observable.
    The noise model then gives them exactly the idling (per layer, or for the time the layers take) of a
    qubit left alone for as long as the code runs. The code's own detectors and observables are dropped."""
    first = circuit.num_qubits
    spare = list(range(first, first + n))
    out = stim.Circuit()
    out.append("R", spare)
    for inst in circuit.flattened():
        if inst.name not in ANNOTATIONS:
            out.append(inst)
    out.append("M", spare)
    for i in range(n):
        out.append("OBSERVABLE_INCLUDE", [stim.target_rec(i - n)], i)
    return out


def run(task):
    backend = load_backend(task)
    d = resolve_distance(task, backend)
    cycles = resolve_cycles(task, d)
    logical_qubits = 12 if task.code == "gross" else 1
    qt, compiled = _compiled(task, backend, get_code(task.code, d, cycles, resolve_code_schedule(task)))
    circuit = _noisy(task, backend, qt, compiled)
    # As many idle qubits as the code has logical qubits; the idle error is that any of them flips
    idle_circuit = _noisy(task, backend, qt, with_idle_qubits(compiled, logical_qubits))
    raw = raw_error_rate(circuit, task.num_samples) / task.num_samples
    idle = raw_error_rate(idle_circuit, task.num_samples) / task.num_samples
    corrected = decode(task.code, circuit, task.num_samples, task.decoder, task.backend, task.error_type)
    return [{
        **base_row(task, d, cycles),
        "raw_error_rate": f"{raw:.5f}",
        "idle_error_rate": f"{idle:.5f}",
        "corrected_error_rate": f"{corrected:.5f}",
    }]
