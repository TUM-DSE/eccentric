from ..decoders import decode
from .base import load_backend, noisy_circuit, resolve_cycles, resolve_distance, result_row

REQUIRED = ("backends", "codes", "decoders", "error_types")


def run(task):
    backend = load_backend(task)
    d = resolve_distance(task, backend)
    cycles = resolve_cycles(task, d)
    circuit = noisy_circuit(task, backend, d, cycles)
    ler = decode(task.code, circuit, task.num_samples, task.decoder, task.backend, task.error_type)
    return [result_row(task, d, cycles, f"{ler:.3f}")]
