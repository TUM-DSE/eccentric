from ..codes import make_color_code_circuit
from ..decoders import decode
from ..noise import get_noise_model
from .base import result_row

REQUIRED = ("backends", "codes", "decoders", "error_types", "distances")


def run(task):
    d = task.distance
    phenomenological = task.error_type == "phenomenological"
    circuit = make_color_code_circuit(
        obs_basis="Z",
        base_data_width=d,
        rounds=d,
        noise_strength=task.error_probability if phenomenological else 0,
    )
    if not phenomenological:
        noise_model = get_noise_model(task.error_type, None, task.error_probability, None)
        circuit = noise_model.noisy_circuit(circuit)
    ler = decode(task.code, circuit, task.num_samples, task.decoder, task.backend, task.error_type)
    return [result_row(task, d, d, f"{ler:.3f}")]
