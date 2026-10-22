import pytest
import stim

from eccentric.decoders import decode
from eccentric.experiments import Task, logical_error


def surface_circuit(p=0.003):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3,
        after_clifford_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p,
    )


@pytest.mark.parametrize("decoder", ["bplsd", "union_find", "belief_find", "belief_matching", "relay_bp", "tesseract"])
def test_decoder_on_surface_code(decoder):
    ler = decode("surface", surface_circuit(), 300, decoder, "custom_full", "sd6")
    assert ler is not None and 0.0 <= ler < 0.1


@pytest.mark.parametrize("decoder", ["relay_bp", "bplsd"])
def test_qldpc_decoder_end_to_end(decoder):
    task = Task(
        experiment="T",
        num_samples=100,
        backend="custom_full",
        backend_size=144,
        code="bb72",
        distance=6,
        cycles=2,
        decoder=decoder,
        error_type="sd6",
        error_probability=0.001,
    )
    (row,) = logical_error.run(task)
    assert 0.0 <= float(row["logical_error_rate"]) < 0.5
