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


def test_belief_find_with_an_undetectable_logical_error():
    # Qubit 0's flip changes the logical and no detector (as a SWAP fault of a routed circuit can);
    # qubit 1's flip trips the detector and is corrected
    circuit = stim.Circuit("""
        R 0 1
        X_ERROR(0.2) 0 1
        M 0 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """)
    ler = decode("x", circuit, 2000, "belief_find", "custom_full", "sd6")
    assert ler is not None and 0.15 < ler < 0.25


def test_bposd_builds_one_decoder_per_circuit():
    from eccentric.decoders import bp_osd

    circuit, other = surface_circuit(), surface_circuit(0.005)
    bp_osd.bposd_batch(circuit, 10, False)
    first = bp_osd._last["value"][3]
    assert 0.0 <= bp_osd.bposd_batch(circuit, 200, False) < 0.1
    assert bp_osd._last["value"][3] is first
    bp_osd.bposd_batch(other, 10, False)
    assert bp_osd._last["value"][3] is not first
    # The checked variant has its own matrices (no xor priors) and OSD order
    bp_osd.bposd_chk(other, 10, False)
    assert bp_osd._last["key"] == (False, False, 7)
