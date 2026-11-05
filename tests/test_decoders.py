import numpy as np
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


class _Replay:
    """Stands in for stim's detector sampler: hands out fixed shots"""

    def __init__(self, dets, obs):
        self.dets, self.obs = dets, obs

    def sample(self, shots, separate_observables=True, bit_packed=False, **kwargs):
        d, o = self.dets[:shots], self.obs[:shots]
        if bit_packed:
            return np.packbits(d, axis=1, bitorder="little"), np.packbits(o, axis=1, bitorder="little")
        return d, o


def test_relay_bp_decodes_the_exact_syndrome(monkeypatch):
    # 36 detectors, not a multiple of 8: Relay-BP's sinter wrapper would pad each syndrome with 4 bits, and then
    # Relay-BP rarely converges. Our wrapper must give the same predictions as Relay-BP on the exact syndromes
    relay = pytest.importorskip("relay_bp")
    from relay_bp.stim.sinter.check_matrices import CheckMatrices
    circuit = stim.Circuit.generated("surface_code:unrotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.01,
                                     before_measure_flip_probability=0.01, after_reset_flip_probability=0.01)
    assert circuit.num_detectors % 8 != 0
    dets, obs = circuit.compile_detector_sampler(seed=3).sample(400, separate_observables=True)
    monkeypatch.setattr(stim.Circuit, "compile_detector_sampler", lambda self, *a, **k: _Replay(dets, obs))
    ler = decode("surface", circuit, 400, "relay_bp", "custom_full", "sd6")
    m = CheckMatrices.from_dem(circuit.detector_error_model(approximate_disjoint_errors=True))
    decoder = relay.RelayDecoderF64(m.check_matrix, error_priors=m.error_priors, gamma0=0.1, pre_iter=60, num_sets=60,
                                    set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)
    runner = relay.ObservableDecoderRunner(decoder, m.observables_matrix, include_decode_result=False)
    exact = runner.decode_observables_batch(dets.astype(np.uint8), parallel=False).astype(bool)
    assert ler == np.any(exact != obs, axis=1).mean()


def test_mwpm_matches_on_the_decomposed_error_model(monkeypatch):
    # CHANGELOG fix 5: PyMatching drops errors with more than two detectors, so MWPM must get them decomposed
    import pymatching
    seen = []
    original = pymatching.Matching.from_detector_error_model
    monkeypatch.setattr(pymatching.Matching, "from_detector_error_model",
                        staticmethod(lambda dem, *a, **k: seen.append(dem) or original(dem, *a, **k)))
    circuit = surface_circuit(0.005)
    raw = circuit.detector_error_model(approximate_disjoint_errors=True)
    detectors = lambda targets: sum(t.is_relative_detector_id() for t in targets)
    assert any(detectors(e.targets_copy()) > 2 for e in raw.flattened() if e.type == "error")
    decode("surface", circuit, 100, "mwpm", "custom_full", "sd6")
    for e in seen[0].flattened():
        if e.type == "error":
            pieces, current = [], []
            for t in e.targets_copy():
                if t.is_separator():
                    pieces.append(current)
                    current = []
                else:
                    current.append(t)
            assert all(detectors(p) <= 2 for p in pieces + [current])


def test_mwpm_fails_on_errors_that_do_not_decompose():
    circuit = stim.Circuit("""
        R 0 1 2
        X_ERROR(0.1) 0
        CX 0 1 0 2
        M 0 1 2
        DETECTOR rec[-1]
        DETECTOR rec[-2]
        DETECTOR rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """)
    assert decode("surface", circuit, 10, "mwpm", "custom_full", "sd6") is None
