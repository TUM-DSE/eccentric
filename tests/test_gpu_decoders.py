"""GPU decoders (eccentric/decoders/gpu, analysis/GPU.md) against the CPU ones on the same shots. Skipped without CuPy and
an NVIDIA GPU; bposd_gpu and relay_bp_gpu also need CUDA-Q QEC (requirements-gpu.txt)"""
import numpy as np
import pytest
import stim

cp = pytest.importorskip("cupy")
try:
    from eccentric.decoders.gpu import runtime
    runtime.cupy().cuda.runtime.getDeviceCount()
except Exception:  # no driver or no device
    pytest.skip("no CUDA device", allow_module_level=True)

from eccentric.decoders import decode
from eccentric.decoders.gpu import GPU_DECODERS


def _surface(d=5, p=0.006):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


def _color(d=5, p=0.004):
    return stim.Circuit.generated("color_code:memory_xyz", distance=d, rounds=d, after_clifford_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


def _shots(circuit, n, seed=1):
    return circuit.compile_detector_sampler(seed=seed).sample(n, separate_observables=True)


@pytest.mark.parametrize("method,scaling", [("minimum_sum", 1.0), ("minimum_sum", 0.625)])
def test_bp_matches_ldpc_bit_for_bit(method, scaling):
    from ldpc import BpDecoder
    from eccentric.decoders.bp_osd import dem_to_check_matrices
    from eccentric.decoders.gpu.bp import GpuBp
    for circuit in (_surface(), _color()):
        chk, _, priors = dem_to_check_matrices(circuit.detector_error_model(), xor_priors=True)
        dets, _ = _shots(circuit, 300)
        cpu = BpDecoder(chk, error_channel=list(priors), max_iter=60, bp_method=method, ms_scaling_factor=scaling)
        decoding, llrs, converged, iterations = (x.get() for x in GpuBp(chk, priors, max_iter=60, bp_method=method,
                                                                         ms_scaling_factor=scaling).decode(dets))
        for i in np.flatnonzero(dets.any(axis=1)):
            assert np.array_equal(cpu.decode(dets[i].astype(np.uint8)), decoding[i])
            assert np.array_equal(np.asarray(cpu.log_prob_ratios).view(np.int64), llrs[i].view(np.int64))
            assert bool(cpu.converge) == bool(converged[i]) and cpu.iter == iterations[i]


def test_bplsd_and_belief_find_match_ldpc_shot_by_shot():
    from ldpc import BeliefFindDecoder, BpLsdDecoder
    from eccentric.decoders.ldpc_decoders import _check_matrices
    from eccentric.decoders.gpu.ldpc_gpu import belief_find_decoder, bplsd_decoder
    circuit = _surface(5, 0.008)
    dets, _ = _shots(circuit, 400)
    chk, obs, priors = _check_matrices(circuit, False)
    lsd = BpLsdDecoder(chk, error_channel=list(priors), max_iter=10000, bp_method="minimum_sum", ms_scaling_factor=1.0,
                       lsd_method="lsd_cs", lsd_order=min(7, chk.shape[1] - chk.shape[0]))
    gpu = bplsd_decoder(circuit, False)
    predicted = gpu.predict(dets)
    assert gpu.stats["postprocessed"] > 20  # the LSD stage runs
    expected = np.array([(obs @ lsd.decode(s)) % 2 for s in dets.astype(np.uint8)]).astype(bool)
    assert np.array_equal(predicted, expected)
    seen = np.asarray(chk.sum(axis=0)).ravel() > 0
    bf = BeliefFindDecoder(chk[:, seen], error_channel=list(np.asarray(priors)[seen]), max_iter=10000,
                           bp_method="minimum_sum", ms_scaling_factor=1.0, uf_method="inversion")
    expected = np.array([(obs[:, seen] @ bf.decode(s)) % 2 for s in dets.astype(np.uint8)]).astype(bool)
    assert np.array_equal(belief_find_decoder(circuit, False).predict(dets), expected)


def test_gari_matches_the_c_decoder_shot_by_shot():
    from eccentric.decoders.gari.gari import decode_observables, gari_encoding
    from eccentric.decoders.gpu.gari_gpu import GariGpu
    circuit = _surface(5, 0.004)
    enc = gari_encoding(circuit)
    dets, _ = _shots(circuit, 3000)
    assert np.array_equal(GariGpu(enc).predict(dets), decode_observables(enc, dets))


def test_union_find_is_close_to_ldpc_on_surface_code():
    # Not a copy of ldpc: equal weights are ordered differently, which shifts the logical error rate by 0.1-2% on
    # some codes (analysis/GPU.md); on the surface code the two agree within statistics
    from ldpc import UnionFindDecoder
    from eccentric.decoders.ldpc_decoders import _check_matrices
    from eccentric.decoders.gpu.union_find_gpu import union_find_decoder
    circuit = _surface(5, 0.004)
    dets, actual = _shots(circuit, 4000)
    chk, obs, priors = _check_matrices(circuit, False)
    uf = UnionFindDecoder(chk, uf_method="inversion")
    pr = np.clip(priors, 1e-12, 1 - 1e-12)
    llrs = np.log((1 - pr) / pr)
    cpu = np.array([(obs @ uf.decode(s, llrs=llrs)) % 2 for s in dets.astype(np.uint8)]).astype(bool)
    gpu = union_find_decoder(circuit, False).predict(dets)
    assert np.all(gpu == cpu, axis=1).mean() > 0.97
    e_gpu, e_cpu = np.any(gpu != actual, axis=1), np.any(cpu != actual, axis=1)
    b, c = np.sum(e_gpu & ~e_cpu), np.sum(e_cpu & ~e_gpu)
    assert abs(b - c) <= 4 * np.sqrt(b + c + 1)


def test_hierarchical_previous_mode_follows_the_cpu_decoder():
    from eccentric.decoders.gpu.ldpc_gpu import HierarchicalGpu
    from eccentric.decoders.hierarchical import Hierarchical
    # A toy two-level structure on a surface code's detectors: the CPU decoder's level stages skip zero syndromes
    circuit = _surface(3, 0.004)
    levels = np.where(np.arange(circuit.num_detectors) % 2 == 0, 1, 2)
    dem = circuit.detector_error_model()
    dets, _ = _shots(circuit, 600)
    cpu = Hierarchical(dem, levels, soft=True)
    expected = np.array([(cpu.obs @ cpu.decode(s)) % 2 for s in dets.astype(np.uint8)]).astype(bool)
    predicted = HierarchicalGpu(dem, levels, soft=True).predict(dets)
    # product-sum BP rounds differently on the GPU: decisions agree except for rare ties
    assert np.all(predicted == expected, axis=1).mean() > 0.99


@pytest.mark.parametrize("decoder", sorted(GPU_DECODERS))
def test_decode_dispatches_gpu_decoders(decoder):
    if decoder in ("bposd_gpu", "bposd_tuned_gpu", "relay_bp_gpu"):
        pytest.importorskip("cudaq_qec")
    if decoder.startswith("hier"):
        pytest.skip("needs a concatenated code (tests/test_new_decoders.py builds them; slow)")
    circuit = _surface(3, 0.003)
    ler = decode("surface", circuit, 4000, decoder, "custom_full", "sd6")
    assert ler is not None and 0 <= ler < 0.1
