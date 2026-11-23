"""BP-OSD on the GPU through NVIDIA CUDA-Q QEC's nv-qldpc-decoder (pip install cudaq-qec), with the settings of
ECCentric's CPU decoders (bposd_gpu, bposd_tuned_gpu). nv-qldpc is closed source; its BP and OSD are NVIDIA's own
implementations, so the corrections differ from ldpc's on a few shots and the logical error rates agree statistically
(analysis/GPU.md). relay_decoder here (nv-qldpc's Relay-BP) is not a decoder name any more: it uses one set of memory
strengths for all shots, so its logical error rate depends on the seed; relay_bp_gpu is relay_gpu.py's port."""
import numpy as np
import stim

from ..bp_osd import BP, dem_to_check_matrices
from . import runtime
from .ldpc_gpu import _cached, logical_error_rate

# Shots per decode_batch call (nv-qldpc decodes a call's shots in parallel, bp_batch_size at a time)
CHUNK = 1 << 15


def _qec():
    runtime.cupy()  # NVRTC and the driver, as for the other GPU decoders
    try:
        import cudaq_qec
    except ImportError as e:
        raise RuntimeError("bposd_gpu and bposd_tuned_gpu need CUDA-Q QEC (pip install cudaq-qec)") from e
    return cudaq_qec


class NvQldpc:
    """nv-qldpc-decoder on a check matrix, predicting observable flips (observables @ errors + bias)"""

    def __init__(self, chk, obs, priors, syndrome_bias=None, observables_bias=None, **options):
        qec = _qec()
        self.chk = chk.tocsr()
        self.obs = obs.tocsr()
        self.syndrome_bias = None if syndrome_bias is None else np.asarray(syndrome_bias, dtype=np.uint8)
        self.observables_bias = None if observables_bias is None else np.asarray(observables_bias, dtype=np.uint8)
        dense = np.ascontiguousarray(self.chk.toarray().astype(np.uint8))
        self.chunk = int(max(1, min(CHUNK, (runtime.MEMORY_BUDGET // 2) // max(1, 32 * self.chk.nnz))))
        # options (e.g. an nv-qldpc preset) override these defaults
        options = {"use_sparsity": True, "proc_float": "fp64", "bp_batch_size": self.chunk, **options}
        self.decoder = qec.get_decoder("nv-qldpc-decoder", dense, error_rate_vec=[float(p) for p in priors], **options)
        self.converged = 0
        self.shots = 0

    def predict(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        if self.syndrome_bias is not None:
            dets = dets ^ self.syndrome_bias
        out = np.empty((dets.shape[0], self.obs.shape[0]), dtype=bool)
        for start in range(0, dets.shape[0], self.chunk):
            part = np.ascontiguousarray(dets[start:start + self.chunk])
            results = self.decoder.decode_batch(part)
            errors = np.array([r.result for r in results], dtype=np.float32) > 0.5
            self.converged += sum(bool(r.converged) for r in results)
            self.shots += len(results)
            pred = (self.obs @ errors.T.astype(np.uint8)).T % 2
            if self.observables_bias is not None:
                pred ^= self.observables_bias
            out[start:start + len(part)] = pred.astype(bool)
        return out


def bposd_decoder(circuit: stim.Circuit, approximate_disjoint_errors: bool, chk_variant: bool, bp: str = "paper"):
    """bp_osd.bposd_batch (chk_variant False: priors combined as independent errors, OSD-CS of order
    min(7, columns - rows)) or bposd_chk (summed priors, order 7), with BP settings BP[bp] (min-sum, parallel)"""
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    chk, obs, priors = dem_to_check_matrices(dem, xor_priors=not chk_variant)
    order = 7 if chk_variant else min(7, chk.shape[1] - chk.shape[0])
    return NvQldpc(chk, obs, priors, bp_method=1, scale_factor=float(BP[bp]["ms_scaling_factor"]),
                   max_iterations=int(BP[bp]["max_iter"]), use_osd=True, osd_method=3, osd_order=int(order),
                   osd_batch_size=1024)


def bposd_gpu(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool, chk_variant: bool = False,
              bp: str = "paper") -> float:
    dec = _cached(("bposd", approximate_disjoint_errors, chk_variant, bp), circuit,
                  lambda: bposd_decoder(circuit, approximate_disjoint_errors, chk_variant, bp))
    return logical_error_rate(dec.predict, circuit, num_shots)


# relay-bp 0.2.1's SinterDecoder_RelayBP defaults, which ECCentric's relay_bp uses: memory strength gamma0 0.1 for
# the first 60 iterations, then up to 60 legs of up to 60 iterations with memory strengths drawn uniformly from
# [-0.24, 0.66], stopping after 5 legs converge and keeping the lowest-weight solution; min-sum without scaling
RELAY = dict(bp_method=3, composition=1, scale_factor=1.0, gamma0=0.1, gamma_dist=[-0.24, 0.66], max_iterations=60,
             srelay_config={"pre_iter": 60, "num_sets": 60, "stopping_criterion": "NConv", "stop_nconv": 5},
             use_osd=False, bp_seed=42, repeatable=True, clip_value=200.0)


def relay_decoder(circuit: stim.Circuit, approximate_disjoint_errors: bool):
    """The check matrices relay-bp builds for the circuit (decided errors pruned into syndrome and observable
    biases), decoded by nv-qldpc's Relay-BP"""
    from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    matrices = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=dem).check_matrices
    import scipy.sparse as sp
    return NvQldpc(sp.csr_matrix(matrices.check_matrix), sp.csr_matrix(matrices.observables_matrix),
                   np.asarray(matrices.error_priors, dtype=np.float64), matrices.syndrome_bias,
                   matrices.observables_bias, **RELAY)


def relay_bp_nvqldpc(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    """nv-qldpc's Relay-BP (for comparison; ECCentric's relay_bp_gpu is relay_gpu.relay_bp_gpu)"""
    dec = _cached(("relay", approximate_disjoint_errors), circuit,
                  lambda: relay_decoder(circuit, approximate_disjoint_errors))
    return logical_error_rate(dec.predict, circuit, num_shots)


def tcbp_jobs_decoder(circuit: stim.Circuit, approximate_disjoint_errors: bool):
    """TCBP-JOBS (Howe, Maurya et al., arXiv:2611.21321, preprint): nv-qldpc's "tcbp-jobs" preset (10 sum-product BP
    iterations, fp32; columns ordered by a TCBP model; JOBS on shots where BP does not converge). The TCBP model is
    fitted here per circuit with init_tcbp_model's defaults (from the check matrix alone, about 10 s; the paper fits
    once per code family and reuses it across distances and error rates)"""
    from cudaq_qec.plugins import nv_qldpc
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    chk, obs, priors = dem_to_check_matrices(dem, xor_priors=True)
    model = nv_qldpc.init_tcbp_model(chk.tocsr())
    return NvQldpc(chk, obs, priors, **nv_qldpc.preset("tcbp-jobs", tcbp_model=model))


def tcbp_jobs_gpu(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    dec = _cached(("tcbp_jobs", approximate_disjoint_errors), circuit,
                  lambda: tcbp_jobs_decoder(circuit, approximate_disjoint_errors))
    return logical_error_rate(dec.predict, circuit, num_shots)
