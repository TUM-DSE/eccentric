"""GPU versions of the ldpc decoders whose BP stage dominates: BP on the GPU for a batch of shots (bp.cu, ldpc's
parallel min-sum BP with the same arithmetic), then ldpc's own post-processing (LSD, union-find, OSD) on the CPU
for the shots BP did not converge on, from the GPU's log-likelihood ratios. With the same shots, bplsd_gpu and
belief_find_gpu therefore give the CPU decoders' corrections exactly; hier_bposd_gpu's level stages use
product-sum BP, whose tanh and log round differently on the GPU (see analysis/GPU.md)."""
import numpy as np
import stim
from ldpc import BpOsdDecoder, UnionFindDecoder
from ldpc.lsd_decoder import LsdDecoder

from ..bp_osd import dem_to_check_matrices
from . import runtime
from .bp import GpuBp


def logical_error_rate(predict, circuit: stim.Circuit, num_shots: int, batch: int = 1 << 18) -> float:
    """Sample the circuit in batches and count the shots whose predicted observable flips are wrong"""
    sampler = circuit.compile_detector_sampler()
    errors, done = 0, 0
    while done < num_shots:
        n = min(batch, num_shots - done)
        dets, actual = sampler.sample(n, separate_observables=True)
        errors += int(np.count_nonzero(np.any(predict(dets) != actual, axis=1)))
        done += n
    return errors / num_shots


class ObservableMap:
    """Observable flips of decodings (shots x n, on the GPU): the parity of each observable's columns"""

    def __init__(self, obs):
        cp = runtime.cupy()
        obs = obs.tocsr() if hasattr(obs, "tocsr") else obs
        self.columns = [cp.asarray(obs.indices[obs.indptr[k]:obs.indptr[k + 1]].astype(np.int64))
                        for k in range(obs.shape[0])]
        self.obs = obs

    def gpu(self, decoding):
        cp = runtime.cupy()
        out = cp.zeros((decoding.shape[0], len(self.columns)), dtype=cp.uint8)
        for k, cols in enumerate(self.columns):
            if len(cols):
                out[:, k] = (decoding[:, cols].sum(axis=1, dtype=cp.int32) & 1).astype(cp.uint8)
        return out

    def cpu(self, decoding):
        return ((self.obs @ np.asarray(decoding, dtype=np.uint8)) % 2).astype(np.uint8)


class GpuBpPost:
    """BP on the GPU, then `post(syndrome, llrs)` on the CPU for each shot BP did not converge on"""

    def __init__(self, chk, obs, priors, post, **bp):
        self.bp = GpuBp(chk, priors, **bp)
        self.obs = ObservableMap(obs)
        self.post = post
        self.stats = dict(shots=0, postprocessed=0)

    def predict(self, dets: np.ndarray) -> np.ndarray:
        cp = runtime.cupy()
        out = np.empty((dets.shape[0], len(self.obs.columns)), dtype=bool)
        chunk = self.bp.chunk_shots(want_llrs=True)
        for start in range(0, dets.shape[0], chunk):
            syn = np.ascontiguousarray(dets[start:start + chunk], dtype=np.uint8)
            decoding, llrs, converged, _ = self.bp.decode(syn)
            pred = self.obs.gpu(decoding).get()
            failed = cp.flatnonzero(~converged)
            if len(failed):
                llr_failed = llrs[failed].get()
                failed = failed.get()
                for row, shot in enumerate(failed):
                    pred[shot] = self.obs.cpu(self.post(syn[shot], llr_failed[row]))
            out[start:start + chunk] = pred.astype(bool)
            self.stats["shots"] += len(syn)
            self.stats["postprocessed"] += len(failed)
        return out


_last = {}


def _cached(key, circuit, build):
    """One decoder per circuit: ECCentric's adaptive decode calls the decoder once per batch of shots"""
    if _last.get("circuit") is not circuit or _last.get("key") != key:
        _last.clear()
        _last.update(circuit=circuit, key=key, value=build())
    return _last["value"]


def _check_matrices(circuit, approximate_disjoint_errors):
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    return dem_to_check_matrices(dem, xor_priors=True)


def bplsd_decoder(circuit, approximate_disjoint_errors):
    """ldpc_decoders.bplsd's settings: min-sum BP (10,000 iterations, no scaling), LSD-CS of order
    min(7, columns - rows), one bit per growth step"""
    chk, obs, priors = _check_matrices(circuit, approximate_disjoint_errors)
    lsd = LsdDecoder(chk, bits_per_step=1, lsd_method="lsd_cs", lsd_order=min(7, chk.shape[1] - chk.shape[0]))
    return GpuBpPost(chk, obs, priors, lambda s, llr: lsd.decode(s, llr), max_iter=10000,
                     bp_method="minimum_sum", ms_scaling_factor=1.0)


def belief_find_decoder(circuit, approximate_disjoint_errors):
    """ldpc_decoders.belief_find's settings: min-sum BP (10,000 iterations, no scaling), then union-find with
    inversion guided by BP's log-likelihood ratios, all candidate bits per growth step"""
    chk, obs, priors = _check_matrices(circuit, approximate_disjoint_errors)
    seen = np.asarray(chk.sum(axis=0)).ravel() > 0
    chk, obs, priors = chk[:, seen], obs[:, seen], np.asarray(priors)[seen]
    uf = UnionFindDecoder(chk, uf_method="inversion")
    return GpuBpPost(chk, obs, priors, lambda s, llr: uf.decode(s, llrs=llr, bits_per_step=0), max_iter=10000,
                     bp_method="minimum_sum", ms_scaling_factor=1.0)


def bplsd_gpu(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    dec = _cached(("bplsd", approximate_disjoint_errors), circuit,
                  lambda: bplsd_decoder(circuit, approximate_disjoint_errors))
    return logical_error_rate(dec.predict, circuit, num_shots)


def belief_find_gpu(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    dec = _cached(("belief_find", approximate_disjoint_errors), circuit,
                  lambda: belief_find_decoder(circuit, approximate_disjoint_errors))
    return logical_error_rate(dec.predict, circuit, num_shots)


class HierarchicalGpu:
    """hierarchical.Hierarchical with its level stages and the final stage's BP on the GPU and OSD (for the shots
    the final BP does not converge on) on the CPU.

    zero_syndrome: what a level whose detectors are all zero passes on. ldpc's BpDecoder returns at once for a zero
    syndrome without touching its log-likelihood ratios, so the CPU decoder passes on those of the last earlier shot
    (in sampling order, across batches) whose detectors at that level were not all zero, and zeros (probability
    1/2) before the first: "previous" does the same, "run" runs BP on the zero syndrome instead"""

    def __init__(self, dem: stim.DetectorErrorModel, levels: np.ndarray, soft: bool = True,
                 zero_syndrome: str = "previous"):
        from ..hierarchical import EPS, FINAL, STAGE_BP
        cp = runtime.cupy()
        self.chk, obs, self.priors = dem_to_check_matrices(dem)
        self.chk = self.chk.tocsr()
        self.obs = ObservableMap(obs)
        self.soft, self.eps = soft, EPS
        self.rows = [np.flatnonzero(levels == level) for level in range(1, int(levels.max()) + 1)]
        self.rows = [rows for rows in self.rows if len(rows)]
        stage = dict(max_iter=STAGE_BP["max_iter"], bp_method=STAGE_BP["bp_method"])
        self.stages = [GpuBp(self.chk[rows], self.priors, **stage) for rows in self.rows]
        self.rows_gpu = [cp.asarray(rows) for rows in self.rows]
        self.final = GpuBp(self.chk, self.priors, max_iter=FINAL["max_iter"], bp_method=FINAL["bp_method"],
                           ms_scaling_factor=FINAL["ms_scaling_factor"])
        self.osd = BpOsdDecoder(self.chk.tocsc(), error_channel=list(self.priors), **FINAL)
        self.stats = dict(shots=0, postprocessed=0)
        if zero_syndrome not in ("previous", "run"):
            raise ValueError(zero_syndrome)
        self.zero_syndrome = zero_syndrome
        # Each stage's log-likelihood ratios of the last shot with a nonzero syndrome there (ldpc starts at zero)
        self.carry = [cp.zeros(len(self.priors), dtype=cp.float64) for _ in self.stages]

    def predict(self, dets: np.ndarray) -> np.ndarray:
        cp = runtime.cupy()
        out = np.empty((dets.shape[0], len(self.obs.columns)), dtype=bool)
        chunk = min(stage.chunk_shots(True) for stage in self.stages + [self.final]) // 2
        for start in range(0, dets.shape[0], max(1, chunk)):
            syn_host = np.ascontiguousarray(dets[start:start + chunk], dtype=np.uint8)
            syn = cp.asarray(syn_host)
            probs = None
            for k, (rows, stage) in enumerate(zip(self.rows_gpu, self.stages)):
                level_syn = cp.ascontiguousarray(syn[:, rows])
                if self.zero_syndrome == "run":
                    _, llrs, _, _ = stage.decode(level_syn, probs=probs, zero_shortcut=False)
                else:
                    _, llrs, _, _ = stage.decode(level_syn, probs=probs)
                    nonzero = level_syn.any(axis=1)
                    # index of the last shot so far with a nonzero syndrome here (-1: none in this batch)
                    last = cp.where(nonzero, cp.arange(len(nonzero)), -1).get()
                    last = np.maximum.accumulate(last)
                    previous = cp.concatenate([self.carry[k][None, :], llrs])
                    llrs = previous[cp.asarray(last + 1)]
                    if (last >= 0).any():
                        self.carry[k] = llrs[int(np.flatnonzero(last >= 0)[-1])].copy()
                posterior = 1.0 / (1.0 + cp.exp(cp.clip(llrs, -700, 700)))
                probs = (cp.clip(posterior, self.eps, 1 - self.eps) if self.soft
                         else cp.where(posterior > 0.5, 1 - self.eps, self.eps))
                del llrs, posterior
            decoding, _, converged, _ = self.final.decode(syn, probs=probs, want_llrs=False)
            pred = self.obs.gpu(decoding).get()
            failed = cp.flatnonzero(~converged)
            if len(failed):
                probs_failed = (probs[failed] if probs is not None else cp.tile(cp.asarray(self.priors),
                                                                                 (len(failed), 1))).get()
                failed = failed.get()
                for row, shot in enumerate(failed):
                    self.osd.update_channel_probs(probs_failed[row])
                    pred[shot] = self.obs.cpu(self.osd.decode(syn_host[shot]))
            out[start:start + chunk] = pred.astype(bool)
            self.stats["shots"] += len(syn_host)
            self.stats["postprocessed"] += len(failed)
        return out


def hierarchical_gpu(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool, code_name: str,
                     distance=None, soft: bool = True, zero_syndrome: str = "previous") -> float:
    from ..hierarchical import detector_levels

    def build():
        levels = detector_levels(circuit, code_name, distance)
        dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
        return HierarchicalGpu(dem, levels, soft, zero_syndrome)

    dec = _cached(("hier", approximate_disjoint_errors, code_name, distance, soft, zero_syndrome), circuit, build)
    return logical_error_rate(dec.predict, circuit, num_shots)
