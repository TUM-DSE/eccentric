"""Relay-BP on the GPU (relay.cu): relay-bp 0.2.1's decoder with the settings of ECCentric's relay_bp (relay-bp's
SinterDecoder_RelayBP defaults) on relay-bp's own check matrices. With an explicit table of memory strengths it gives
relay-bp's corrections; by default every shot draws its own memory strengths, as relay-bp's random stream does."""
import math

import numpy as np
import scipy.sparse as sp
import stim

from . import runtime
from .ldpc_gpu import ObservableMap, _cached, logical_error_rate

# relay-bp 0.2.1's SinterDecoder_RelayBP defaults (alpha None = no scaling)
RELAY = dict(gamma0=0.1, pre_iter=60, num_sets=60, set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)
DBL_MAX = np.finfo(np.float64).max


class RelayGpu:
    def __init__(self, check_matrix, error_priors, observables_matrix, syndrome_bias=None, observables_bias=None,
                 explicit_gammas=None, seed: int = 0, **settings):
        cp = runtime.cupy()
        self.settings = dict(RELAY, **settings)
        csr = sp.csr_matrix(check_matrix, dtype=np.uint8)
        csr.sum_duplicates()
        csr.sort_indices()
        self.m, self.n = csr.shape
        rows = np.repeat(np.arange(self.m), np.diff(csr.indptr))
        col_edge = np.lexsort((rows, csr.indices)).astype(np.int32)
        col_ptr = np.zeros(self.n + 1, dtype=np.int32)
        np.cumsum(np.bincount(csr.indices, minlength=self.n), out=col_ptr[1:])
        self.row_ptr, self.row_col = cp.asarray(csr.indptr.astype(np.int32)), cp.asarray(csr.indices.astype(np.int32))
        self.col_ptr, self.col_edge = cp.asarray(col_ptr), cp.asarray(col_edge)
        # relay-bp: ln((1 - p) / p) with the C library's log (Rust's f64::ln), +inf stored as f64::MAX
        lpr = np.array([math.log((1.0 - p) / p) if p > 0 else math.inf for p in np.asarray(error_priors, float)])
        self.lpr = cp.asarray(np.where(lpr == math.inf, DBL_MAX, lpr))
        self.quality_lpr = cp.asarray(np.where(np.isfinite(lpr), lpr, 0.0))
        self.obs = ObservableMap(sp.csr_matrix(observables_matrix))
        self.syndrome_bias = None if syndrome_bias is None else np.asarray(syndrome_bias, dtype=np.uint8)
        self.observables_bias = None if observables_bias is None else np.asarray(observables_bias, dtype=np.uint8)
        if explicit_gammas is not None:
            table = np.ascontiguousarray(np.asarray(explicit_gammas, dtype=np.float64))
            assert table.shape[1] == self.n
            self.table, self.table_rows = cp.asarray(table), table.shape[0]
        else:
            self.table, self.table_rows = None, 0
        self.seed = int(seed)
        self.shots_done = 0   # shot index offset of the next batch, so every shot draws its own memory strengths
        self.kernel = runtime.module("relay").get_function("relay_decode")
        E = int(csr.nnz)
        self.state_bytes = (2 * E + 2 * self.n + (2 * self.n + 7) // 8) * 8
        self.threads = runtime.threads_for(max(self.m, self.n))
        per_sm = max(1, 2048 // self.threads)
        self.use_shared = self.state_bytes <= runtime.max_shared_per_block()
        if self.use_shared:
            self.kernel.max_dynamic_shared_size_bytes = self.state_bytes
            self.blocks = runtime.multiprocessors() * per_sm
            self.workspace = None
        else:
            fit = max(1, (runtime.MEMORY_BUDGET // 2) // self.state_bytes)
            self.blocks = int(min(runtime.multiprocessors() * per_sm, fit))
            self.workspace = cp.empty(self.blocks * self.state_bytes // 8, dtype=cp.float64)
        self.counter = cp.zeros(1, dtype=cp.int32)
        self.chunk = int(max(1, min(1 << 18, (runtime.MEMORY_BUDGET // 4) // (self.n + self.m + 16))))
        self.stats = dict(shots=0, converged=0, legs=0)

    def predict(self, dets: np.ndarray) -> np.ndarray:
        cp = runtime.cupy()
        st = self.settings
        out = np.empty((dets.shape[0], len(self.obs.columns)), dtype=bool)
        null = np.uint64(0)
        for start in range(0, dets.shape[0], self.chunk):
            syn = np.ascontiguousarray(dets[start:start + self.chunk], dtype=np.uint8)
            if self.syndrome_bias is not None:
                syn = syn ^ self.syndrome_bias
            shots = syn.shape[0]
            syn_gpu = cp.asarray(syn)
            best = cp.empty((shots, self.n), dtype=cp.uint8)
            converged = cp.empty(shots, dtype=cp.uint8)
            legs = cp.empty(shots, dtype=cp.int32)
            self.counter.fill(0)
            self.kernel((int(min(self.blocks, shots)),), (self.threads,),
                        (np.int32(self.m), np.int32(self.n), self.row_ptr, self.row_col, self.col_ptr, self.col_edge,
                         self.lpr, self.quality_lpr, syn_gpu, np.int32(shots), np.float64(st["gamma0"]),
                         np.int32(st["pre_iter"]), np.int32(st["num_sets"]), np.int32(st["set_max_iter"]),
                         np.int32(st["stop_nconv"]), self.table if self.table is not None else null,
                         np.int32(self.table_rows), np.float64(st["gamma_dist_interval"][0]),
                         np.float64(st["gamma_dist_interval"][1]), np.uint64(self.seed), np.int64(self.shots_done),
                         self.workspace if self.workspace is not None else null, np.int32(int(self.use_shared)),
                         best, converged, legs, self.counter),
                        shared_mem=self.state_bytes if self.use_shared else 0)
            pred = self.obs.gpu(best).get()
            if self.observables_bias is not None:
                pred ^= self.observables_bias
            out[start:start + shots] = pred.astype(bool)
            self.shots_done += shots
            self.stats["shots"] += shots
            self.stats["converged"] += int(converged.sum())
            self.stats["legs"] += int(legs.sum())
        return out


def relay_decoder(circuit: stim.Circuit, approximate_disjoint_errors: bool, **kwargs) -> RelayGpu:
    """relay-bp's check matrices for the circuit (decided errors pruned into syndrome and observable biases)"""
    from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    m = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=dem).check_matrices
    return RelayGpu(m.check_matrix, m.error_priors, m.observables_matrix, m.syndrome_bias, m.observables_bias, **kwargs)


def relay_bp_gpu(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    dec = _cached(("relay_kernel", approximate_disjoint_errors), circuit,
                  lambda: relay_decoder(circuit, approximate_disjoint_errors))
    return logical_error_rate(dec.predict, circuit, num_shots)
