"""Union-find decoding with inversion on the GPU (union_find.cu), with ldpc_decoders.union_find's setting: the
error model's check matrix, the priors' log-likelihood ratios as bit weights, all candidate bits per step. Bits of
equal weight are ordered differently from ldpc (whose order comes from its hash sets), and the logical error rate
depends on that order: it differs from union_find's by 0.1-2% (relative), a known difference (analysis/GPU.md)."""
import numpy as np
import scipy.sparse as sp
import stim

from . import runtime
from .ldpc_gpu import _cached, _check_matrices, logical_error_rate


class UnionFindGpu:
    def __init__(self, chk, obs, priors, ties: str = "random"):
        cp = runtime.cupy()
        csr = sp.csr_matrix(chk, dtype=np.uint8)
        csr.sum_duplicates()
        csr.sort_indices()
        csc = csr.tocsc()
        csc.sort_indices()
        self.m, self.n = csr.shape
        priors = np.clip(np.asarray(priors, dtype=np.float64), 1e-12, 1 - 1e-12)
        weights = np.log((1 - priors) / priors)
        # Bits by increasing weight. ldpc orders equal weights by its hash sets' iteration order and an unstable
        # sort, which acts like a random order; the logical error rate depends on it (analysis/GPU.md), so ties go by a
        # fixed random permutation here
        tie = {"index": np.arange(self.n), "reverse": -np.arange(self.n),
               "random": np.random.default_rng(12345).permutation(self.n)}[ties]
        perm = np.lexsort((tie, weights)).astype(np.int32)
        self.row_ptr, self.row_col = cp.asarray(csr.indptr.astype(np.int32)), cp.asarray(csr.indices.astype(np.int32))
        self.col_ptr, self.col_row = cp.asarray(csc.indptr.astype(np.int32)), cp.asarray(csc.indices.astype(np.int32))
        self.perm = cp.asarray(perm)
        obs = sp.csr_matrix(obs)
        self.obs_ptr = cp.asarray(obs.indptr.astype(np.int32))
        self.obs_col = cp.asarray(obs.indices.astype(np.int32))
        self.num_obs = obs.shape[0]
        m, n = self.m, self.n
        self.ws_ints = 8 * m + 3 * n + (2 * n + 3) // 4 + m * ((n + 1 + 31) // 32)
        self.threads = 256
        fit = max(1, (runtime.MEMORY_BUDGET // 2) // (self.ws_ints * 4))
        self.blocks = int(min(runtime.multiprocessors() * (2048 // self.threads), fit))
        self.workspace = cp.empty(self.blocks * self.ws_ints, dtype=cp.int32)
        self.kernel = runtime.module("union_find").get_function("uf_decode")
        self.counter = cp.zeros(1, dtype=cp.int32)
        self.stuck = 0

    def predict(self, dets: np.ndarray) -> np.ndarray:
        cp = runtime.cupy()
        shots = dets.shape[0]
        syn = cp.asarray(np.ascontiguousarray(dets, dtype=np.uint8))
        predicted = cp.empty((shots, self.num_obs), dtype=cp.uint8)
        status = cp.empty(shots, dtype=cp.uint8)
        self.counter.fill(0)
        self.kernel((int(min(self.blocks, max(1, shots))),), (self.threads,),
                    (np.int32(self.m), np.int32(self.n), self.row_ptr, self.row_col, self.col_ptr, self.col_row,
                     self.perm, syn, np.int32(shots), self.workspace, np.int64(self.ws_ints), self.obs_ptr,
                     self.obs_col, np.int32(self.num_obs), predicted, status, self.counter))
        self.stuck += int(shots - int(status.sum()))
        return predicted.get().astype(bool)


def union_find_decoder(circuit, approximate_disjoint_errors):
    chk, obs, priors = _check_matrices(circuit, approximate_disjoint_errors)
    return UnionFindGpu(chk, obs, priors)


def union_find_gpu(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    dec = _cached(("union_find", approximate_disjoint_errors), circuit,
                  lambda: union_find_decoder(circuit, approximate_disjoint_errors))
    return logical_error_rate(dec.predict, circuit, num_shots, batch=1 << 16)
