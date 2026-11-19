"""Batched belief propagation on the GPU (bp.cu): ldpc 2.4.1's parallel-schedule BP, one shot per block."""
import math

import numpy as np
import scipy.sparse as sp

from . import runtime

METHODS = {"product_sum": 0, "minimum_sum": 1}


class GpuBp:
    """BP with ldpc's BpDecoder settings (parallel schedule). Channel probabilities are fixed (`priors`) or given
    per shot to decode()."""

    def __init__(self, pcm, priors=None, max_iter: int = 100, bp_method: str = "minimum_sum",
                 ms_scaling_factor: float = 1.0):
        cp = runtime.cupy()
        csr = sp.csr_matrix(pcm, dtype=np.uint8)
        csr.sum_duplicates()
        csr.sort_indices()
        self.m, self.n = csr.shape
        self.edges = int(csr.nnz)
        # Columns' edges in ascending row order, as CSR edge numbers
        rows = np.repeat(np.arange(self.m), np.diff(csr.indptr))
        order = np.lexsort((rows, csr.indices))
        col_ptr = np.zeros(self.n + 1, dtype=np.int32)
        np.cumsum(np.bincount(csr.indices, minlength=self.n), out=col_ptr[1:])
        self.row_ptr = cp.asarray(csr.indptr.astype(np.int32))
        self.row_col = cp.asarray(csr.indices.astype(np.int32))
        self.col_ptr = cp.asarray(col_ptr)
        self.col_edge = cp.asarray(order.astype(np.int32))
        self.max_iter = int(max_iter)
        self.method = METHODS[bp_method]
        self.ms_scaling = float(ms_scaling_factor)
        self.llr0 = None
        if priors is not None:
            # ldpc computes the channel LLRs with the C library's log on the host; math.log calls the same function
            llr = np.array([math.log((1 - p) / p) for p in np.asarray(priors, dtype=np.float64)])
            self.llr0 = cp.asarray(llr)
        self.kernel = runtime.module("bp").get_function("bp_parallel")
        self.threads = runtime.threads_for(max(self.m, self.n))
        self.state_doubles = 2 * self.edges + self.n
        self.use_shared = self.state_doubles * 8 <= runtime.max_shared_per_block()
        if self.use_shared:
            self.kernel.max_dynamic_shared_size_bytes = self.state_doubles * 8
            per_sm = max(1, (2048 // self.threads))
            self.blocks = runtime.multiprocessors() * per_sm
        else:
            per_sm = max(1, 2048 // self.threads)
            fit = max(1, (runtime.MEMORY_BUDGET // 2) // (self.state_doubles * 8))
            self.blocks = int(min(runtime.multiprocessors() * per_sm, fit))
            self.workspace = cp.empty(self.blocks * self.state_doubles, dtype=cp.float64)
        self.counter = cp.zeros(1, dtype=cp.int32)

    def chunk_shots(self, want_llrs: bool) -> int:
        """Shots per kernel launch, so that the outputs of one launch stay within half the memory budget"""
        per_shot = self.n * (1 + (8 if want_llrs else 0)) + self.m + 16
        return int(max(1, min(1 << 18, (runtime.MEMORY_BUDGET // 2) // per_shot)))

    def decode(self, syndromes, probs=None, want_llrs: bool = True, zero_shortcut: bool = True):
        """syndromes: shots x m (0/1, numpy or CuPy). probs: shots x n channel probabilities, or None for the
        fixed priors. Returns CuPy arrays: decoding (shots x n uint8), log-likelihood ratios (shots x n, or None),
        converged (shots, bool) and iterations (shots)"""
        cp = runtime.cupy()
        syn = cp.ascontiguousarray(cp.asarray(syndromes, dtype=cp.uint8))
        shots = syn.shape[0]
        if probs is None and self.llr0 is None:
            raise ValueError("no channel probabilities: give priors to GpuBp or probs to decode")
        if probs is not None:
            probs = cp.ascontiguousarray(cp.asarray(probs, dtype=cp.float64))
        decoding = cp.empty((shots, self.n), dtype=cp.uint8)
        llrs = cp.empty((shots, self.n), dtype=cp.float64) if want_llrs else None
        converged = cp.empty(shots, dtype=cp.uint8)
        iterations = cp.empty(shots, dtype=cp.int32)
        if shots == 0:
            return decoding, llrs, converged.astype(bool), iterations
        self.counter.fill(0)
        blocks = int(min(self.blocks, shots))
        null = np.uint64(0)
        self.kernel(
            (blocks,), (self.threads,),
            (np.int32(self.m), np.int32(self.n), self.row_ptr, self.row_col, self.col_ptr, self.col_edge,
             self.llr0 if probs is None else null, probs if probs is not None else null, syn,
             np.int32(shots), np.int32(self.max_iter), np.int32(self.method), np.float64(self.ms_scaling),
             null if self.use_shared else self.workspace, np.int32(self.edges), np.int32(int(self.use_shared)), np.int32(int(zero_shortcut)),
             decoding, llrs if want_llrs else null, converged, iterations, self.counter),
            shared_mem=self.state_doubles * 8 if self.use_shared else 0)
        return decoding, llrs, converged.astype(bool), iterations
