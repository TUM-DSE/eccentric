"""GARI on the GPU (gari.cu): the same encoding as ../gari (gari_encoding), decoded one shot per thread block
with the C decoder's serial schedule split into levels of independent rows."""
import numpy as np
import stim

from ..gari.gari import SETTINGS, cached_encoding
from . import runtime
from .ldpc_gpu import logical_error_rate

try:
    from numba import njit
except ImportError:  # the schedule is then computed in plain Python (slow for large circuits)
    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)


@njit(cache=True)
def _xorshift_permute(array, seed):
    """random_permutation of hbp_decoder_v2.c (in place)"""
    n = array.shape[0]
    if n <= 1:
        return
    state = np.uint32(seed)
    if state == 0:
        state = np.uint32(1)
    for i in range(n - 1, 0, -1):
        state ^= np.uint32(state << np.uint32(13))
        state ^= np.uint32(state >> np.uint32(17))
        state ^= np.uint32(state << np.uint32(5))
        j = np.int64(state % np.uint32(i + 1))
        tmp = array[i]
        array[i] = array[j]
        array[j] = tmp


@njit(cache=True)
def _schedule(R, N, row_ptr, row_var, n_loop, seed, hshape0, dxshape0):
    """Each iteration's row order for custom schedule 2 (bottom rows, then the dx rows, then the dz rows, the last
    two shuffled anew every iteration from the previous iteration's order) split into levels: a row's level is
    one more than the last level among the earlier rows of the iteration that share a variable with it.
    Returns the rows level by level, the levels' offsets and each iteration's first level"""
    rows_a = np.arange(dxshape0).astype(np.int32)
    rows_b = (np.arange(hshape0 - dxshape0) + dxshape0).astype(np.int32)
    bottom = (np.arange(R - hshape0) + hshape0).astype(np.int32)
    sched = np.empty((n_loop + 1) * R, dtype=np.int32)
    level_sizes = np.zeros((n_loop + 1) * R, dtype=np.int64)
    total = 0
    iter_level = np.zeros(n_loop + 2, dtype=np.int32)
    last = np.zeros(N, dtype=np.int32)
    level = np.empty(R, dtype=np.int32)
    order = np.empty(R, dtype=np.int32)
    for loop in range(n_loop + 1):
        if seed:
            _xorshift_permute(rows_a, seed + loop)
            _xorshift_permute(rows_b, seed + loop)
        order[:R - hshape0] = bottom
        order[R - hshape0:R - hshape0 + dxshape0] = rows_a
        order[R - hshape0 + dxshape0:] = rows_b
        last[:] = 0
        depth = 0
        for k in range(R):
            r = order[k]
            lv = 0
            for e in range(row_ptr[r], row_ptr[r + 1]):
                if last[row_var[e]] > lv:
                    lv = last[row_var[e]]
            lv += 1
            level[r] = lv
            for e in range(row_ptr[r], row_ptr[r + 1]):
                last[row_var[e]] = lv
            if lv > depth:
                depth = lv
        counts = np.zeros(depth + 1, dtype=np.int64)
        for r in range(R):
            counts[level[r]] += 1
        offsets = np.zeros(depth + 2, dtype=np.int64)
        for lv in range(1, depth + 1):
            offsets[lv + 1] = offsets[lv] + counts[lv]
        base = loop * R
        for k in range(R):
            r = order[k]
            lv = level[r]
            sched[base + offsets[lv]] = r
            offsets[lv] += 1
        for lv in range(1, depth + 1):
            level_sizes[total] = counts[lv]
            total += 1
        iter_level[loop + 1] = iter_level[loop] + depth
    return sched, level_sizes[:total].copy(), iter_level


class GariGpu:
    def __init__(self, enc, settings: dict = SETTINGS):
        cp = runtime.cupy()
        if settings["custom_schedule"] != 2 or not settings["early_stopping"] or settings["ensemble"] != 1:
            raise ValueError("gari_gpu implements GARI's standalone setting (schedule 2, early stopping, one member)")
        self.enc = enc
        R, width = enc.hrows.shape
        lengths = (enc.hrows > 0).sum(axis=1)
        row_ptr = np.zeros(R + 1, dtype=np.int32)
        np.cumsum(lengths, out=row_ptr[1:])
        row_var = (enc.hrows[enc.hrows > 0] - 1).astype(np.int32)  # row-major order keeps each row's column order
        N = len(enc.llr)
        n_loop = int(settings["iterations"])
        sched, sizes, iter_level = _schedule(R, N, row_ptr, row_var, n_loop, int(settings["seed"]),
                                             int(enc.shape_h[0]), int(enc.shape_dx[0]))
        # Level lv's rows are sched[start[lv]:end[lv]], within its iteration's block of R rows
        start = np.repeat(np.arange(n_loop + 1, dtype=np.int64) * R, np.diff(iter_level))
        for loop in range(n_loop + 1):
            a, b = iter_level[loop], iter_level[loop + 1]
            start[a:b] += np.concatenate([[0], np.cumsum(sizes[a:b])[:-1]])
        end = start + sizes
        self.levels_per_iteration = np.diff(iter_level)
        self.R, self.N, self.n_loop = R, N, n_loop
        self.alpha = float(settings["alpha"])
        self.row_ptr, self.row_var = cp.asarray(row_ptr), cp.asarray(row_var)
        self.llr = cp.asarray(np.ascontiguousarray(enc.llr, dtype=np.float64))
        self.sched = cp.asarray(sched)
        self.level_start = cp.asarray(start.astype(np.int32))
        self.level_end = cp.asarray(end.astype(np.int32))
        self.iter_level = cp.asarray(iter_level.astype(np.int32))
        l_dz = np.asarray(enc.l_dz)
        obs_ptr = np.zeros(l_dz.shape[0] + 1, dtype=np.int32)
        obs_ptr[1:] = np.cumsum((l_dz != 0).sum(axis=1))
        self.obs_ptr = cp.asarray(obs_ptr)
        self.obs_col = cp.asarray(np.nonzero(l_dz)[1].astype(np.int32))
        self.num_obs = l_dz.shape[0]
        self.kernel = runtime.module("gari").get_function("gari_decode")
        E = int(row_ptr[-1])
        self.state_bytes = (E + N + (N + 7) // 8) * 8
        self.threads = runtime.threads_for(int(np.max(np.concatenate([[1], sizes]))), cap=256)
        self.use_shared = self.state_bytes <= runtime.max_shared_per_block()
        per_sm = max(1, 2048 // self.threads)
        if self.use_shared:
            self.kernel.max_dynamic_shared_size_bytes = self.state_bytes
            self.blocks = runtime.multiprocessors() * per_sm
            self.workspace = None
        else:
            fit = max(1, (runtime.MEMORY_BUDGET // 2) // self.state_bytes)
            self.blocks = int(min(runtime.multiprocessors() * per_sm, fit))
            self.workspace = cp.empty(self.blocks * self.state_bytes // 8, dtype=cp.float64)
        self.counter = cp.zeros(1, dtype=cp.int32)

    def predict(self, dets: np.ndarray) -> np.ndarray:
        cp = runtime.cupy()
        enc = self.enc
        shots = dets.shape[0]
        syn = np.ones((shots, self.R), dtype=np.int8)
        syn[:, :len(enc.order)] = 1 - 2 * dets[:, enc.order].astype(np.int8)
        syn = cp.asarray(syn)
        predicted = cp.empty((shots, self.num_obs), dtype=cp.int8)
        self.iterations = cp.empty(shots, dtype=cp.int32)
        self.counter.fill(0)
        null = np.uint64(0)
        self.kernel((int(min(self.blocks, max(1, shots))),), (self.threads,),
                    (np.int32(self.R), np.int32(self.N), self.row_ptr, self.row_var, self.llr, syn, np.int32(shots),
                     np.int32(self.n_loop), np.float64(self.alpha), self.sched, self.level_start, self.level_end,
                     self.iter_level, np.int32(enc.mx), np.int32(enc.shape_h[0]), np.int32(enc.nx), self.obs_ptr,
                     self.obs_col, np.int32(self.num_obs), self.workspace if self.workspace is not None else null,
                     np.int32(int(self.use_shared)), predicted, self.iterations, self.counter),
                    shared_mem=self.state_bytes if self.use_shared else 0)
        return predicted.get().astype(bool)


_last = {}


def gari_gpu(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    enc = cached_encoding(circuit, approximate_disjoint_errors)
    if _last.get("enc") is not enc:
        _last.clear()
        _last.update(enc=enc, value=GariGpu(enc))
    return logical_error_rate(_last["value"].predict, circuit, num_shots, batch=1 << 16)
