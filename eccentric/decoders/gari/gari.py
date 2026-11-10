import contextlib
import ctypes
import hashlib
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "hbp_decoder_v2.c")
# The authors' setting for standalone GARI (gari-nms: 400 iterations, normalization factor 0.96875, random
# serial schedule with seed 1, uninformative priors on the effective-error variables, custom schedule 2, early
# stopping); a single decoder, no ensemble
SETTINGS = dict(iterations=400, alpha=0.96875, seed=1, custom_schedule=2, early_stopping=True, ensemble=1)
X_DETECTOR, Z_DETECTOR = 1, 3


def _library():
    """The C decoder, compiled on first use into a cache directory (one file per source version)"""
    with open(SOURCE, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    cache = os.environ.get("ECCENTRIC_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "eccentric"))
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"gari_{digest}_{os.uname().machine}.so")
    if not os.path.exists(path):
        fd, tmp = tempfile.mkstemp(dir=cache, suffix=".so")
        os.close(fd)
        subprocess.run(["cc", "-O3", "-fPIC", "-shared", "-fopenmp", "-o", tmp, SOURCE], check=True,
                       capture_output=True)
        os.replace(tmp, path)
    lib = ctypes.CDLL(path)
    one = lib.ldpc_dec_msaa_quantum_serial_big_matrix_c
    P, I, D, C8 = ctypes.POINTER, ctypes.c_int, ctypes.c_double, ctypes.c_int8
    one.argtypes = [P(D), I, P(I), I, I, P(I), I, I, D, P(C8), I, I, I, I, I, P(C8), P(D)]
    one.restype = I
    f = lib.ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler
    P, I, D, C8 = ctypes.POINTER, ctypes.c_int, ctypes.c_double, ctypes.c_int8
    f.argtypes = [P(D), P(D), I, I, I, P(I), I, I, P(I), I, I, D, P(C8), I, I, I, I, I, I, I, ctypes.c_char,
                  I, I, I, I, P(C8), I, I, P(C8), I, I, P(C8), P(I)]
    f.restype = None
    return f, one


@contextlib.contextmanager
def _quiet():
    """The C decoder prints debugging lines on every call: send the process's stdout to /dev/null meanwhile"""
    sys.stdout.flush()
    saved, null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        yield
    finally:
        # C's own stdout buffer, or its lines come out after stdout is restored
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        os.close(null)


def _data_qubits(circuit):
    """Qubits measured exactly once (the data qubits of a memory circuit; from the relay-bp library)"""
    times = np.zeros(circuit.num_qubits, dtype=int)
    for inst in circuit:
        if inst.name.startswith("M") and inst.name != "MPP" and not inst.gate_args_copy():
            for t in inst.targets_copy():
                times[t.qubit_value] += 1
    return [q for q in range(circuit.num_qubits) if times[q] == 1]


ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}


def _frames(noiseless):
    """Per TICK: the qubits that hold a data qubit then, and the qubits in a Hadamard frame then. Each qubit's
    identity is followed through SWAPs (routing moves qubits); data qubits are the identities measured exactly
    once (the relay-bp library's rule, per identity instead of per physical qubit). A qubit's frame is reset by R,
    toggled by H and travels with its identity through SWAPs. Compiled circuits contain only these, CX and M"""
    ops = list(noiseless)
    holder = list(range(noiseless.num_qubits))  # physical qubit -> identity
    measured = np.zeros(noiseless.num_qubits, dtype=int)
    positions, hadamard_frames, hadamard = [], [], set()
    for inst in ops:
        name = inst.name
        if name == "TICK":
            positions.append(list(holder))
            hadamard_frames.append(frozenset(hadamard))
            continue
        if name in ANNOTATIONS or name.endswith("_ERROR") or name.startswith("DEPOLARIZE"):
            continue
        targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "SWAP":
            for a, b in zip(targets[::2], targets[1::2]):
                holder[a], holder[b] = holder[b], holder[a]
                ha, hb = a in hadamard, b in hadamard
                hadamard.discard(a); hadamard.discard(b)
                if ha:
                    hadamard.add(b)
                if hb:
                    hadamard.add(a)
        elif name == "H":
            hadamard ^= set(targets)
        elif name in ("R", "RZ"):
            hadamard -= set(targets)
        elif name == "RX":
            # |+>: a reset followed by H
            hadamard |= set(targets)
        elif name in ("M", "MZ", "MX"):
            # MX measures in the Hadamard frame and leaves the qubit's frame as it was
            for q in targets:
                measured[holder[q]] += 1
        elif name in ("MR", "MRZ"):
            for q in targets:
                measured[holder[q]] += 1
            hadamard -= set(targets)
        elif name == "MRX":
            for q in targets:
                measured[holder[q]] += 1
            hadamard |= set(targets)
        elif name not in ("CX", "CNOT", "CZ"):
            raise ValueError(f"GARI's detector classification does not follow {name} gates")
    data = {i for i in range(noiseless.num_qubits) if measured[i] == 1}
    data_at = [frozenset(q for q, ident in enumerate(h) if ident in data) for h in positions]
    return data_at, hadamard_frames


def _detector_types(circuit):
    """Z_DETECTOR for the detectors that detect X errors on the data qubits, X_DETECTOR for those that detect Z
    errors, by stim's detecting regions read in each data qubit's own frame (compiled circuits keep data qubits
    in a Hadamard frame for parts of a round). Returns the types and, per detector, how many (tick, data qubit)
    places would flip it with an X error and with a Z error. (The authors, via the relay-bp library, inject errors
    after reset and measurement layers instead, which assumes the data qubits are not in a Hadamard frame there)"""
    noiseless = circuit.flattened().without_noise()
    data_at, frames = _frames(noiseless)
    n = noiseless.num_detectors
    votes_x_errors, votes_z_errors = np.zeros(n), np.zeros(n)
    for target, ticks in noiseless.detecting_regions().items():
        if not target.is_relative_detector_id():
            continue
        k = target.val
        for tick, pauli in ticks.items():
            if tick >= len(data_at):
                continue
            for q in data_at[tick]:
                if q >= len(pauli) or not pauli[q]:
                    continue
                p = pauli[q]  # 0 = I, 1 = X, 2 = Y, 3 = Z
                if q in frames[tick]:
                    p = {1: 3, 3: 1}.get(p, p)
                votes_x_errors[k] += p in (2, 3)
                votes_z_errors[k] += p in (1, 2)
    types = np.where(votes_x_errors > votes_z_errors, Z_DETECTOR, X_DETECTOR)
    return types, votes_x_errors, votes_z_errors


@dataclass
class Encoding:
    order: np.ndarray       # detector indices, X-type first then Z-type: the rows of the decoding matrix
    hrows: np.ndarray       # nonzero columns per row of the GARI matrix (1-based, zero-padded)
    hcols: np.ndarray       # nonzero rows per column
    llr: np.ndarray
    llr_ab: np.ndarray
    shape_h: tuple
    shape_dx: tuple
    nx: int
    nz: int
    mx: int
    dz: np.ndarray
    l_dz: np.ndarray
    flip_sides: bool        # the observables are flipped by errors on the X-type side (X-basis memory)
    h: sp.csr_matrix = None  # the error model's check matrix (detectors x errors) and observables matrix
    l: sp.csr_matrix = None
    i_x: np.ndarray = None  # its single-type errors of either side and its Y-type errors, in the GARI order
    i_z: np.ndarray = None
    i_y: np.ndarray = None


def _columns(matrix):
    m = sp.csc_matrix(matrix)
    return [tuple(m.indices[m.indptr[j]:m.indptr[j + 1]]) for j in range(m.shape[1])]


def _padded(lists):
    width = max(1, max((len(x) for x in lists), default=1))
    out = np.zeros((len(lists), width), dtype=np.int32)
    for i, x in enumerate(lists):
        out[i, :len(x)] = np.asarray(x, dtype=np.int64) + 1
    return out


def gari_encoding(circuit: stim.Circuit, approximate_disjoint_errors: bool = False) -> Encoding:
    """The GARI matrix and priors of a memory circuit (raises ValueError if GARI does not apply to it)"""
    from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices

    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=approximate_disjoint_errors)
    matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    h = sp.csr_matrix(matrices.check_matrix, dtype=np.int8)
    l = sp.csr_matrix(matrices.observables_matrix, dtype=np.int8)
    priors = np.asarray(matrices.priors, dtype=float)
    det_type, votes_x, votes_z = _detector_types(circuit)
    both = int(((votes_x > 0) & (votes_z > 0)).sum())
    if both:
        raise ValueError(f"GARI needs a CSS circuit: {both} detectors detect both X and Z errors on the data qubits")
    hx, hz = h[np.flatnonzero(det_type == X_DETECTOR)], h[np.flatnonzero(det_type == Z_DETECTOR)]
    on_x, on_z = np.asarray(hx.sum(axis=0)).ravel() > 0, np.asarray(hz.sum(axis=0)).ravel() > 0
    # Which side flips the observables: the Z-type side in a Z-basis memory. The authors' decoder predicts the
    # observables from the effective errors of the second (dz) block, so for an X-basis memory the sides swap
    obs_cols = np.asarray(l.sum(axis=0)).ravel() > 0
    flips_x_only, flips_z_only = (obs_cols & on_x & ~on_z).any(), (obs_cols & on_z & ~on_x).any()
    if flips_x_only and flips_z_only:
        raise ValueError("GARI needs the observables on one side: errors seen only by X-type and only by Z-type "
                         "detectors both flip them")
    flip = bool(flips_x_only)
    if flip:
        det_type = np.where(det_type == X_DETECTOR, Z_DETECTOR, X_DETECTOR)
        hx, hz, on_x, on_z = hz, hx, on_z, on_x
    i_y = np.flatnonzero(on_x & on_z)
    i_x, i_z = np.flatnonzero(on_x & ~on_z), np.flatnonzero(on_z & ~on_x)
    dx, dz = hx[:, i_x], hz[:, i_z]
    mx, nx = dx.shape
    mz, nz = dz.shape
    m, n = h.shape
    # Each column's part on a side, matched to the single-type error with the same detectors there
    index_x = {c: i for i, c in reversed(list(enumerate(_columns(dx))))}
    index_z = {c: i for i, c in reversed(list(enumerate(_columns(dz))))}
    part_x, part_z = _columns(hx), _columns(hz)
    x_of_y = np.array([index_x.get(part_x[j], -1) for j in i_y], dtype=int)
    z_of_y = np.array([index_z.get(part_z[j], -1) for j in i_y], dtype=int)
    if (x_of_y < 0).any() or (z_of_y < 0).any():
        missing = int((x_of_y < 0).sum() + (z_of_y < 0).sum())
        raise ValueError(f"GARI needs every part of a Y-type error to be a single-type error; {missing} parts are not")
    # The GARI matrix: [dx 0; 0 dz] over the effective errors, then for each single-type error a check tying its
    # effective error to itself and the Y-type errors that contain it
    ny = len(i_y)
    rows, cols = [], []
    dxc, dzc = sp.coo_matrix(dx), sp.coo_matrix(dz)
    rows += list(dxc.row); cols += list(dxc.col)
    rows += list(mx + dzc.row); cols += list(nx + dzc.col)
    k = np.arange(nx + nz)
    rows += list(m + k); cols += list(k)
    rows += list(m + k); cols += list(nx + nz + k)
    rows += list(m + x_of_y); cols += list(2 * (nx + nz) + np.arange(ny))
    rows += list(m + nx + z_of_y); cols += list(2 * (nx + nz) + np.arange(ny))
    big = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(m + nx + nz, n + nx + nz))
    big.sum_duplicates()
    if big.data.max(initial=1) > 1:
        raise ValueError("GARI matrix has a repeated entry")
    hrows = _padded([big.indices[big.indptr[i]:big.indptr[i + 1]] for i in range(big.shape[0])])
    bigc = big.tocsc()
    hcols = _padded([bigc.indices[bigc.indptr[j]:bigc.indptr[j + 1]] for j in range(big.shape[1])])
    # Priors: uninformative on the effective errors, the error model's on the others (the authors' prt = 0)
    big_priors = np.concatenate([np.full(nx + nz, 0.5), priors[i_x], priors[i_z], priors[i_y]])
    with np.errstate(divide="ignore"):
        llr = np.log((1 - big_priors) / big_priors)
    # For choosing among ensemble members: the summed prior of the errors whose part is each single-type error
    ab = np.zeros(nx + nz)
    all_x = np.array([index_x.get(c, -1) for c in part_x])
    all_z = np.array([index_z.get(c, -1) for c in part_z])
    np.add.at(ab, all_x[all_x >= 0], priors[all_x >= 0])
    np.add.at(ab, nx + all_z[all_z >= 0], priors[all_z >= 0])
    with np.errstate(divide="ignore"):
        llr_ab = np.log((1 - ab) / np.where(ab > 0, ab, 1e-300))
    order = np.concatenate([np.flatnonzero(det_type == X_DETECTOR), np.flatnonzero(det_type == Z_DETECTOR)])
    return Encoding(order=order, hrows=hrows, hcols=hcols, llr=llr, llr_ab=llr_ab, shape_h=(m, n), shape_dx=(mx, nx),
                    nx=nx, nz=nz, mx=mx, dz=dz.toarray().astype(np.int8), l_dz=l[:, i_z].toarray().astype(np.int8),
                    flip_sides=flip, h=h, l=l, i_x=i_x, i_z=i_z, i_y=i_y)


def decode_observables(enc: Encoding, detection_events: np.ndarray, settings: dict = SETTINGS) -> np.ndarray:
    """Predicted observable flips (shots x observables) for detection events (shots x detectors, 0/1)"""
    shots = detection_events.shape[0]
    syndromes = np.ones((shots, enc.hrows.shape[0]), dtype=np.int8)
    syndromes[:, :len(enc.order)] = 1 - 2 * detection_events[:, enc.order].astype(np.int8)
    predicted = np.zeros((shots, enc.l_dz.shape[0]), dtype=np.int8)
    iterations = np.zeros(shots, dtype=np.int32)
    P = ctypes.POINTER
    arr = lambda a, t: np.ascontiguousarray(a).ctypes.data_as(P(t))
    llr, llr_ab = np.ascontiguousarray(enc.llr, np.float64), np.ascontiguousarray(enc.llr_ab, np.float64)
    hrows, hcols = np.ascontiguousarray(enc.hrows), np.ascontiguousarray(enc.hcols)
    dz, l_dz = np.ascontiguousarray(enc.dz), np.ascontiguousarray(enc.l_dz)
    syndromes = np.ascontiguousarray(syndromes)
    decoder = _library()[0]
    with _quiet():
        decoder(arr(llr, ctypes.c_double), arr(llr_ab, ctypes.c_double), shots, len(llr), settings["iterations"],
                arr(hrows, ctypes.c_int), hrows.shape[0], hrows.shape[1], arr(hcols, ctypes.c_int), hcols.shape[0],
                hcols.shape[1], settings["alpha"], arr(syndromes, ctypes.c_int8), settings["seed"], enc.shape_h[0],
                enc.shape_dx[0], settings["custom_schedule"], int(settings["early_stopping"]), 1, settings["ensemble"],
                b"i", enc.nx, enc.nz, enc.mx, enc.shape_h[0], arr(dz, ctypes.c_int8), dz.shape[0], dz.shape[1],
                arr(l_dz, ctypes.c_int8), l_dz.shape[0], l_dz.shape[1], predicted.ctypes.data_as(P(ctypes.c_int8)),
                iterations.ctypes.data_as(P(ctypes.c_int)))
    return predicted.astype(bool)


_ENCODINGS = {}


def cached_encoding(circuit: stim.Circuit, approximate_disjoint_errors: bool) -> Encoding:
    """gari_encoding, kept per circuit: ECCentric's adaptive decode calls the decoder once per batch"""
    key = (hashlib.sha1(str(circuit).encode()).hexdigest(), approximate_disjoint_errors)
    if key not in _ENCODINGS:
        _ENCODINGS.clear()
        _ENCODINGS[key] = gari_encoding(circuit, approximate_disjoint_errors)
    return _ENCODINGS[key]


def gari(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    enc = cached_encoding(circuit, approximate_disjoint_errors)
    dets, actual = circuit.compile_detector_sampler().sample(shots=num_shots, separate_observables=True)
    predicted = decode_observables(enc, dets)
    return np.count_nonzero(np.any(predicted != actual, axis=1)) / num_shots


# GARI++ (beta reimplementation of Maan et al., arXiv:2611.12833, which has no public code): GARI for at most
# GARI_PP_ITERATIONS iterations with the paper's normalization factor per code family; a shot it does not
# converge on keeps the estimate of the iteration with the lightest residual syndrome ("predictive pipelining")
# and the residual is decoded with union-find guided by GARI's posterior log-likelihoods of that iteration
GARI_PP_ITERATIONS = 7
GARI_PP_ALPHA = {"surface": 0.9063, "qldpc": 0.725, "color": 0.825}
QLDPC_PREFIXES = ("bb", "bpc", "gb", "hgp", "lp", "lcs", "qt", "mitten", "qlp", "cbb", "tb", "mhc", "golay", "gross")


def gari_pp_alpha(code_name):
    """The paper's normalization factor for the code's family (surface codes', for families it did not test)"""
    if code_name and code_name.startswith("color"):
        return GARI_PP_ALPHA["color"]
    if code_name and code_name.startswith(QLDPC_PREFIXES):
        return GARI_PP_ALPHA["qldpc"]
    return GARI_PP_ALPHA["surface"]


def gari_pp_decode(enc, detection_events, alpha, iterations=GARI_PP_ITERATIONS, seed=1):
    """Predicted observable flips (shots x observables) and per shot whether GARI converged. Like the authors'
    decoder, everything is in terms of GARI's effective errors on the side that flips the observables (its
    convergence test is on those; the original error variables are not resolved): the residual syndrome of that
    side, its matrix dz for the second stage, and the posterior LLRs of its effective errors as union-find's
    guidance"""
    from ldpc import UnionFindDecoder

    one = _library()[1]
    dz = sp.csr_matrix(enc.dz)
    graphlike = np.diff(sp.csc_matrix(dz).indptr).max(initial=0) <= 2
    residual = UnionFindDecoder(dz, uf_method="peeling" if graphlike else "inversion")
    P = ctypes.POINTER
    llr = np.ascontiguousarray(enc.llr, np.float64)
    hrows, hcols = np.ascontiguousarray(enc.hrows), np.ascontiguousarray(enc.hcols)
    n_big = len(llr)
    est = np.zeros(n_big, dtype=np.int8)
    lam = np.zeros(n_big, dtype=np.float64)
    nx, nz, mx, m = enc.nx, enc.nz, enc.mx, len(enc.order)
    l_dz = enc.l_dz.astype(np.int64)
    shots = detection_events.shape[0]
    predicted = np.zeros((shots, l_dz.shape[0]), dtype=np.int8)
    converged = np.zeros(shots, dtype=bool)

    def run(syndrome, n_iter, early):
        return one(llr.ctypes.data_as(P(ctypes.c_double)), n_iter, hrows.ctypes.data_as(P(ctypes.c_int)),
                   hrows.shape[0], hrows.shape[1], hcols.ctypes.data_as(P(ctypes.c_int)), hcols.shape[0],
                   hcols.shape[1], alpha, syndrome.ctypes.data_as(P(ctypes.c_int8)), seed, enc.shape_h[0],
                   enc.shape_dx[0], 2, int(early), est.ctypes.data_as(P(ctypes.c_int8)),
                   lam.ctypes.data_as(P(ctypes.c_double)))

    with _quiet():
        for shot in range(shots):
            ordered = detection_events[shot, enc.order].astype(np.int8)
            syndrome = np.ones(hrows.shape[0], dtype=np.int8)
            syndrome[:m] = 1 - 2 * ordered
            s_z = ordered[mx:m]
            if run(syndrome, iterations, True) < iterations:
                converged[shot] = True
                predicted[shot] = (l_dz @ est[nx:nx + nz]) % 2
                continue
            # The iteration whose effective errors leave the lightest residual syndrome on the observables' side
            # (the first of equals). The decoder's random schedule is seeded, so t iterations reproduce the first t
            # of a longer run
            best = None
            for t in range(1, iterations + 1):
                run(syndrome, t, False)
                e = est[nx:nx + nz].copy()
                res = (s_z + dz @ e) % 2
                weight = int(res.sum())
                if best is None or weight < best[0]:
                    best = (weight, e, res.astype(np.uint8), lam[nx:nx + nz].copy())
            weight, e, res, posterior = best
            if weight:
                e = (e + residual.decode(res, llrs=np.clip(np.abs(posterior), 1e-6, 50))) % 2
            predicted[shot] = (l_dz @ e) % 2
    return predicted.astype(bool), converged


def gari_pp(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool, code_name: str = None) -> float:
    enc = cached_encoding(circuit, approximate_disjoint_errors)
    dets, actual = circuit.compile_detector_sampler().sample(shots=num_shots, separate_observables=True)
    predicted, _ = gari_pp_decode(enc, dets, gari_pp_alpha(code_name))
    return np.count_nonzero(np.any(predicted != actual, axis=1)) / num_shots
