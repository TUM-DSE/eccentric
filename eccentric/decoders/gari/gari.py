import ctypes
import hashlib
import os
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
    f = lib.ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler
    P, I, D, C8 = ctypes.POINTER, ctypes.c_int, ctypes.c_double, ctypes.c_int8
    f.argtypes = [P(D), P(D), I, I, I, P(I), I, I, P(I), I, I, D, P(C8), I, I, I, I, I, I, I, ctypes.c_char,
                  I, I, I, I, P(C8), I, I, P(C8), I, I, P(C8), P(I)]
    f.restype = None
    return f


def _data_qubits(circuit):
    """Qubits measured exactly once (the data qubits of a memory circuit; from the relay-bp library)"""
    times = np.zeros(circuit.num_qubits, dtype=int)
    for inst in circuit:
        if inst.name.startswith("M") and inst.name != "MPP" and not inst.gate_args_copy():
            for t in inst.targets_copy():
                times[t.qubit_value] += 1
    return [q for q in range(circuit.num_qubits) if times[q] == 1]


def _sensitive_detectors(circuit, pauli):
    """Detectors flipped by a `pauli` error on a data qubit after any reset or measurement layer (the relay-bp
    library's filter_detectors_by_basis, which the authors use for IBM's circuits)"""
    noiseless = circuit.flattened().without_noise()
    reference = noiseless.compile_detector_sampler().sample(1)[0]
    sensitive = np.zeros(noiseless.num_detectors, dtype=bool)
    qubits = _data_qubits(noiseless)
    positions = [0] + [i + 1 for i, inst in enumerate(noiseless) if inst.name[0] in "RM"]
    for position in positions:
        for q in qubits:
            injected = noiseless.copy()
            injected.insert(position, stim.CircuitInstruction(f"{pauli}_ERROR", [q], [1.0]))
            sensitive |= injected.compile_detector_sampler().sample(1)[0] != reference
    return sensitive


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
    z_type = _sensitive_detectors(circuit, "X")
    x_type = _sensitive_detectors(circuit, "Z")
    if (z_type & x_type).any():
        raise ValueError(f"GARI needs a CSS circuit: {int((z_type & x_type).sum())} detectors detect both X and Z errors")
    det_type = np.where(z_type, Z_DETECTOR, X_DETECTOR)
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
                    flip_sides=flip)


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
    _library()(arr(llr, ctypes.c_double), arr(llr_ab, ctypes.c_double), shots, len(llr), settings["iterations"],
               arr(hrows, ctypes.c_int), hrows.shape[0], hrows.shape[1], arr(hcols, ctypes.c_int), hcols.shape[0],
               hcols.shape[1], settings["alpha"], arr(syndromes, ctypes.c_int8), settings["seed"], enc.shape_h[0],
               enc.shape_dx[0], settings["custom_schedule"], int(settings["early_stopping"]), 1, settings["ensemble"],
               b"i", enc.nx, enc.nz, enc.mx, enc.shape_h[0], arr(dz, ctypes.c_int8), dz.shape[0], dz.shape[1],
               arr(l_dz, ctypes.c_int8), l_dz.shape[0], l_dz.shape[1], predicted.ctypes.data_as(P(ctypes.c_int8)),
               iterations.ctypes.data_as(P(ctypes.c_int)))
    return predicted.astype(bool)


def gari(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    enc = gari_encoding(circuit, approximate_disjoint_errors)
    dets, actual = circuit.compile_detector_sampler().sample(shots=num_shots, separate_observables=True)
    predicted = decode_observables(enc, dets)
    return np.count_nonzero(np.any(predicted != actual, axis=1)) / num_shots
