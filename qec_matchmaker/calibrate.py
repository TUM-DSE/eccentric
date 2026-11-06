"""Scaling predicted decode times to another machine.

The time models predict ms per shot on one core of the reference machine the training times were
measured on. `measure()` times each installed decoder on `benchmark_circuits()` here, `speed_factors()`
divides by `REFERENCE`, the same measurement on the reference machine, and a predicted time multiplied
by a decoder's factor is that decoder's time on this machine. Factors are per decoder because decoders
do not all speed up alike from one machine to another.
"""
import datetime
import importlib
import json
import os
import platform
import time

import numpy as np
import stim

from .features import features_from_circuit

DECODERS = ("mwpm", "mwpf", "bposd", "bposd_tuned", "bplsd", "belief_find", "relay_bp", "tesseract", "chromobius")

# ms per shot on one core of the reference machine: geometric mean over benchmark_circuits() of
# measure(shots=2000, seed=0). None until measured there: run fill_reference() on the reference
# machine and paste the values here
REFERENCE = {name: None for name in DECODERS}

P = 0.003
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                    "RAYON_NUM_THREADS")


def _color_code(p, rounds):
    """Distance-3 color code memory measured with MPP, its detectors annotated with Chromobius' basis
    and color (4th coordinate, 0-2 X and 3-5 Z), after Chromobius' getting-started example"""
    tiles = [(0, 1, 2, 3), (1, 2, 4, 5), (2, 3, 5, 6)]
    stabilizers = " ".join("*".join(f"{basis}{q}" for q in tile) for basis in "XZ" for tile in tiles)
    detectors = "\n".join(f"DETECTOR({k}, 0, 0, {k}) rec[{k - 6}] rec[{k - 12}]" for k in range(6))
    observable = "MPP " + "*".join(f"Z{q}" for q in range(7)) + "\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    data_noise = f"DEPOLARIZE1({p}) 0 1 2 3 4 5 6\n"
    return stim.Circuit(
        f"{observable}MPP {stabilizers}\nTICK\n"
        f"REPEAT {rounds} {{\n{data_noise}MPP({p}) {stabilizers}\n{detectors}\nSHIFT_COORDS(0, 0, 1)\nTICK\n}}\n"
        f"{data_noise}MPP {stabilizers}\n{detectors}\n{observable}")


def benchmark_circuits() -> list:
    """The fixed small circuits decoders are timed on, as (name, circuit), all with p = 0.003 circuit
    noise and d rounds: rotated surface code memory at d = 3 and 5, Stim's color code memory at d = 3 (if
    this Stim generates it) and a Chromobius-annotated color code at d = 3"""
    noise = dict(after_clifford_depolarization=P, before_measure_flip_probability=P,
                 after_reset_flip_probability=P, before_round_data_depolarization=P)
    circuits = [(f"surface_d{d}", stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                                         **noise)) for d in (3, 5)]
    try:
        circuits.append(("color_xyz_d3", stim.Circuit.generated("color_code:memory_xyz", distance=3, rounds=3,
                                                                **noise)))
    except ValueError:
        pass
    circuits.append(("color_annotated_d3", _color_code(P, rounds=3)))
    return circuits


def _check_matrices(circuit):
    """Check and observables matrices (scipy csc) and priors of the circuit's DEM, mechanisms with equal
    symptoms merged into one with the XOR of their probabilities, as ECCentric builds them for ldpc"""
    from scipy import sparse
    columns, priors = {}, []
    for inst in circuit.detector_error_model(approximate_disjoint_errors=True).flattened():
        if inst.type != "error":
            continue
        targets, p = inst.targets_copy(), inst.args_copy()[0]
        key = (frozenset(t.val for t in targets if t.is_relative_detector_id()),
               frozenset(t.val for t in targets if t.is_logical_observable_id()))
        j = columns.setdefault(key, len(priors))
        if j == len(priors):
            priors.append(0.0)
        priors[j] = priors[j] * (1 - p) + p * (1 - priors[j])

    def matrix(part, rows):
        entries = [(i, j) for key, j in columns.items() for i in key[part]]
        r, c = zip(*entries) if entries else ((), ())
        return sparse.csc_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(rows, len(priors)))
    return matrix(0, circuit.num_detectors), matrix(1, circuit.num_observables), np.array(priors)


# Each builder takes the decoder's entry point and a circuit and returns a function from detection
# events (shots x detectors, bool) to predicted observable flips, with ECCentric's decoder settings
_BP = dict(max_iter=10000, bp_method="minimum_sum", ms_scaling_factor=1.0)


def _per_shot(decoder, obs):
    return lambda dets: np.array([obs @ decoder.decode(s) % 2 for s in dets], bool)


def _packed(decode_packed, num_observables):
    def decode(dets):
        flips = decode_packed(np.packbits(dets, axis=1, bitorder="little"))
        return np.unpackbits(flips, axis=1, bitorder="little", count=num_observables).astype(bool)
    return decode


def _mwpm(Matching, circuit):
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    return Matching.from_detector_error_model(dem).decode_batch


def _bposd(BpOsdDecoder, circuit, bp=_BP):
    chk, obs, priors = _check_matrices(circuit)
    return _per_shot(BpOsdDecoder(chk, error_channel=list(priors), osd_method="osd_cs",
                                  osd_order=min(7, chk.shape[1] - chk.shape[0]), **bp), obs)


# ECCentric's bposd_tuned
_BP_TUNED = dict(max_iter=100, bp_method="minimum_sum", ms_scaling_factor=0.625)


def _bplsd(BpLsdDecoder, circuit):
    chk, obs, priors = _check_matrices(circuit)
    return _per_shot(BpLsdDecoder(chk, error_channel=list(priors), lsd_method="lsd_cs",
                                  lsd_order=min(7, chk.shape[1] - chk.shape[0]), **_BP), obs)


def _belief_find(BeliefFindDecoder, circuit):
    chk, obs, priors = _check_matrices(circuit)
    # BeliefFind rejects mechanisms that flip no detector; no decoder can correct them anyway
    seen = np.asarray(chk.sum(axis=0)).ravel() > 0
    decoder = BeliefFindDecoder(chk[:, seen], error_channel=list(priors[seen]), uf_method="inversion", **_BP)
    return _per_shot(decoder, obs[:, seen])


def _sinter(decoder, circuit):
    compiled = decoder.compile_decoder_for_dem(dem=circuit.detector_error_model(approximate_disjoint_errors=True))
    return _packed(lambda packed: compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed),
                   circuit.num_observables)


def _relay_bp(SinterDecoder_RelayBP, circuit):
    # On the exact syndromes: relay-bp 0.2.1's sinter decode_shots_bit_packed pads them to a multiple of 8 bits
    # (trmue/relay#34), after which Relay-BP rarely converges; ECCentric decodes them the same way
    compiled = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=circuit.detector_error_model(approximate_disjoint_errors=True))
    m = compiled.check_matrices

    def decode(dets):
        s = dets.astype(np.uint8)
        if m.syndrome_bias is not None:
            s = (s + m.syndrome_bias) % 2
        pred = compiled.observable_decoder.decode_observables_batch(s, parallel=False).astype(bool)
        return pred ^ m.observables_bias.astype(bool) if m.observables_bias is not None else pred
    return decode


def _chromobius(compile_decoder_for_dem, circuit):
    compiled = compile_decoder_for_dem(circuit.detector_error_model(approximate_disjoint_errors=True))
    return _packed(compiled.predict_obs_flips_from_dets_bit_packed, circuit.num_observables)


_DECODERS = {
    "mwpm": ("pymatching", "Matching", _mwpm),
    "bposd": ("ldpc", "BpOsdDecoder", _bposd),
    "bposd_tuned": ("ldpc", "BpOsdDecoder", lambda cls, c: _bposd(cls, c, _BP_TUNED)),
    "mwpf": ("mwpf", "SinterMWPFDecoder", lambda cls, c: _sinter(cls(cluster_node_limit=50), c)),
    "bplsd": ("ldpc", "BpLsdDecoder", _bplsd),
    "belief_find": ("ldpc", "BeliefFindDecoder", _belief_find),
    "relay_bp": ("relay_bp.stim.sinter.decoders", "SinterDecoder_RelayBP", _relay_bp),
    "tesseract": ("tesseract_decoder", "make_tesseract_sinter_decoders_dict",
                  lambda make, c: _sinter(make()["tesseract"], c)),
    "chromobius": ("chromobius", "compile_decoder_for_dem", _chromobius),
}


def _load(name):
    """The decoder's entry point, or None if its library is not installed"""
    module, attr, _ = _DECODERS[name]
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        return None


def _applies(name, features):
    """Matching needs errors that decompose into edges, Chromobius color-annotated detectors"""
    if name == "mwpm":
        return features["decomposable"] == 1
    if name == "chromobius":
        return features["color_annotated"] == 1
    return True


def measure(decoders=None, shots: int = 2000, seed: int = 0) -> dict:
    """Ms of process time per shot of each installed decoder (all of `DECODERS` if None) on this machine:
    the geometric mean over the benchmark circuits it applies to, decoder construction included.
    Decoders whose library is not installed are left out. With ldpc's decoders this takes minutes: their
    BP runs up to 10000 iterations"""
    decoders = DECODERS if decoders is None else decoders
    unknown = set(decoders) - set(DECODERS)
    if unknown:
        raise ValueError(f"unknown decoders {sorted(unknown)}; known: {DECODERS}")
    # One core, as for the training times; thread pools read these when they start
    for var in THREAD_VARIABLES:
        os.environ.setdefault(var, "1")
    circuits = []
    for _, circuit in benchmark_circuits():
        dets, _ = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        circuits.append((circuit, features_from_circuit(circuit), dets))
    measured = {}
    for name in decoders:
        entry = _load(name)
        if entry is None:
            continue
        build, ms = _DECODERS[name][2], []
        for circuit, features, dets in circuits:
            if not _applies(name, features):
                continue
            start = time.process_time()
            build(entry, circuit)(dets)
            ms.append(1000 * (time.process_time() - start) / shots)
        if ms:
            measured[name] = float(np.exp(np.mean(np.log(np.maximum(ms, 1e-9)))))
    return measured


def fill_reference(measured=None, **measure_kwargs) -> dict:
    """Fill `REFERENCE` in place from `measured` (default: `measure(**measure_kwargs)` here); meant for the
    reference machine, after which the values belong in the literal above"""
    measured = measure(**measure_kwargs) if measured is None else measured
    REFERENCE.update({name: float(ms) for name, ms in measured.items() if name in REFERENCE})
    return REFERENCE


def speed_factors(measured, reference=REFERENCE) -> dict:
    """Per decoder, measured / reference ms per shot, for decoders both have: how many times slower this
    machine is; multiply a predicted time by it to get this machine's"""
    return {name: ms / reference[name] for name, ms in measured.items()
            if ms is not None and reference.get(name)}


def save(path, factors):
    """Write speed factors as JSON, with the machine and date they were measured on"""
    with open(path, "w") as f:
        json.dump({"speed_factors": factors, "platform": platform.platform(),
                   "date": datetime.date.today().isoformat()}, f, indent=2, sort_keys=True)


def load(path) -> dict:
    """Speed factors written by `save`"""
    with open(path) as f:
        return {name: float(factor) for name, factor in json.load(f)["speed_factors"].items()}
