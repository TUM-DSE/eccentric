"""Features of a noisy circuit, its detector error model, or a check matrix with priors.

Everything the models see comes from here. The error-model features are computed from the detector
error model without decomposition, so a circuit, its DEM and its check matrix with priors give the
same numbers. A detector fires with probability (1 - prod(1 - 2p)) / 2 over the error mechanisms that
flip it, which is exact for independent mechanisms.
"""
import math

import numpy as np
import stim

from .detector_types import FEATURES as DETECTOR_TYPE_FEATURES, detector_type_features

# Bump when a feature's definition changes: stored features of another version are recomputed
FEATURE_VERSION = 2
# Features added by each version, which a record of the version before can be upgraded with alone
# (pipeline.extract): version 2 added the detector types (detector_types.py)
ADDED_IN = {2: DETECTOR_TYPE_FEATURES}

FEATURE_NAMES = [
    "detectors", "observables", "mechanisms", "mechanism_weight_mean", "mechanism_weight_max",
    "hyperedge_fraction", "error_mass", "error_mass_per_detector", "det_fire_mean", "det_fire_median",
    "det_fire_p90", "det_fire_max", "logical_mass", "undetectable_faults", "undetectable_probability",
    "graphlike", "color_annotated", "qubits", "two_qubit_gates", "swaps", "ticks", "measurements", "rounds",
    "decomposable", *DETECTOR_TYPE_FEATURES,
]
CIRCUIT_ONLY = ("qubits", "two_qubit_gates", "swaps", "ticks", "measurements", "decomposable", *DETECTOR_TYPE_FEATURES)


def _summarize(num_detectors, num_observables, probs, weights, fire, has_obs):
    probs, weights, has_obs = np.asarray(probs, float), np.asarray(weights, int), np.asarray(has_obs, bool)
    nan = math.nan
    undetectable = has_obs & (weights == 0)
    return {
        "detectors": num_detectors,
        "observables": num_observables,
        "mechanisms": len(probs),
        "mechanism_weight_mean": float(weights.mean()) if len(weights) else nan,
        "mechanism_weight_max": int(weights.max()) if len(weights) else nan,
        "hyperedge_fraction": float((weights > 2).mean()) if len(weights) else nan,
        "error_mass": float(probs.sum()),
        "error_mass_per_detector": float(probs.sum()) / num_detectors if num_detectors else nan,
        "det_fire_mean": float(fire.mean()) if len(fire) else nan,
        "det_fire_median": float(np.median(fire)) if len(fire) else nan,
        "det_fire_p90": float(np.quantile(fire, 0.9)) if len(fire) else nan,
        "det_fire_max": float(fire.max()) if len(fire) else nan,
        "logical_mass": float(probs[has_obs].sum()),
        "undetectable_faults": int(undetectable.sum()),
        "undetectable_probability": float(probs[undetectable].sum()),
        "graphlike": int(len(weights) == 0 or weights.max() <= 2),
    }


def _firing(num_detectors, det_index, det_logs):
    """Firing probability of each detector from log(1 - 2p) of the mechanisms flipping it"""
    total = np.zeros(num_detectors)
    np.add.at(total, det_index, det_logs)
    return (1 - np.exp(total)) / 2


def features_from_dem(dem: stim.DetectorErrorModel) -> dict:
    """Features of a detector error model; circuit-only features are NaN"""
    probs, weights, has_obs, det_index, det_logs = [], [], [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        probs.append(p)
        weights.append(len(dets))
        has_obs.append(any(t.is_logical_observable_id() for t in inst.targets_copy()))
        with np.errstate(divide="ignore"):
            log = math.log(1 - 2 * p) if p < 0.5 else -math.inf
        det_index.extend(dets)
        det_logs.extend([log] * len(dets))
    fire = _firing(dem.num_detectors, np.array(det_index, dtype=int), np.array(det_logs))
    features = _summarize(dem.num_detectors, dem.num_observables, probs, weights, fire, has_obs)
    coords = dem.get_detector_coordinates()
    features["color_annotated"] = int(dem.num_detectors > 0 and all(len(c) >= 4 for c in coords.values()))
    times = [c[2] for c in coords.values() if len(c) >= 3]
    features["rounds"] = len(set(times)) if times and len(times) == dem.num_detectors else math.nan
    features.update({name: math.nan for name in CIRCUIT_ONLY})
    return features


def features_from_circuit(circuit: stim.Circuit, check_decomposition: bool = True) -> dict:
    """Features of a noisy circuit: its detector error model's, plus its size and, optionally, whether
    its errors decompose into edges (whether matching applies)"""
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    features = features_from_dem(dem)
    features.update(detector_type_features(circuit, dem))
    pairs = swaps = 0
    for inst in circuit.flattened():
        data = stim.gate_data(inst.name)
        if data.is_two_qubit_gate and data.is_unitary:
            n = len(inst.targets_copy()) // 2
            pairs += n
            swaps += n if inst.name == "SWAP" else 0
    features.update(qubits=circuit.num_qubits, two_qubit_gates=pairs, swaps=swaps, ticks=circuit.num_ticks,
                    measurements=circuit.num_measurements)
    if check_decomposition:
        try:
            circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
            features["decomposable"] = 1
        except ValueError:
            features["decomposable"] = 0
    return features


def features_from_matrices(check_matrix, priors, observables_matrix=None) -> dict:
    """Features of a check matrix (detectors x mechanisms, dense or scipy sparse) with mechanism priors,
    as decoder-bench traces store them; without coordinates, `color_annotated` and `rounds` are NaN"""
    H = check_matrix.tocsc() if hasattr(check_matrix, "tocsc") else np.asarray(check_matrix)
    probs = np.asarray(priors, float).ravel()
    num_detectors, num_mechanisms = H.shape
    weights = np.asarray((H != 0).sum(axis=0)).ravel()
    if observables_matrix is not None:
        L = observables_matrix.tocsc() if hasattr(observables_matrix, "tocsc") else np.asarray(observables_matrix)
        has_obs = np.asarray((L != 0).sum(axis=0)).ravel() > 0
        num_observables = L.shape[0]
    else:
        has_obs, num_observables = np.zeros(num_mechanisms, bool), math.nan
    with np.errstate(divide="ignore"):
        logs = np.where(probs < 0.5, np.log(np.clip(1 - 2 * probs, 0, None)), -np.inf)
    rows, cols = (H != 0).nonzero() if not hasattr(H, "tocoo") else (H.tocoo().row, H.tocoo().col)
    fire = _firing(num_detectors, np.asarray(rows), logs[np.asarray(cols)])
    features = _summarize(num_detectors, num_observables, probs, weights, fire, has_obs)
    if observables_matrix is None:
        features.update(logical_mass=math.nan, undetectable_faults=math.nan, undetectable_probability=math.nan)
    features.update(color_annotated=math.nan, rounds=math.nan)
    features.update({name: math.nan for name in CIRCUIT_ONLY})
    return features
