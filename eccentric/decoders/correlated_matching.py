"""Correlated matching (mwpm_corr): PyMatching's two-pass correlated MWPM on a decomposed error model.

Errors that flip more than two detectors (hyperedges) must be split into graphlike components for
matching. Stim splits most of them; correlated matching (PyMatching >= 2.3, enable_correlations)
then keeps the split: after a first matching, the edges whose partner components were matched are
reweighted by their conditional probability and the syndrome is matched again. This is the approach of
correlated decoding of transversal gates (Cain et al., Phys. Rev. Lett. 133, 240602, 2024), where a
transversal CNOT copies errors between blocks.

Stim cannot split some hyperedges, for example a fault of a surgery or transversal-CNOT gadget that
flips five detectors (plain MWPM then fails). Those are split here by matching their own detectors on
the graph of the model's graphlike errors: the edges of that matching are the components. A split
whose components flip other observables than the error itself is kept and counted ("mismatched").
"""
import pymatching
import stim
import numpy as np

_last = {}


def _targets(instruction):
    """Components of an error instruction: lists of (detectors, observables), split at ^"""
    components, dets, obs = [], [], []
    for t in instruction.targets_copy():
        if t.is_separator():
            components.append((dets, obs))
            dets, obs = [], []
        elif t.is_relative_detector_id():
            dets.append(t.val)
        elif t.is_logical_observable_id():
            obs.append(t.val)
    components.append((dets, obs))
    return components


def _format(p, components):
    parts = [" ".join([f"D{d}" for d in dets] + [f"L{o}" for o in obs]) for dets, obs in components]
    return f"error({p!r}) " + " ^ ".join(part for part in parts if part)


def _fold_undetectable(components):
    """Correlated matching rejects components without detectors: their observables go to the first
    component that has detectors (an error without any detector is left out)"""
    detected = [(list(d), set(o)) for d, o in components if d]
    if not detected:
        return []
    for dets, obs in components:
        if not dets:
            detected[0][1].symmetric_difference_update(obs)
    return [(d, sorted(o)) for d, o in detected]


def decomposed_dem(circuit: stim.Circuit, approximate_disjoint_errors: bool):
    """The circuit's error model with every error split into graphlike components and every probability
    at most 1/2; how many errors had to be split here (stim could not) and how many of those splits flip
    other observables or were left out; and the frame (detectors, observables) of the errors that were
    more likely than not, which the decoder flips in every shot instead (correlated matching rejects
    probabilities above 1/2: an independent error of probability p > 1/2 is the same as that error always
    happening and another of probability 1 - p, which makes the substitution exact)"""
    try:
        dem = circuit.detector_error_model(decompose_errors=True,
                                           approximate_disjoint_errors=approximate_disjoint_errors).flattened()
    except ValueError:
        dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True,
                                           approximate_disjoint_errors=approximate_disjoint_errors).flattened()
    det_frame = np.zeros(dem.num_detectors, dtype=bool)
    obs_frame = np.zeros(dem.num_observables, dtype=bool)
    entries = []
    for inst in dem:
        if inst.type != "error":
            entries.append(str(inst))
            continue
        p, components = inst.args_copy()[0], _targets(inst)
        if p > 0.5:
            for dets, obs in components:
                det_frame[dets] ^= True
                obs_frame[obs] ^= True
            p = 1 - p
        if p > 0:
            entries.append((p, components))
    # The matching graph of the graphlike components
    graph = pymatching.Matching()
    edge_obs = {}
    for entry in entries:
        if isinstance(entry, str):
            continue
        p, components = entry
        for dets, obs in components:
            if 1 <= len(dets) <= 2 and 0 < p < 0.5:
                key = tuple(sorted(dets)) if len(dets) == 2 else (dets[0], -1)
                if key not in edge_obs:
                    edge_obs[key] = frozenset(obs)
                    weight = float(np.log((1 - p) / p))
                    if len(dets) == 2:
                        graph.add_edge(dets[0], dets[1], fault_ids=set(obs), weight=weight, merge_strategy="independent")
                    else:
                        graph.add_boundary_edge(dets[0], fault_ids=set(obs), weight=weight, merge_strategy="independent")
    lines, split, mismatched = [], 0, 0
    for entry in entries:
        if isinstance(entry, str):
            lines.append(entry)
            continue
        p, components = entry
        if all(len(dets) <= 2 for dets, _ in components):
            components = _fold_undetectable(components)
            if components:
                lines.append(_format(p, components))
            continue
        new = []
        for dets, obs in components:
            if len(dets) <= 2:
                new.append((dets, obs))
                continue
            if max(dets) >= graph.num_detectors:
                # A detector no graphlike error touches: the component cannot be split, and is left out
                mismatched += 1
                continue
            syndrome = np.zeros(graph.num_detectors, dtype=np.uint8)
            syndrome[dets] = 1
            try:
                edges = graph.decode_to_edges_array(syndrome)
            except ValueError:
                raise ValueError("an error flips an odd number of detectors in a part of the matching graph "
                                 "without a boundary, which matching cannot decode") from None
            flipped = set()
            for u, v in edges:
                key = (min(u, v), max(u, v)) if v >= 0 and u >= 0 else (max(u, v), -1)
                e_obs = edge_obs.get(key, frozenset())
                flipped ^= set(e_obs)
                new.append(([d for d in key if d >= 0], sorted(e_obs)))
            mismatched += flipped != set(obs)
        split += 1
        new = _fold_undetectable(new)
        if new:
            lines.append(_format(p, new))
    return stim.DetectorErrorModel("\n".join(lines)), split, mismatched, (det_frame, obs_frame)


def _matcher(circuit, approximate_disjoint_errors, correlated):
    key = (approximate_disjoint_errors, correlated)
    if _last.get("circuit") is not circuit or _last.get("key") != key:
        dem, split, mismatched, frame = decomposed_dem(circuit, approximate_disjoint_errors)
        matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated)
        _last.clear()
        _last.update(circuit=circuit, key=key, value=(matcher, frame))
    return _last["value"]


def matching_decode(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool, correlated: bool = True):
    matcher, (det_frame, obs_frame) = _matcher(circuit, approximate_disjoint_errors, correlated)
    detection_events, observable_flips = circuit.compile_detector_sampler().sample(num_shots, separate_observables=True)
    # The errors more likely than not are taken to have happened (decomposed_dem)
    predictions = matcher.decode_batch(detection_events ^ det_frame, enable_correlations=correlated) ^ obs_frame
    return int(np.any(predictions != observable_flips, axis=1).sum()) / num_shots


def mwpm_corr(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool):
    return matching_decode(circuit, num_shots, approximate_disjoint_errors, correlated=True)
