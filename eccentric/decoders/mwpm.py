import pymatching
import stim
import numpy as np

# The last circuit's matcher: a sweep decodes one circuit batch by batch, and building the error model and the
# matching graph can cost more than decoding a batch (as for mwpm_corr and BP-OSD)
_last = {}


def _matcher(circuit: stim.Circuit, approximate_disjoint_errors: bool) -> pymatching.Matching:
    if _last.get("circuit") is not circuit or _last.get("approximate") != approximate_disjoint_errors:
        # PyMatching drops errors with more than two detectors, so they must be decomposed into edges
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=approximate_disjoint_errors)
        _last.clear()
        _last.update(circuit=circuit, approximate=approximate_disjoint_errors,
                     matcher=pymatching.Matching.from_detector_error_model(dem))
    return _last["matcher"]


def mwpm(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool):
    matcher = _matcher(circuit, approximate_disjoint_errors)
    detection_events, observable_flips = circuit.compile_detector_sampler().sample(num_shots, separate_observables=True)
    predictions = matcher.decode_batch(detection_events)
    # A shot fails when any observable is predicted wrong
    return int(np.any(predictions != observable_flips, axis=1).sum()) / num_shots
