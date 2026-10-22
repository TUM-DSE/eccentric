import numpy as np
import stim
from beliefmatching import BeliefMatching


def belief_matching(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    # BP followed by MWPM with the BP marginals as edge weights (Higgott et al., arXiv:2203.04948)
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    dem = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=approximate_disjoint_errors
    )
    matcher = BeliefMatching(dem, max_bp_iters=20)
    predictions = matcher.decode_batch(detection_events)
    return np.count_nonzero(np.any(predictions != observable_flips, axis=1)) / num_shots
