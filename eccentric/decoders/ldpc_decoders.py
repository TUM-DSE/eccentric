import numpy as np
import stim
from ldpc import BeliefFindDecoder, BpLsdDecoder, UnionFindDecoder

from .bp_osd import dem_to_check_matrices

# BP settings shared with bposd_batch, so that only the post-processing differs


def _decode_shots(decoder, circuit: stim.Circuit, num_shots: int, obs) -> float:
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    num_errors = 0
    for shot in range(num_shots):
        predicted_for_shot = (obs @ decoder(detection_events[shot])) % 2
        if not np.array_equal(observable_flips[shot], predicted_for_shot):
            num_errors += 1
    return num_errors / num_shots


def _check_matrices(circuit: stim.Circuit, approximate_disjoint_errors: bool):
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    return dem_to_check_matrices(dem, xor_priors=True)


def bplsd(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    chk, obs, priors = _check_matrices(circuit, approximate_disjoint_errors)
    decoder = BpLsdDecoder(
        chk,
        error_channel=list(priors),
        max_iter=10000,
        bp_method="minimum_sum",
        ms_scaling_factor=1.0,
        lsd_method="lsd_cs",
        lsd_order=min(7, chk.shape[1] - chk.shape[0]),
    )
    return _decode_shots(decoder.decode, circuit, num_shots, obs)


def union_find(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    chk, obs, priors = _check_matrices(circuit, approximate_disjoint_errors)
    # Inversion works for any check matrix, peeling only for matchable ones
    decoder = UnionFindDecoder(chk, uf_method="inversion")
    priors = np.clip(priors, 1e-12, 1 - 1e-12)
    llrs = np.log((1 - priors) / priors)
    return _decode_shots(lambda s: decoder.decode(s, llrs=llrs), circuit, num_shots, obs)


def belief_find(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    chk, obs, priors = _check_matrices(circuit, approximate_disjoint_errors)
    decoder = BeliefFindDecoder(
        chk,
        error_channel=list(priors),
        max_iter=10000,
        bp_method="minimum_sum",
        ms_scaling_factor=1.0,
        uf_method="inversion",
    )
    return _decode_shots(decoder.decode, circuit, num_shots, obs)
