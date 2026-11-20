import stim
import numpy as np
import chromobius as chromobius_lib

# The last circuit's compiled decoder: a sweep decodes one circuit batch by batch, and compiling the decoder can
# cost more than decoding a batch (as for mwpm and BP-OSD)
_last = {}


def _decoder(circuit: stim.Circuit, approximate_disjoint_errors: bool):
    if _last.get("circuit") is not circuit or _last.get("approximate") != approximate_disjoint_errors:
        _last.clear()
        _last.update(circuit=circuit, approximate=approximate_disjoint_errors,
                     decoder=chromobius_lib.compile_decoder_for_dem(
                         circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)))
    return _last["decoder"]


def chromobius(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool):
    decoder = _decoder(circuit, approximate_disjoint_errors)
    dets, actual_obs_flips = circuit.compile_detector_sampler().sample(
        shots=num_shots,
        separate_observables=True,
        bit_packed=True,
    )
    predicted_obs_flips = decoder.predict_obs_flips_from_dets_bit_packed(dets)
    return np.count_nonzero(np.any(predicted_obs_flips != actual_obs_flips, axis=1)) / num_shots
