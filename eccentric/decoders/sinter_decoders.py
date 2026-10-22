import numpy as np
import sinter
import stim

# Decoders that implement sinter's Decoder interface, run with their libraries' default settings


def _decode(decoder: sinter.Decoder, circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    dets, actual_obs_flips = circuit.compile_detector_sampler().sample(
        shots=num_shots,
        separate_observables=True,
        bit_packed=True,
    )
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    predicted_obs_flips = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    return np.count_nonzero(np.any(predicted_obs_flips != actual_obs_flips, axis=1)) / num_shots


def relay_bp(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    # Relay-BP (Müller et al., arXiv:2506.01779) with the parameters of the paper
    from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP
    return _decode(SinterDecoder_RelayBP(), circuit, num_shots, approximate_disjoint_errors)


def tesseract(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    # Tesseract most-likely-error search decoder (https://github.com/quantumlib/tesseract-decoder)
    from tesseract_decoder import make_tesseract_sinter_decoders_dict
    return _decode(make_tesseract_sinter_decoders_dict()["tesseract"], circuit, num_shots, approximate_disjoint_errors)
