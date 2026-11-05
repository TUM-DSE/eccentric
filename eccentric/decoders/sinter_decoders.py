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
    # Relay-BP (Müller et al., arXiv:2506.01779) with the parameters of the paper. Not through its sinter
    # decode_shots_bit_packed: that unpacks the detection events without their count, so a circuit whose number
    # of detectors is not a multiple of 8 gets up to 7 extra syndrome bits, and Relay-BP then rarely converges
    # (bivariate bicycle d = 6, 252 detectors: 37x slower and LER 0.120 instead of 0.095)
    from relay_bp.stim.sinter.decoders import SinterDecoder_RelayBP
    dets, actual_obs_flips = circuit.compile_detector_sampler().sample(shots=num_shots, separate_observables=True)
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    compiled = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=dem)
    matrices = compiled.check_matrices
    syndromes = dets.astype(np.uint8)
    if matrices.syndrome_bias is not None:
        syndromes = (syndromes + matrices.syndrome_bias) % 2
    predicted = compiled.observable_decoder.decode_observables_batch(syndromes, parallel=False)
    if matrices.observables_bias is not None:
        predicted = (predicted + matrices.observables_bias) % 2
    return np.count_nonzero(np.any(predicted.astype(bool) != actual_obs_flips, axis=1)) / num_shots


def tesseract(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool) -> float:
    # Tesseract most-likely-error search decoder (https://github.com/quantumlib/tesseract-decoder)
    from tesseract_decoder import make_tesseract_sinter_decoders_dict
    return _decode(make_tesseract_sinter_decoders_dict()["tesseract"], circuit, num_shots, approximate_disjoint_errors)
