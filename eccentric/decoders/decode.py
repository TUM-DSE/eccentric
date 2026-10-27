import logging
import stim
import numpy as np

from .bp_osd import bposd_chk, bposd_batch
from .mwpm import mwpm
from .chromobius import chromobius
from .ldpc_decoders import bplsd, union_find, belief_find
from .belief_matching import belief_matching
from .sinter_decoders import relay_bp, tesseract

DECODERS = {
    "bplsd": bplsd,
    "union_find": union_find,
    "belief_find": belief_find,
    "belief_matching": belief_matching,
    "relay_bp": relay_bp,
    "tesseract": tesseract,
}


def decode(
        code_name: str, circuit: stim.Circuit, num_shots: int, decoder: str, backend_name: str, error_type: str
) -> float:
    approximate_disjoint_errors = False
    if backend_name.split("_")[0] in ("variance", "real", "native") and (error_type == "variance" or error_type.split("_")[0] == "real"):
        approximate_disjoint_errors = True

    if decoder == "mwpm":
        try:
            return mwpm(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"MWPM failed: {e}")
    elif decoder == "chromobius":
        try:
            return chromobius(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"Chromobius failed: {e}")
    elif decoder == "bposd":
        try:
            if code_name == "steane" or code_name == "bacon":
                return bposd_chk(circuit, num_shots, approximate_disjoint_errors)
            else:
                return bposd_batch(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"BP-OSD failed: {e}")
            return None
    elif decoder == "bposd_chk":
        try:
            return bposd_chk(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"BP-OSD CHK failed: {e}")
            return None
    elif decoder == "bposd_batch":
        try:
            return bposd_batch(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"BP-OSD BATCH failed: {e}")
            return None
    elif decoder in DECODERS:
        try:
            return DECODERS[decoder](circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"{decoder} failed: {e}")
            return None
    else:
        raise NotImplementedError

def raw_error_rate(circuit: stim.Circuit, num_shots: int):
    sampler = circuit.compile_detector_sampler()
    _, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    errors = np.any(observable_flips, axis=1)
    num_errors = np.count_nonzero(errors)

    return num_errors
