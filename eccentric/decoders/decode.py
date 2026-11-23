import logging
import stim

from .. import plugins
import numpy as np

from .bp_osd import bposd_chk, bposd_batch
from .mwpm import mwpm
from .chromobius import chromobius
from .ldpc_decoders import bplsd, union_find, belief_find
from .belief_matching import belief_matching
from .sinter_decoders import mwpf, relay_bp, tesseract
from .gari import gari, gari_pp
from .correlated_matching import mwpm_corr
from .hierarchical import hierarchical
from .gpu import GPU_DECODERS, gpu_decoder

DECODERS = {
    "bplsd": bplsd,
    "union_find": union_find,
    "belief_find": belief_find,
    "belief_matching": belief_matching,
    "relay_bp": relay_bp,
    "tesseract": tesseract,
    "mwpf": mwpf,
    "gari": gari,
    "mwpm_corr": mwpm_corr,
}


def decode(
        code_name: str, circuit: stim.Circuit, num_shots: int, decoder: str, backend_name: str, error_type: str,
        distance=None
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
    elif decoder in ("bposd", "bposd_tuned"):
        # bposd_tuned: the same BP-OSD with tuned BP settings (bp_osd.BP)
        bp = "tuned" if decoder == "bposd_tuned" else "paper"
        try:
            if code_name == "steane" or code_name == "bacon":
                return bposd_chk(circuit, num_shots, approximate_disjoint_errors, bp=bp)
            else:
                return bposd_batch(circuit, num_shots, approximate_disjoint_errors, bp=bp)
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
    elif decoder == "gari_pp":
        # GARI++, beta reimplementation (eccentric/decoders/gari): its first stage's setting depends on the code
        try:
            return gari_pp(circuit, num_shots, approximate_disjoint_errors, code_name)
        except Exception as e:
            logging.error(f"gari_pp failed: {e}")
            return None
    elif decoder in ("hier_bposd", "hier_bposd_hard"):
        # Level-by-level decoding of concatenated codes (decoders/hierarchical.py): needs the code's levels
        try:
            return hierarchical(circuit, num_shots, approximate_disjoint_errors, code_name, distance,
                                soft=decoder == "hier_bposd")
        except Exception as e:
            logging.error(f"{decoder} failed: {e}")
            return None
    elif decoder in GPU_DECODERS:
        # GPU versions (decoders/gpu, analysis/GPU.md): same settings as the CPU decoder of the same name without "_gpu"
        try:
            run = gpu_decoder(decoder)
            if decoder in ("bposd_gpu", "bposd_tuned_gpu"):
                return run(circuit, num_shots, approximate_disjoint_errors, chk_variant=code_name in ("steane", "bacon"),
                           bp="tuned" if decoder == "bposd_tuned_gpu" else "paper")
            if decoder in ("hier_bposd_gpu", "hier_bposd_hard_gpu"):
                return run(circuit, num_shots, approximate_disjoint_errors, code_name, distance,
                           soft=decoder == "hier_bposd_gpu")
            if decoder in ("tcbp_jobs_gpu", "tcbp_jobs_sc_gpu"):
                # the code family's TCBP model (decoders/gpu/tcbp_models.py)
                return run(circuit, num_shots, approximate_disjoint_errors, code_name=code_name)
            return run(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"{decoder} failed: {e}")
            return None
    elif decoder in DECODERS or plugins.get("decoders", decoder) is not None:
        run = DECODERS.get(decoder) or plugins.get("decoders", decoder)
        try:
            return run(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"{decoder} failed: {e}")
            return None
    else:
        raise NotImplementedError(f"unknown decoder {decoder}")

def raw_error_rate(circuit: stim.Circuit, num_shots: int):
    sampler = circuit.compile_detector_sampler()
    _, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    errors = np.any(observable_flips, axis=1)
    num_errors = np.count_nonzero(errors)

    return num_errors
