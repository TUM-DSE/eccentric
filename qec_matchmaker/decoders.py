"""The decoders Matchmaker knows (named as in ECCentric), and when each one applies to a circuit."""
import math

DECODERS = {
    "mwpm": "minimum-weight perfect matching (PyMatching)",
    "mwpf": "hypergraph minimum-weight parity factor (MWPF)",
    "bposd": "BP with ordered-statistics decoding (ldpc)",
    "bposd_tuned": "BP-OSD with tuned BP: 100 iterations, min-sum scaling 0.625",
    "bplsd": "BP with localized-statistics decoding (ldpc)",
    "belief_find": "BP with union-find (ldpc)",
    "relay_bp": "Relay-BP",
    "tesseract": "Tesseract search decoder",
    "chromobius": "Chromobius color-code decoder",
    "gari": "GARI: normalized min-sum on the GARI-transformed correlated error model (CSS circuits only)",
    "gari_pp": "GARI++ (ECCentric's beta reimplementation): short GARI plus union-find on its residual",
}


# Decoders that refuse circuits for reasons no feature shows (GARI needs CSS detectors and observables on one side,
# which Floquet, subsystem and XZZX circuits break): whether they apply is learned from the decodes that refused
# ("decoder not applicable" in ECCentric). A decoder maps to the decoder whose refusals train its model: GARI++
# was only run where GARI applies, so it shares GARI's
LEARNED_APPLICABILITY = {"gari": "gari", "gari_pp": "gari"}


def _flag(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else bool(value)


def applicable(features: dict) -> dict:
    """Which decoders can decode a circuit with these features: True, False, or None when the features
    cannot tell. Matching needs errors that decompose into edges, which only the circuit shows (graph-like
    errors in a DEM settle it in favour); Chromobius needs color-annotated detectors"""
    decomposable = _flag(features.get("decomposable"))
    if decomposable is None and _flag(features.get("graphlike")):
        decomposable = True
    color = _flag(features.get("color_annotated"))
    return {name: decomposable if name == "mwpm" else color if name == "chromobius" else True for name in DECODERS}
