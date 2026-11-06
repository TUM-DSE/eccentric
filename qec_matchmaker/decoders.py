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
}


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
