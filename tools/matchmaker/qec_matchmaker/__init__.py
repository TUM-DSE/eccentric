"""Predict how decoding a QEC circuit will go, and pick a decoder, before decoding it."""
from .features import FEATURE_NAMES, FEATURE_VERSION, features_from_circuit, features_from_dem, features_from_matrices

__all__ = ["FEATURE_NAMES", "FEATURE_VERSION", "features_from_circuit", "features_from_dem", "features_from_matrices"]
__version__ = "0.1.0"
