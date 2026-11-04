"""advise(): predictions and a decoder recommendation for one circuit."""
import json
import os
from dataclasses import dataclass, field

import pandas as pd
import stim

from .features import FEATURE_NAMES, features_from_circuit, features_from_dem

# The trained models ship inside the package; QEC_MATCHMAKER_MODELS points elsewhere
MODEL_DIR = os.environ.get("QEC_MATCHMAKER_MODELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))


def features_of(source) -> dict:
    """Features of a stim.Circuit, stim.DetectorErrorModel, features dict, or a file: .stim (circuit), .dem,
    or .h5/.hdf5 (decoder-bench trace)"""
    if isinstance(source, dict):
        return source
    if isinstance(source, stim.Circuit):
        return features_from_circuit(source)
    if isinstance(source, stim.DetectorErrorModel):
        return features_from_dem(source)
    path = str(source)
    if path.endswith((".h5", ".hdf5")):
        from .io.decoder_bench import features_from_trace
        return features_from_trace(path)
    if path.endswith(".dem"):
        return features_from_dem(stim.DetectorErrorModel.from_file(path))
    return features_from_circuit(stim.Circuit.from_file(path))


@dataclass
class Advice:
    features: dict
    predictions: pd.DataFrame
    recommendation: object
    warnings: list = field(default_factory=list)

    @property
    def decoder(self) -> str:
        return self.recommendation.decoder

    @property
    def above_threshold(self) -> float:
        return float(self.predictions.p_above.iloc[0])

    @property
    def ler(self) -> tuple:
        row = self.predictions.iloc[0]
        return float(row.ler_q10), float(row.ler_q50), float(row.ler_q90)

    @property
    def time_per_shot_ms(self) -> dict:
        return dict(zip(self.predictions.decoder, self.predictions.ms_q50))

    def expected_cost(self, errors: int = 100) -> pd.DataFrame:
        """Shots and core-hours each decoder needs to see `errors` logical errors, from the predictions"""
        p = self.predictions
        shots = errors / p.ler_decoder_q50.clip(lower=1e-12)
        return pd.DataFrame({"decoder": p.decoder, "shots": shots.round(), "core_hours": shots * p.ms_q50 / 3.6e6,
                             "applicable": p.applicable})

    def summary(self) -> str:
        lo, mid, hi = self.ler
        lines = [f"above threshold: {self.above_threshold:.0%}",
                 f"logical error rate (best decoder): {mid:.2e} (80% interval {lo:.1e} - {hi:.1e})",
                 f"recommended decoder: {self.decoder} ({self.recommendation.reason})"]
        lines += [f"warning: {w}" for w in self.warnings + list(self.recommendation.warnings)]
        table = self.predictions[["decoder", "applicable", "ler_decoder_q50", "ms_q50"]].rename(
            columns={"ler_decoder_q50": "LER", "ms_q50": "ms/shot"})
        return "\n".join(lines) + "\n\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3g}")

    def to_dict(self) -> dict:
        return {"features": self.features, "decoder": self.decoder, "above_threshold": self.above_threshold,
                "ler": self.ler, "warnings": self.warnings + list(self.recommendation.warnings),
                "reason": self.recommendation.reason,
                "predictions": json.loads(self.predictions.to_json(orient="records"))}


def advise(source, objective: str = "cheapest", budget_ms: float = None, per_round: bool = False,
           model_dir: str = None, speed_factors: dict = None, tolerance: float = 1.25) -> Advice:
    """Predict how decoding `source` will go and recommend a decoder. Objectives: "accurate", "cheapest"
    (cheapest decoder within `tolerance` of the most accurate) and "budget" (most accurate within
    `budget_ms` per shot, or per round with per_round=True). `speed_factors` ({decoder: factor}, see
    qec_matchmaker.calibrate) scales the predicted times to another machine"""
    from .models import ModelBundle
    from .recommend import recommend

    features = features_of(source)
    bundle = ModelBundle.load(model_dir or MODEL_DIR)
    frame = pd.DataFrame([{name: features.get(name) for name in FEATURE_NAMES}], dtype=float)
    frame["circuit_key"] = "input"
    predictions = bundle.predict(frame)
    warnings = []
    if speed_factors:
        for col in ("ms_q10", "ms_q50", "ms_q90"):
            predictions[col] *= predictions.decoder.map(speed_factors).fillna(1.0)
    else:
        warnings.append("times are for one core of the reference machine; calibrate for yours (qec_matchmaker.calibrate)")
    unknown = sorted(set(predictions.decoder[predictions.applicable.isna()]))
    if unknown:
        warnings.append(f"cannot tell from these features whether {', '.join(unknown)} applies (pass the circuit, "
                        "not only its error model); left out")
    outside = predictions.outside_training.iloc[0]
    if outside:
        warnings.append(f"outside the training data, predictions extrapolate: {outside}")
    rounds = features.get("rounds") if per_round else None
    choice = recommend(predictions, objective=objective, budget_ms=budget_ms, rounds=rounds, tolerance=tolerance)
    return Advice(features, predictions, choice, warnings)
