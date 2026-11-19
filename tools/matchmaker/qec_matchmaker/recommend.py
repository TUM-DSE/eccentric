"""Choosing a decoder for one circuit from the models' predictions (`ModelBundle.predict`).

Objectives: `accurate` (lowest predicted LER), `cheapest` (lowest predicted time among the decoders
predicted within `tolerance` of the lowest LER), `budget` (lowest predicted LER among the decoders whose
median time fits `budget_ms` per shot, or per round when `rounds` is given; `budget_per_round` requires
it). Only applicable decoders are candidates. A circuit likely above threshold gets the cheapest decoder
and a warning, since there the decoders' LERs all approach 0.5.
"""
from dataclasses import dataclass, field

import pandas as pd

OBJECTIVES = ("accurate", "cheapest", "budget", "budget_per_round")


@dataclass
class Recommendation:
    decoder: str | None
    objective: str
    reason: str
    warnings: list = field(default_factory=list)
    table: pd.DataFrame = field(default=None, repr=False)  # the candidates, best first


def recommend(predictions: pd.DataFrame, objective: str = "cheapest", budget_ms: float | None = None,
              rounds: int | None = None, tolerance: float = 1.25, confidence: float = 0.8) -> Recommendation:
    """The decoder to use on one circuit, given its rows of `ModelBundle.predict`. For "cheapest", a decoder is
    eligible if its regret (its LER over the best decoder's) is predicted within the tolerance with the given
    confidence: the regret quantile at that level (0.5, 0.7, 0.8 or 0.9), not the median, must be within it"""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}")
    if "circuit_key" in predictions.columns and predictions.circuit_key.nunique() > 1:
        raise ValueError("predictions of a single circuit expected")
    if objective.startswith("budget") and budget_ms is None:
        raise ValueError(f"objective {objective} needs budget_ms")
    if objective == "budget_per_round" and not rounds:
        raise ValueError("objective budget_per_round needs rounds")

    warnings = []
    fits = predictions[predictions.applicable.astype(bool)]
    candidates = fits.dropna(subset=["ler_decoder_q50", "ms_q50"])
    if len(candidates) < len(fits):
        missing = ", ".join(sorted(set(fits.decoder) - set(candidates.decoder)))
        warnings.append(f"left out for lack of predictions (not seen in training): {missing}")
    if candidates.empty:
        return Recommendation(None, objective, "no applicable decoder with predictions", warnings, candidates)

    p_above = float(candidates.p_above.iloc[0])
    if p_above > 0.5:
        table = candidates.sort_values(["ms_q50", "ler_decoder_q50"])
        top = table.iloc[0]
        warnings.append(f"likely above threshold (P = {p_above:.2f}): every decoder's LER will be near 0.5, so the "
                        "choice of decoder will not matter")
        return Recommendation(top.decoder, objective, f"cheapest applicable decoder ({top.ms_q50:.3g} ms per shot), "
                              "as the circuit is likely above threshold", warnings, table)

    if objective == "accurate":
        table = candidates.sort_values(["ler_decoder_q50", "ms_q50"])
        top = table.iloc[0]
        reason = f"lowest predicted LER ({top.ler_decoder_q50:.2g}; {top.ms_q50:.3g} ms per shot)"
    elif objective == "cheapest":
        lowest = candidates.ler_decoder_q50.min()
        column = f"regret_q{round(100 * confidence)}"
        if column not in candidates.columns:
            raise ValueError(f"confidence must be one of the predicted regret quantiles, got {confidence}")
        # The lowest predicted LER is the best decoder's times the smallest median regret
        pessimistic = candidates.ler_q50 * candidates[column]
        table = candidates.assign(eligible=pessimistic <= tolerance * lowest).sort_values(
            ["eligible", "ms_q50", "ler_decoder_q50"], ascending=[False, True, True])
        top = table.iloc[0]
        reason = (f"cheapest ({top.ms_q50:.3g} ms per shot) of the decoders predicted within {tolerance:g}x of the "
                  f"lowest LER ({lowest:.2g}) with {confidence:.0%} confidence; its predicted LER is "
                  f"{top.ler_decoder_q50:.2g}")
    else:
        unit = "round" if rounds else "shot"
        cost = candidates.ms_q50 / rounds if rounds else candidates.ms_q50
        table = candidates.assign(cost_ms=cost, eligible=cost <= budget_ms)
        if table.eligible.any():
            table = table.sort_values(["eligible", "ler_decoder_q50", "cost_ms"], ascending=[False, True, True])
            top = table.iloc[0]
            reason = (f"lowest predicted LER ({top.ler_decoder_q50:.2g}) of the decoders predicted within "
                      f"{budget_ms:g} ms per {unit} ({top.cost_ms:.3g} ms)")
        else:
            table = table.sort_values(["cost_ms", "ler_decoder_q50"])
            top = table.iloc[0]
            reason = f"fastest decoder ({top.cost_ms:.3g} ms per {unit}); none is predicted to fit the budget"
            warnings.append(f"no applicable decoder is predicted to fit {budget_ms:g} ms per {unit}")
    return Recommendation(top.decoder, objective, reason, warnings, table)
