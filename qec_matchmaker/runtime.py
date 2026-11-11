"""Predict, pilot, commit: decoding one circuit with Matchmaker's predictions checked as it goes.

    outcome = execute(predictions, decode, objective="cheapest")

`predictions` are one circuit's rows of `ModelBundle.predict`; `decode(decoder, shots, seconds)` runs a decoder
on the circuit and returns a `Batch`: with shots and seconds given it is a pilot (at most that many shots,
stopped after that many seconds), with both None it is the full decode to the caller's own error target,
continuing from the pilot.

1. Skip the circuit if it is predicted above threshold with probability over `skip_above`.
2. Pilot the recommended decoder: its first `pilot_shots` shots, with `slack` times the predicted time (its 90%
   quantile) as a cap, at most `max_cap` seconds. A decoder that does not finish its pilot is slower than
   predicted: switch to the next candidate (at most `switches` times), having paid only the cap, not a timeout.
3. If more than `above` of the pilot's shots are logical errors, the circuit looks above threshold. A weak
   decoder can fail where a better one works, so the circuit is given up as "above" only when the reference
   fails its pilot too: the decoder predicted most accurate among those not yet found too slow (a pilot that
   does not finish passes the reference to the next most accurate), and, when the reference stands in for a
   more accurate decoder, another decoder has failed as well. After a failure the reference is piloted next. No decoder is piloted twice, and only the pilots were paid. (Counting any two failures instead gave
   up useful circuits near threshold once two cheap decoders of one family, GARI and GARI++, were candidates.)
4. Otherwise commit: the full decode with the piloted decoder.

The candidates are `recommend`'s table in its order (for "cheapest": the eligible decoders by predicted time,
then the others), so a switch goes to the next best choice under the same objective.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from .recommend import recommend


@dataclass
class Batch:
    shots: int
    errors: int
    seconds: float
    # False if the decoder was stopped at its time cap before the shots were done
    complete: bool = True


@dataclass
class Outcome:
    # "decoded", "skipped" (predicted above threshold), "above" (the pilot found it above threshold),
    # "gave up" (no candidate finished its pilot)
    status: str
    decoder: Optional[str] = None
    result: Optional[Batch] = None
    seconds: float = 0.0
    # (step, decoder, seconds) of everything that ran
    trace: list = field(default_factory=list)


def execute(predictions: pd.DataFrame, decode: Callable[[str, Optional[int], Optional[float]], Optional[Batch]],
            objective: str = "cheapest", *, skip_above: float = 0.9, pilot_shots: int = 100, slack: float = 10.0,
            min_cap: float = 1.0, max_cap: float = 1800.0, above: float = 0.5, switches: int = 3, **recommend_options) -> Outcome:
    """Decode one circuit as described in the module's docstring. `decode` may return None for a decoder it
    cannot run, which is passed over without counting as a switch"""
    choice = recommend(predictions, objective, **recommend_options)
    if choice.decoder is None:
        return Outcome("gave up")
    table = choice.table
    if float(table.p_above.iloc[0]) > skip_above:
        return Outcome("skipped")
    outcome = Outcome("gave up")

    def run_pilot(row):
        cap = min(max_cap, max(min_cap, slack * pilot_shots * row.ms_q90 / 1000))
        pilot = decode(row.decoder, pilot_shots, cap)
        if pilot is not None:
            outcome.seconds += pilot.seconds
            outcome.trace.append(("pilot", row.decoder, pilot.seconds))
        return pilot

    failed = lambda pilot: pilot.shots and pilot.errors / pilot.shots > above
    by_accuracy = list(table.sort_values("ler_decoder_q50", kind="stable").decoder)
    unusable = set()  # piloted too slowly, or not runnable

    def reference():
        return next((d for d in by_accuracy if d not in unusable), None)

    queue = list(table.itertuples())
    tried, seen_failure, failures = 0, False, 0
    while queue and tried <= switches:
        row = queue.pop(0)
        pilot = run_pilot(row)
        if pilot is None:
            unusable.add(row.decoder)
            continue
        tried += 1
        if not pilot.complete:
            unusable.add(row.decoder)
            if seen_failure:
                queue.sort(key=lambda r: r.decoder != reference())
            continue
        if failed(pilot):
            # A weak decoder can fail where a better one works: the circuit is above threshold only if the
            # reference (the most accurate decoder not found too slow) fails too, and, when the reference stands in
            # for a more accurate decoder that was too slow, one more decoder has failed. The reference goes next
            failures += 1
            if row.decoder == reference() and (row.decoder == by_accuracy[0] or failures >= 2):
                outcome.status, outcome.decoder, outcome.result = "above", row.decoder, pilot
                return outcome
            unusable.add(row.decoder)
            seen_failure = True
            queue.sort(key=lambda r: r.decoder != reference())
            continue
        full = decode(row.decoder, None, None)
        # The full decode continues from the pilot: its time includes the pilot's
        outcome.seconds += full.seconds - pilot.seconds
        outcome.trace.append(("decode", row.decoder, full.seconds - pilot.seconds))
        outcome.status, outcome.decoder, outcome.result = ("decoded" if full.complete else "gave up"), row.decoder, full
        return outcome
    return outcome
