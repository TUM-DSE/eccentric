# Matchmaker evaluation

Extrapolation: the largest distance of every series of distances held out and predicted from the rest; scores on the 112 held-out circuits (268 decodes). Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 112 | 3% | 93% | 0% | 0% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 78 | 0.18 | 0.51 | 0.96 | 68% |

## Regret (a decoder's LER over the best measured)

159 measured decodes: median error 0.06, p90 0.29.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 244 | 0.17 | 0.54 | 87% | 0.98 | 76% |
| mwpm | 69 | 0.13 | 0.33 | 99% | 0.96 | 80% |
| bposd | 110 | 0.15 | 0.65 | 79% | 0.95 | 74% |
| relay_bp | 22 | 0.18 | 0.34 | 95% | 0.82 | 100% |
| tesseract | 39 | 0.22 | 0.39 | 92% | 0.97 | 67% |
| chromobius | 4 | 0.80 | 1.02 | 25% | 0.20 | 25% |

Timed-out decodes: 0, predicted median above their bound for n/a. Decodes without a prediction (decoder not in the fold's training): 0.

## Decoder choice

On 63 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 79% | 1.00 | 1.70 | 5.48 | 40 |
| recommend, cheapest | 83% | 1.00 | 1.52 | 1.00 | 25 |
| always bposd | 57% | 1.14 | 2.51 | 1221.60 | 1 |
| always tesseract | 79% | 1.00 | 1.70 | 284.10 | 42 |

On the 0 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|

Chosen by recommend, accurate: tesseract 59, relay_bp 3, mwpm 1.
Chosen by recommend, cheapest: tesseract 37, mwpm 14, bposd 7, relay_bp 5.

## Per held-out code family (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| Toric | 4 | 0% | 100% | 0.58 | 0.73 | 100% | 100% |
| XZZX surface | 4 | 0% | 100% | 0.29 | 0.30 | 100% | 100% |
| Planar honeycomb | 15 | 0% | 87% | 0.24 | 0.39 | 98% | 100% |
| Surface | 14 | 0% | 100% | 0.22 | 0.43 | 81% | 71% |
| Heavy-hex | 14 | 0% | 100% | 0.17 | 0.94 | 74% | 100% |
| Color | 14 | 7% | 93% | 0.17 | 0.51 | 53% | 0% |
| Bacon-Shor | 14 | 7% | 93% | 0.12 | 0.62 | 84% | 50% |
| Honeycomb | 15 | 0% | 93% | 0.11 | 0.18 | 95% | 85% |
| Concatenated Steane | 14 | 7% | 79% | 0.07 | 0.49 | 94% | 100% |
| Repetition | 4 | 0% | 100% | n/a | n/a | 100% | n/a |
