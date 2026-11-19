# Matchmaker evaluation

Extrapolation: the quietest 20% of circuits (error mass per detector) held out and predicted from the rest; scores on the 362 held-out circuits (1085 decodes). Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 362 | 0% | 100% | 0% | 0% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 200 | 0.30 | 1.06 | 0.80 | 50% |

## Regret (a decoder's LER over the best measured)

518 measured decodes: median error 0.16, p90 0.60.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 1080 | 0.60 | 1.55 | 43% | 0.87 | 52% |
| mwpm | 136 | 0.31 | 0.74 | 70% | 0.77 | 70% |
| bposd | 358 | 0.65 | 1.61 | 42% | 0.81 | 34% |
| bplsd | 114 | 0.79 | 1.67 | 32% | 0.72 | 49% |
| belief_find | 114 | 0.84 | 1.64 | 31% | 0.76 | 42% |
| relay_bp | 196 | 0.71 | 1.71 | 37% | 0.58 | 58% |
| tesseract | 149 | 0.49 | 1.19 | 49% | 0.93 | 81% |
| chromobius | 13 | 0.79 | 1.32 | 31% | -0.40 | 38% |

Timed-out decodes: 0, predicted median above their bound for n/a. Decodes without a prediction (decoder not in the fold's training): 0.

## Decoder choice

On 192 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.35 | 1.00 | 140 |
| recommend, cheapest | 89% | 1.00 | 1.48 | 1.00 | 109 |
| always bposd | 73% | 1.00 | 2.83 | 1.27 | 13 |
| always tesseract | 92% | 1.00 | 1.35 | 1.00 | 140 |

On the 30 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 100% | 1.00 | 1.00 | 1.00 |
| recommend, cheapest | 93% | 1.00 | 1.00 | 1.00 |
| always bposd | 33% | 1.94 | 9.80 | 2.02 |
| always tesseract | 100% | 1.00 | 1.00 | 1.00 |

Chosen by recommend, accurate: tesseract 192.
Chosen by recommend, cheapest: tesseract 150, mwpm 42.

## Per held-out code family (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| Repetition | 12 | 0% | 100% | 1.93 | 2.11 | 50% | 100% |
| Quantum Reed-Muller | 2 | 0% | 100% | 1.05 | 1.26 | 0% | n/a |
| Semi-hyperbolic Floquet | 15 | 0% | 100% | 1.05 | 1.08 | 55% | 100% |
| Toric | 9 | 0% | 100% | 0.98 | 1.42 | 50% | 100% |
| Mitten | 4 | 0% | 100% | 0.94 | 0.95 | 6% | 100% |
| Quantum Tanner | 8 | 0% | 100% | 0.87 | 1.32 | 43% | 75% |
| Surface | 28 | 0% | 100% | 0.74 | 1.08 | 47% | 100% |
| Planar honeycomb | 9 | 0% | 100% | 0.47 | 0.58 | 50% | 67% |
| XZZX surface | 9 | 0% | 100% | 0.41 | 0.87 | 67% | 100% |
| Hypergraph product | 9 | 0% | 100% | 0.32 | 1.65 | 30% | 71% |
| Bivariate bicycle | 68 | 0% | 100% | 0.30 | 0.75 | 32% | 96% |
| Heavy-hex | 23 | 0% | 100% | 0.28 | 0.71 | 85% | 89% |
| Concatenated Steane | 10 | 0% | 100% | 0.26 | 0.45 | 82% | 100% |
| Bacon-Shor | 14 | 0% | 100% | 0.23 | 0.59 | 80% | 75% |
| Hyperbolic Floquet | 18 | 0% | 100% | 0.20 | 0.70 | 56% | 100% |
| Generalized bicycle | 9 | 0% | 100% | 0.18 | 0.48 | 22% | 86% |
| Color | 22 | 0% | 100% | 0.17 | 0.39 | 71% | 91% |
| Balanced product cyclic | 45 | 0% | 100% | 0.16 | 1.05 | 27% | 100% |
| Lift-connected surface | 34 | 0% | 100% | 0.15 | 0.64 | 50% | 97% |
| Lifted product | 5 | 0% | 100% | 0.14 | 0.54 | 44% | 60% |
| Honeycomb | 9 | 0% | 100% | 0.13 | 0.42 | 42% | 100% |
