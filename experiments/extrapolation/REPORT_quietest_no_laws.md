# Matchmaker evaluation

Extrapolation: the quietest 20% of circuits (error mass per detector) held out and predicted from the rest; scores on the 362 held-out circuits (1085 decodes). Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold. Training options: {'parametric': False}.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 362 | 0% | 100% | 0% | 0% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 200 | 0.74 | 2.21 | 0.40 | 28% |

## Regret (a decoder's LER over the best measured)

518 measured decodes: median error 0.16, p90 0.60.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 1080 | 0.82 | 2.05 | 36% | 0.81 | 37% |
| mwpm | 136 | 0.43 | 0.84 | 56% | 0.77 | 46% |
| bposd | 358 | 0.82 | 1.98 | 35% | 0.74 | 35% |
| bplsd | 114 | 1.24 | 2.38 | 18% | 0.56 | 16% |
| belief_find | 114 | 1.15 | 2.38 | 23% | 0.60 | 15% |
| relay_bp | 196 | 0.50 | 1.85 | 48% | 0.57 | 64% |
| tesseract | 149 | 1.00 | 2.07 | 28% | 0.82 | 34% |
| chromobius | 13 | 2.94 | 3.33 | 0% | 0.85 | 0% |

Timed-out decodes: 0, predicted median above their bound for n/a. Decodes without a prediction (decoder not in the fold's training): 0.

## Decoder choice

On 192 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.35 | 1.00 | 140 |
| recommend, cheapest | 86% | 1.00 | 1.65 | 1.00 | 106 |
| always bposd | 73% | 1.00 | 2.83 | 1.27 | 13 |
| always tesseract | 92% | 1.00 | 1.35 | 1.00 | 140 |

On the 30 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 100% | 1.00 | 1.00 | 1.00 |
| recommend, cheapest | 83% | 1.00 | 2.25 | 1.00 |
| always bposd | 33% | 1.94 | 9.80 | 2.02 |
| always tesseract | 100% | 1.00 | 1.00 | 1.00 |

Chosen by recommend, accurate: tesseract 192.
Chosen by recommend, cheapest: tesseract 141, mwpm 51.

## Per held-out code family (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| Mitten | 4 | 0% | 100% | 2.93 | 3.13 | 6% | 100% |
| Repetition | 12 | 0% | 100% | 2.46 | 2.74 | 21% | 100% |
| Quantum Tanner | 8 | 0% | 100% | 1.66 | 2.45 | 0% | 75% |
| Toric | 9 | 0% | 100% | 1.58 | 2.21 | 33% | 100% |
| Hypergraph product | 9 | 0% | 100% | 1.45 | 2.35 | 18% | 71% |
| Semi-hyperbolic Floquet | 15 | 0% | 100% | 1.44 | 1.64 | 32% | 100% |
| Quantum Reed-Muller | 2 | 0% | 100% | 1.34 | 1.63 | 0% | n/a |
| Surface | 28 | 0% | 100% | 1.25 | 2.01 | 49% | 100% |
| Planar honeycomb | 9 | 0% | 100% | 1.06 | 1.08 | 42% | 67% |
| Honeycomb | 9 | 0% | 100% | 1.05 | 1.20 | 21% | 100% |
| Bivariate bicycle | 68 | 0% | 100% | 0.99 | 1.84 | 37% | 96% |
| Generalized bicycle | 9 | 0% | 100% | 0.90 | 1.11 | 8% | 86% |
| XZZX surface | 9 | 0% | 100% | 0.85 | 1.63 | 39% | 100% |
| Balanced product cyclic | 45 | 0% | 100% | 0.72 | 2.44 | 15% | 100% |
| Heavy-hex | 23 | 0% | 100% | 0.61 | 1.00 | 94% | 89% |
| Bacon-Shor | 14 | 0% | 100% | 0.51 | 1.15 | 84% | 75% |
| Lifted product | 5 | 0% | 100% | 0.47 | 1.69 | 12% | 60% |
| Lift-connected surface | 34 | 0% | 100% | 0.46 | 2.96 | 43% | 97% |
| Color | 22 | 0% | 100% | 0.38 | 1.07 | 55% | 91% |
| Hyperbolic Floquet | 18 | 0% | 100% | 0.35 | 0.87 | 46% | 100% |
| Concatenated Steane | 10 | 0% | 100% | 0.30 | 0.67 | 73% | 100% |
