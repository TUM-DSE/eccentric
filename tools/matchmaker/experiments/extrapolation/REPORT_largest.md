# Matchmaker evaluation

Extrapolation: the largest 20% of circuits (detectors) held out and predicted from the rest; scores on the 358 held-out circuits (1212 decodes). Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 358 | 43% | 93% | 95% | 90% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 245 | 0.10 | 0.56 | 0.91 | 87% |

## Regret (a decoder's LER over the best measured)

845 measured decodes: median error 0.04, p90 0.40.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 1112 | 0.21 | 0.68 | 83% | 0.96 | 71% |
| mwpm | 46 | 0.13 | 0.36 | 100% | 0.97 | 80% |
| bposd | 319 | 0.22 | 0.60 | 83% | 0.95 | 61% |
| bplsd | 151 | 0.19 | 0.48 | 90% | 0.97 | 57% |
| belief_find | 201 | 0.13 | 0.49 | 90% | 0.95 | 71% |
| relay_bp | 268 | 0.24 | 1.52 | 77% | 0.88 | 88% |
| tesseract | 127 | 0.32 | 0.69 | 67% | 0.96 | 75% |

Timed-out decodes: 10, predicted median above their bound for 40%. Decodes without a prediction (decoder not in the fold's training): 0.

## Decoder choice

On 223 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 74% | 1.00 | 1.67 | 1.00 | 59 |
| recommend, cheapest | 72% | 1.00 | 1.72 | 1.00 | 38 |
| always bposd | 83% | 1.00 | 1.48 | 2.70 | 40 |
| always tesseract | 93% | 1.00 | 1.03 | 2.70 | 167 |

On the 36 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 69% | 1.01 | 1.77 | 1.00 |
| recommend, cheapest | 58% | 1.01 | 2.98 | 1.00 |
| always bposd | 58% | 1.01 | 1.61 | 2.21 |
| always tesseract | 97% | 1.00 | 1.01 | 7.95 |

Chosen by recommend, accurate: relay_bp 132, tesseract 74, mwpm 17.
Chosen by recommend, cheapest: relay_bp 132, tesseract 41, bposd 27, mwpm 23.

## Per held-out code family (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| Concatenated Steane | 3 | 0% | 100% | 1.26 | 1.26 | 50% | n/a |
| Color | 13 | 8% | 92% | 0.72 | 1.47 | 70% | n/a |
| Mitten | 3 | 100% | 100% | 0.41 | 0.41 | 42% | 100% |
| Generalized bicycle | 2 | 100% | 100% | 0.41 | 0.41 | 62% | 100% |
| Planar honeycomb | 2 | 0% | 50% | 0.18 | 0.18 | 75% | 100% |
| Heavy-hex | 15 | 7% | 93% | 0.18 | 0.54 | 77% | n/a |
| Honeycomb | 2 | 0% | 100% | 0.17 | 0.17 | 100% | 100% |
| Semi-hyperbolic Floquet | 25 | 28% | 100% | 0.17 | 0.97 | 82% | 69% |
| Surface | 19 | 0% | 100% | 0.16 | 0.25 | 50% | 100% |
| Lift-connected surface | 6 | 83% | 67% | 0.14 | 0.20 | 100% | 17% |
| Hyperbolic Floquet | 13 | 69% | 100% | 0.13 | 0.30 | 86% | 82% |
| Hypergraph product | 22 | 73% | 95% | 0.09 | 0.43 | 87% | 84% |
| Bivariate bicycle | 116 | 43% | 92% | 0.08 | 0.61 | 76% | 73% |
| Bacon-Shor | 13 | 54% | 69% | 0.08 | 0.09 | 100% | n/a |
| Balanced product cyclic | 104 | 51% | 95% | 0.04 | 0.24 | 91% | 71% |
