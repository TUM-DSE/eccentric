# Matchmaker evaluation

Held-out code familys: leave-one-family-out, 21 folds over 21 code familys (all circuits of one are held out together); 1810 circuits, 6673 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 1810 | 34% | 94% | 91% | 91% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 1530 | 0.11 | 0.64 | 0.96 | 76% |

## Regret (a decoder's LER over the best measured)

5744 measured decodes: median error 0.02, p90 0.26.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 6418 | 0.17 | 0.68 | 83% | 0.97 | 79% |
| mwpm | 408 | 0.15 | 0.45 | 92% | 0.95 | 83% |
| bposd | 1759 | 0.16 | 0.56 | 87% | 0.97 | 79% |
| bplsd | 898 | 0.15 | 0.51 | 88% | 0.98 | 75% |
| belief_find | 974 | 0.15 | 0.47 | 90% | 0.98 | 78% |
| relay_bp | 1374 | 0.27 | 1.14 | 65% | 0.87 | 79% |
| tesseract | 1005 | 0.17 | 0.55 | 86% | 0.98 | 83% |

Timed-out decodes: 23, predicted median above their bound for 0%. Decodes without a prediction (decoder not in the fold's training): 20.

## Decoder choice

On 1496 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 91% | 1.00 | 1.17 | 1.00 | 473 |
| recommend, cheapest | 87% | 1.00 | 1.32 | 1.00 | 413 |
| always bposd | 75% | 1.02 | 1.82 | 3.17 | 57 |
| always tesseract | 96% | 1.00 | 1.05 | 3.32 | 697 |

On the 599 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.07 | 1.00 |
| recommend, cheapest | 87% | 1.00 | 1.22 | 1.00 |
| always bposd | 60% | 1.18 | 2.20 | 2.64 |
| always tesseract | 100% | 1.00 | 1.01 | 3.78 |

Chosen by recommend, accurate: tesseract 896, relay_bp 507, mwpm 93.
Chosen by recommend, cheapest: tesseract 619, relay_bp 513, mwpm 296, bposd 67, bplsd 1.

## Per held-out code family (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| Repetition | 12 | 0% | 100% | 1.36 | 1.80 | 38% | 100% |
| Quantum Reed-Muller | 4 | 0% | 100% | 0.90 | 0.99 | 25% | n/a |
| Toric | 12 | 0% | 100% | 0.49 | 1.10 | 100% | 100% |
| Surface | 87 | 0% | 100% | 0.29 | 0.93 | 79% | 96% |
| XZZX surface | 12 | 0% | 100% | 0.27 | 0.44 | 100% | 100% |
| Lift-connected surface | 222 | 34% | 90% | 0.26 | 0.69 | 77% | 95% |
| Color | 61 | 3% | 98% | 0.23 | 0.61 | 49% | 89% |
| Heavy-hex | 61 | 2% | 98% | 0.23 | 0.61 | 75% | 95% |
| Hypergraph product | 64 | 58% | 92% | 0.16 | 0.47 | 77% | 95% |
| Lifted product | 42 | 12% | 90% | 0.15 | 0.30 | 87% | 95% |
| Planar honeycomb | 44 | 0% | 98% | 0.15 | 0.39 | 79% | 97% |
| Hyperbolic Floquet | 92 | 61% | 86% | 0.14 | 0.78 | 91% | 92% |
| Concatenated Steane | 47 | 9% | 68% | 0.12 | 0.47 | 90% | 100% |
| Semi-hyperbolic Floquet | 59 | 32% | 93% | 0.09 | 0.34 | 88% | 89% |
| Honeycomb | 50 | 0% | 88% | 0.06 | 0.19 | 82% | 89% |
| Bacon-Shor | 61 | 20% | 90% | 0.06 | 0.35 | 85% | 76% |
| Quantum Tanner | 70 | 40% | 96% | 0.06 | 0.23 | 74% | 97% |
| Bivariate bicycle | 303 | 42% | 94% | 0.03 | 0.98 | 80% | 86% |
| Balanced product cyclic | 361 | 46% | 99% | 0.03 | 0.52 | 90% | 87% |
| Generalized bicycle | 110 | 62% | 90% | 0.02 | 0.44 | 92% | 88% |
| Mitten | 36 | 64% | 100% | 0.00 | 0.35 | 86% | 94% |
