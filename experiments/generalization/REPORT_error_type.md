# Matchmaker evaluation

Held-out noise models: leave-one-error_type-out, 12 folds over 12 noise models (all circuits of one are held out together); 1810 circuits, 6673 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 1810 | 34% | 93% | 89% | 90% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 1530 | 0.09 | 0.64 | 0.94 | 80% |

## Regret (a decoder's LER over the best measured)

5762 measured decodes: median error 0.03, p90 0.33.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 6438 | 0.16 | 0.57 | 87% | 0.97 | 82% |
| mwpm | 408 | 0.14 | 0.41 | 94% | 0.96 | 80% |
| bposd | 1759 | 0.14 | 0.52 | 89% | 0.98 | 82% |
| bplsd | 898 | 0.16 | 0.47 | 90% | 0.98 | 81% |
| belief_find | 974 | 0.16 | 0.47 | 91% | 0.98 | 79% |
| relay_bp | 1374 | 0.17 | 0.70 | 82% | 0.93 | 92% |
| tesseract | 1005 | 0.21 | 0.81 | 79% | 0.96 | 75% |
| chromobius | 20 | 1.57 | 2.29 | 25% | 0.56 | 35% |

Timed-out decodes: 23, predicted median above their bound for 9%. Decodes without a prediction (decoder not in the fold's training): 0.

## Decoder choice

On 1496 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 91% | 1.00 | 1.16 | 1.00 | 480 |
| recommend, cheapest | 82% | 1.00 | 1.52 | 1.00 | 404 |
| always bposd | 75% | 1.02 | 1.82 | 3.17 | 57 |
| always tesseract | 96% | 1.00 | 1.05 | 3.32 | 697 |

On the 599 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 93% | 1.00 | 1.07 | 1.00 |
| recommend, cheapest | 76% | 1.01 | 1.62 | 1.00 |
| always bposd | 60% | 1.18 | 2.20 | 2.64 |
| always tesseract | 100% | 1.00 | 1.01 | 3.78 |

Chosen by recommend, accurate: tesseract 898, relay_bp 494, mwpm 99, bplsd 2, bposd 2, belief_find 1.
Chosen by recommend, cheapest: tesseract 512, relay_bp 511, mwpm 252, bposd 215, bplsd 3, chromobius 2, belief_find 1.

## Per held-out noise model (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| real_apollo | 108 | 0% | 100% | 1.45 | 2.06 | 52% | 100% |
| real_harvard | 93 | 70% | 63% | 0.56 | 0.81 | 79% | 82% |
| sd6 | 292 | 4% | 98% | 0.37 | 1.01 | 73% | 88% |
| real_flamingo_downscaled_m700 | 87 | 6% | 98% | 0.18 | 0.47 | 94% | 89% |
| real_infleqtion | 101 | 6% | 94% | 0.17 | 0.63 | 84% | 90% |
| modsi1000 | 38 | 21% | 87% | 0.16 | 0.55 | 80% | n/a |
| real_flamingo_downscaled | 498 | 27% | 93% | 0.11 | 0.41 | 87% | 91% |
| real_willow | 31 | 13% | 94% | 0.07 | 0.25 | 93% | 88% |
| real_willow_spec | 25 | 16% | 96% | 0.06 | 0.19 | 93% | 88% |
| real_flamingo_m700 | 87 | 48% | 93% | 0.03 | 0.18 | 100% | 93% |
| real_flamingo | 418 | 75% | 92% | 0.01 | 0.12 | 98% | 94% |
| real_heron | 32 | 94% | 100% | 0.00 | 0.02 | 99% | 100% |
