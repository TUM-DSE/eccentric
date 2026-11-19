# Matchmaker evaluation

Held-out devices: leave-one-backend-out, 25 folds over 25 devices (all circuits of one are held out together); 1810 circuits, 6673 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 1810 | 34% | 94% | 89% | 94% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 1530 | 0.07 | 0.61 | 0.95 | 81% |

## Regret (a decoder's LER over the best measured)

5762 measured decodes: median error 0.02, p90 0.32.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 6438 | 0.15 | 0.54 | 88% | 0.98 | 84% |
| mwpm | 408 | 0.13 | 0.40 | 93% | 0.96 | 83% |
| bposd | 1759 | 0.15 | 0.52 | 89% | 0.98 | 81% |
| bplsd | 898 | 0.14 | 0.44 | 91% | 0.98 | 81% |
| belief_find | 974 | 0.14 | 0.41 | 93% | 0.98 | 83% |
| relay_bp | 1374 | 0.15 | 0.66 | 85% | 0.94 | 94% |
| tesseract | 1005 | 0.17 | 0.75 | 81% | 0.97 | 79% |
| chromobius | 20 | 1.28 | 2.50 | 25% | 0.54 | 40% |

Timed-out decodes: 23, predicted median above their bound for 4%. Decodes without a prediction (decoder not in the fold's training): 0.

## Decoder choice

On 1496 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.13 | 1.00 | 488 |
| recommend, cheapest | 89% | 1.00 | 1.29 | 1.00 | 438 |
| always bposd | 75% | 1.02 | 1.82 | 3.17 | 57 |
| always tesseract | 96% | 1.00 | 1.05 | 3.32 | 697 |

On the 599 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 94% | 1.00 | 1.05 | 1.00 |
| recommend, cheapest | 90% | 1.00 | 1.20 | 1.00 |
| always bposd | 60% | 1.18 | 2.20 | 2.64 |
| always tesseract | 100% | 1.00 | 1.01 | 3.78 |

Chosen by recommend, accurate: tesseract 920, relay_bp 469, mwpm 99, bposd 7, bplsd 1.
Chosen by recommend, cheapest: tesseract 703, relay_bp 489, mwpm 223, bposd 79, bplsd 2.

## Per held-out device (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| real_apollo | 108 | 1% | 100% | 1.27 | 1.95 | 54% | 100% |
| real_harvard | 93 | 70% | 63% | 0.56 | 0.81 | 79% | 82% |
| custom_full | 299 | 4% | 97% | 0.39 | 1.14 | 75% | 88% |
| native | 46 | 0% | 87% | 0.27 | 1.09 | 76% | 84% |
| real_infleqtion | 95 | 3% | 95% | 0.22 | 0.57 | 85% | 90% |
| custom_cube | 6 | 33% | 67% | 0.21 | 0.61 | 100% | n/a |
| custom_grid | 7 | 29% | 86% | 0.15 | 0.58 | 100% | n/a |
| real_willow | 50 | 18% | 90% | 0.09 | 0.27 | 87% | 87% |
| native_lr | 34 | 50% | 100% | 0.07 | 1.28 | 78% | 97% |
| real_bicycle | 246 | 35% | 97% | 0.06 | 0.44 | 88% | 92% |
| real_bicycle_lr_2_qpus | 21 | 29% | 95% | 0.05 | 0.13 | 98% | 100% |
| real_loon_1_qpu | 116 | 29% | 98% | 0.05 | 0.25 | 99% | 91% |
| real_nsinfleqtion | 12 | 42% | 92% | 0.04 | 0.18 | 100% | n/a |
| real_bicycle_lr | 144 | 40% | 99% | 0.04 | 0.28 | 95% | 97% |
| real_loon_3_qpus | 28 | 93% | 82% | 0.03 | 0.22 | 100% | 93% |
| real_bicycle_2_qpus | 21 | 24% | 100% | 0.03 | 0.11 | 95% | 100% |
| real_flamingo_1_qpu | 124 | 43% | 92% | 0.03 | 0.15 | 99% | 98% |
| real_loon | 124 | 45% | 98% | 0.01 | 0.16 | 99% | 95% |
| real_flamingo | 188 | 76% | 94% | 0.01 | 0.08 | 99% | 94% |
| real_torino | 32 | 94% | 100% | 0.00 | 0.02 | 99% | 100% |
| real_flamingo_5_qpus | 2 | 100% | 100% | 0.00 | 0.00 | 100% | 100% |
| real_loon_4_qpus | 3 | 100% | 100% | 0.00 | 0.00 | 100% | 100% |
| real_flamingo_4_qpus | 3 | 100% | 100% | 0.00 | 0.00 | 100% | 100% |
| real_loon_5_qpus | 2 | 100% | 100% | 0.00 | 0.00 | 100% | 100% |
| real_nsapollo | 6 | 0% | 100% | n/a | n/a | 75% | n/a |
