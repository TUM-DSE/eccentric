# Matchmaker evaluation

Held-out codes: GroupKFold over code, 10 folds over 67 codes (all circuits of one are held out together); 3146 circuits, 20270 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 3146 | 26% | 98% | 95% | 96% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 2220 | 0.08 | 0.37 | 0.98 | 84% |

## Regret (a decoder's LER over the best measured)

16124 measured decodes: median error 0.03, p90 0.24.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 19602 | 0.13 | 0.39 | 94% | 0.99 | 83% |
| mwpm | 983 | 0.18 | 0.45 | 92% | 0.94 | 87% |
| mwpf | 67 | 0.28 | 0.56 | 82% | 0.98 | 87% |
| bposd | 3095 | 0.13 | 0.38 | 95% | 0.98 | 83% |
| bposd_tuned | 3137 | 0.12 | 0.32 | 97% | 0.99 | 81% |
| bplsd | 3140 | 0.12 | 0.35 | 95% | 0.99 | 82% |
| belief_find | 3140 | 0.12 | 0.35 | 96% | 0.99 | 83% |
| relay_bp | 3146 | 0.13 | 0.34 | 98% | 0.99 | 82% |
| tesseract | 2894 | 0.19 | 0.57 | 85% | 0.98 | 83% |

Timed-out decodes: 155, predicted median above their bound for 62%. Decodes without a prediction (decoder not in the fold's training): 156.

## Decoder choice

On 2774 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 90% | 1.00 | 1.20 | 1.65 | 640 |
| recommend, cheapest | 83% | 1.02 | 1.58 | 1.00 | 401 |
| always bposd | 61% | 1.17 | 2.34 | 11.23 | 245 |
| always tesseract | 92% | 1.00 | 1.17 | 8.35 | 879 |

On the 1820 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 97% | 1.00 | 1.04 | 1.67 |
| recommend, cheapest | 84% | 1.03 | 1.46 | 1.00 |
| always bposd | 52% | 1.28 | 2.62 | 14.07 |
| always tesseract | 100% | 1.00 | 1.01 | 12.36 |

Chosen by recommend, accurate: tesseract 1894, bposd_tuned 597, relay_bp 158, mwpm 124, bposd 1.
Chosen by recommend, cheapest: bposd_tuned 943, tesseract 775, relay_bp 650, mwpm 387, mwpf 19.
