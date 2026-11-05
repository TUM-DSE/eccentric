# Matchmaker evaluation

Held-out codes: GroupKFold over code, 10 folds over 54 codes (all circuits of one are held out together); 1810 circuits, 6673 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 1810 | 34% | 95% | 93% | 92% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 1530 | 0.09 | 0.47 | 0.97 | 82% |

## Regret (a decoder's LER over the best measured)

5744 measured decodes: median error 0.02, p90 0.26.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 6418 | 0.14 | 0.56 | 88% | 0.97 | 82% |
| mwpm | 408 | 0.16 | 0.40 | 95% | 0.96 | 89% |
| bposd | 1759 | 0.12 | 0.41 | 92% | 0.98 | 82% |
| bplsd | 898 | 0.11 | 0.30 | 97% | 0.99 | 84% |
| belief_find | 974 | 0.11 | 0.29 | 97% | 0.99 | 83% |
| relay_bp | 1374 | 0.24 | 1.28 | 67% | 0.87 | 81% |
| tesseract | 1005 | 0.16 | 0.49 | 90% | 0.98 | 82% |

Timed-out decodes: 23, predicted median above their bound for 74%. Decodes without a prediction (decoder not in the fold's training): 20.

## Decoder choice

On 1496 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 91% | 1.00 | 1.16 | 1.00 | 471 |
| recommend, cheapest | 86% | 1.00 | 1.36 | 1.00 | 398 |
| always bposd | 75% | 1.02 | 1.82 | 3.17 | 57 |
| always tesseract | 96% | 1.00 | 1.05 | 3.32 | 697 |

On the 599 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.06 | 1.00 |
| recommend, cheapest | 86% | 1.00 | 1.33 | 1.00 |
| always bposd | 60% | 1.18 | 2.20 | 2.64 |
| always tesseract | 100% | 1.00 | 1.01 | 3.78 |

Chosen by recommend, accurate: tesseract 891, relay_bp 498, mwpm 103, bposd 4.
Chosen by recommend, cheapest: tesseract 577, relay_bp 515, mwpm 321, bposd 81, bplsd 2.
