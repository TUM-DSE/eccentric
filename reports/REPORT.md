# Matchmaker evaluation

Held-out codes: GroupKFold over codes, 10 folds over 54 codes (all circuits of a code are held out together); 1633 circuits, 5965 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 1633 | 34% | 96% | 95% | 94% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 1396 | 0.10 | 0.51 | 0.97 | 83% |

## Regret (a decoder's LER over the best measured)

5163 measured decodes: median error 0.02, p90 0.26.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 5737 | 0.15 | 0.67 | 85% | 0.97 | 83% |
| mwpm | 249 | 0.17 | 0.47 | 90% | 0.92 | 90% |
| bposd | 1582 | 0.14 | 0.53 | 88% | 0.98 | 83% |
| bplsd | 898 | 0.12 | 0.36 | 96% | 0.99 | 85% |
| belief_find | 974 | 0.13 | 0.34 | 96% | 0.99 | 88% |
| relay_bp | 1197 | 0.27 | 1.42 | 63% | 0.86 | 77% |
| tesseract | 837 | 0.16 | 0.52 | 88% | 0.98 | 84% |

Timed-out decodes: 14, predicted median above their bound for 0%. Decodes without a prediction (decoder not in the fold's training): 20.

## Decoder choice

On 1336 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 90% | 1.00 | 1.16 | 1.00 | 428 |
| recommend, cheapest | 87% | 1.00 | 1.27 | 1.00 | 375 |
| always bposd | 75% | 1.01 | 1.88 | 2.77 | 57 |
| always tesseract | 96% | 1.00 | 1.04 | 2.58 | 656 |

On the 599 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.07 | 1.00 |
| recommend, cheapest | 85% | 1.00 | 1.24 | 1.00 |
| always bposd | 60% | 1.18 | 2.20 | 2.64 |
| always tesseract | 100% | 1.00 | 1.01 | 3.78 |

Chosen by recommend, accurate: tesseract 808, relay_bp 522, bposd 4, bplsd 2.
Chosen by recommend, cheapest: tesseract 664, relay_bp 554, bposd 110, bplsd 7, belief_find 1.

Measured decodes that `applicable()` rules out, so never recommended: mwpm 208, chromobius 18.
