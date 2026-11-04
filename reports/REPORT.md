# Matchmaker evaluation

Held-out codes: GroupKFold over codes, 10 folds over 54 codes (all circuits of a code are held out together); 1633 circuits, 5965 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 1633 | 34% | 96% | 95% | 94% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 1396 | 0.11 | 0.57 | 0.97 | 82% |

## Regret (a decoder's LER over the best measured)

5163 measured decodes: median error 0.02, p90 0.26.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 5737 | 0.17 | 0.70 | 83% | 0.97 | 82% |
| mwpm | 249 | 0.19 | 0.54 | 86% | 0.90 | 86% |
| bposd | 1582 | 0.16 | 0.63 | 84% | 0.97 | 81% |
| bplsd | 898 | 0.13 | 0.38 | 95% | 0.99 | 83% |
| belief_find | 974 | 0.14 | 0.38 | 95% | 0.99 | 82% |
| relay_bp | 1197 | 0.30 | 1.41 | 62% | 0.86 | 82% |
| tesseract | 837 | 0.19 | 0.59 | 86% | 0.98 | 83% |

Timed-out decodes: 14, predicted median above their bound for 0%. Decodes without a prediction (decoder not in the fold's training): 20.

## Decoder choice

On 1336 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 90% | 1.00 | 1.16 | 1.00 | 429 |
| recommend, cheapest | 87% | 1.00 | 1.28 | 1.00 | 373 |
| always bposd | 75% | 1.01 | 1.88 | 2.77 | 57 |
| always tesseract | 96% | 1.00 | 1.04 | 2.58 | 656 |

Chosen by recommend, accurate: tesseract 811, relay_bp 518, bposd 4, belief_find 3.
Chosen by recommend, cheapest: tesseract 671, relay_bp 548, bposd 109, belief_find 4, bplsd 4.

Measured decodes that `applicable()` rules out, so never recommended: mwpm 208, chromobius 18.
