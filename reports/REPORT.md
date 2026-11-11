# Matchmaker evaluation

Held-out codes: GroupKFold over code, 10 folds over 67 codes (all circuits of one are held out together); 5044 circuits, 32814 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 5044 | 18% | 98% | 93% | 95% |

## Applicability (learned, for decoders that refuse circuits no feature shows)

| decoder | circuits | refused | accuracy | refusals caught | wrongly refused |
|---|---|---|---|---|---|
| gari | 5044 | 781 | 84% | 42% | 9% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 3482 | 0.10 | 0.38 | 0.98 | 84% |

## Regret (a decoder's LER over the best measured)

26468 measured decodes: median error 0.04, p90 0.28.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 31311 | 0.14 | 0.43 | 92% | 0.99 | 83% |
| mwpm | 1359 | 0.14 | 0.37 | 95% | 0.95 | 85% |
| mwpf | 69 | 0.28 | 0.56 | 86% | 0.98 | 87% |
| bposd | 3394 | 0.13 | 0.39 | 93% | 0.98 | 82% |
| bposd_tuned | 3570 | 0.12 | 0.31 | 97% | 0.99 | 82% |
| bplsd | 3196 | 0.13 | 0.39 | 94% | 0.99 | 83% |
| belief_find | 3157 | 0.13 | 0.38 | 94% | 0.99 | 83% |
| relay_bp | 3597 | 0.12 | 0.33 | 98% | 0.99 | 82% |
| tesseract | 4443 | 0.18 | 0.58 | 85% | 0.98 | 81% |
| gari | 4263 | 0.19 | 0.49 | 89% | 0.98 | 84% |
| gari_pp | 4263 | 0.14 | 0.50 | 89% | 0.97 | 85% |

Timed-out decodes: 193, predicted median above their bound for 64%. Decodes without a prediction (decoder not in the fold's training): 168.

## Decoder choice

On 4637 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.14 | 1.38 | 1448 |
| recommend, cheapest | 90% | 1.00 | 1.22 | 1.00 | 1351 |
| always bposd | 75% | 1.01 | 1.90 | 9.49 | 1865 |
| always tesseract | 95% | 1.00 | 1.06 | 5.43 | 1728 |

On the 1820 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 96% | 1.00 | 1.06 | 1.92 |
| recommend, cheapest | 94% | 1.00 | 1.11 | 1.36 |
| always bposd | 52% | 1.28 | 2.62 | 17.63 |
| always tesseract | 100% | 1.00 | 1.02 | 14.47 |

Chosen by recommend, accurate: tesseract 3548, gari_pp 589, bposd_tuned 342, mwpm 139, relay_bp 12, mwpf 5, bposd 1, belief_find 1.
Chosen by recommend, cheapest: tesseract 2968, gari_pp 728, bposd_tuned 541, mwpm 333, relay_bp 51, mwpf 12, gari 4.

Measured decodes that `applicable()` rules out, so never recommended: gari 329, gari_pp 317.
