# Matchmaker evaluation

Held-out codes: GroupKFold over code, 10 folds over 70 codes (all circuits of one are held out together); 5624 circuits, 34798 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 5621 | 18% | 97% | 91% | 94% |

## Applicability of decoders that refuse circuits (rule from the detector types, else learned)

| decoder | circuits | refused | accuracy | refusals caught | wrongly refused |
|---|---|---|---|---|---|
| gari | 5044 | 781 | 100% | 100% | 0% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 3714 | 0.10 | 0.41 | 0.98 | 83% |

## Regret (a decoder's LER over the best measured)

27302 measured decodes: median error 0.04, p90 0.27.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 33134 | 0.14 | 0.39 | 94% | 0.99 | 83% |
| mwpm | 1577 | 0.15 | 0.38 | 95% | 0.95 | 85% |
| mwpf | 69 | 0.26 | 0.50 | 87% | 0.98 | 84% |
| bposd | 3941 | 0.13 | 0.37 | 94% | 0.99 | 83% |
| bposd_tuned | 3570 | 0.12 | 0.33 | 97% | 0.99 | 80% |
| bplsd | 3196 | 0.13 | 0.38 | 95% | 0.99 | 83% |
| belief_find | 3157 | 0.13 | 0.37 | 95% | 0.99 | 83% |
| relay_bp | 4174 | 0.12 | 0.33 | 98% | 0.99 | 82% |
| tesseract | 4924 | 0.17 | 0.51 | 89% | 0.99 | 82% |
| gari | 4263 | 0.16 | 0.42 | 93% | 0.99 | 85% |
| gari_pp | 4263 | 0.13 | 0.39 | 95% | 0.98 | 85% |

Timed-out decodes: 328, predicted median above their bound for 72%. Decodes without a prediction (decoder not in the fold's training): 168.

## Decoder choice

On 4901 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.15 | 1.46 | 1530 |
| recommend, cheapest | 90% | 1.00 | 1.23 | 1.00 | 1387 |
| always bposd | 75% | 1.01 | 1.90 | 9.56 | 1901 |
| always tesseract | 95% | 1.00 | 1.07 | 5.29 | 1807 |

On the 1820 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 96% | 1.00 | 1.05 | 1.80 |
| recommend, cheapest | 94% | 1.00 | 1.11 | 1.31 |
| always bposd | 52% | 1.28 | 2.62 | 17.63 |
| always tesseract | 100% | 1.00 | 1.02 | 14.47 |

Chosen by recommend, accurate: tesseract 3717, gari_pp 601, bposd_tuned 427, mwpm 143, relay_bp 12, gari 1.
Chosen by recommend, cheapest: tesseract 3143, gari_pp 744, bposd_tuned 596, mwpm 330, relay_bp 73, mwpf 11, gari 4.
