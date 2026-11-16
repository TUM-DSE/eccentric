# Matchmaker evaluation

Held-out codes: GroupKFold over code, 10 folds over 70 codes (all circuits of one are held out together); 5624 circuits, 34798 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 5621 | 18% | 98% | 91% | 95% |

## Applicability of decoders that refuse circuits (rule from the detector types, else learned)

| decoder | circuits | refused | accuracy | refusals caught | wrongly refused |
|---|---|---|---|---|---|
| gari | 5044 | 781 | 100% | 100% | 0% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 3923 | 0.10 | 0.41 | 0.98 | 82% |

## Regret (a decoder's LER over the best measured)

27868 measured decodes: median error 0.04, p90 0.28.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 33134 | 0.14 | 0.42 | 93% | 0.99 | 82% |
| mwpm | 1577 | 0.15 | 0.42 | 92% | 0.95 | 83% |
| mwpf | 69 | 0.23 | 0.51 | 86% | 0.98 | 83% |
| bposd | 3941 | 0.14 | 0.48 | 90% | 0.98 | 81% |
| bposd_tuned | 3570 | 0.12 | 0.33 | 96% | 0.99 | 81% |
| bplsd | 3196 | 0.14 | 0.40 | 93% | 0.99 | 81% |
| belief_find | 3157 | 0.14 | 0.41 | 93% | 0.99 | 81% |
| relay_bp | 4174 | 0.13 | 0.37 | 95% | 0.99 | 82% |
| tesseract | 4924 | 0.19 | 0.59 | 84% | 0.98 | 81% |
| gari | 4263 | 0.16 | 0.40 | 95% | 0.99 | 84% |
| gari_pp | 4263 | 0.12 | 0.36 | 96% | 0.99 | 85% |

Timed-out decodes: 328, predicted median above their bound for 61%. Decodes without a prediction (decoder not in the fold's training): 168.

## Decoder choice

On 5088 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.15 | 1.44 | 1573 |
| recommend, cheapest | 90% | 1.00 | 1.21 | 1.00 | 1467 |
| always bposd | 73% | 1.01 | 1.97 | 9.10 | 1887 |
| always tesseract | 95% | 1.00 | 1.07 | 4.81 | 1850 |

On the 1820 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 97% | 1.00 | 1.05 | 1.88 |
| recommend, cheapest | 95% | 1.00 | 1.09 | 1.39 |
| always bposd | 52% | 1.28 | 2.62 | 17.63 |
| always tesseract | 100% | 1.00 | 1.02 | 14.47 |

Chosen by recommend, accurate: tesseract 3916, gari_pp 625, bposd_tuned 387, mwpm 142, relay_bp 15, bposd 2, bplsd 1.
Chosen by recommend, cheapest: tesseract 3345, gari_pp 739, bposd_tuned 613, mwpm 325, relay_bp 52, mwpf 11, gari 3.
