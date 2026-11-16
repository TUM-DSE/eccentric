# Matchmaker evaluation

Held-out codes: GroupKFold over code, 10 folds over 70 codes (all circuits of one are held out together); 5624 circuits, 36480 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 5621 | 18% | 97% | 91% | 95% |

## Applicability of decoders that refuse circuits (rule from the detector types, else learned)

| decoder | circuits | refused | accuracy | refusals caught | wrongly refused |
|---|---|---|---|---|---|
| gari | 5044 | 781 | 100% | 100% | 0% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 3931 | 0.11 | 0.42 | 0.98 | 83% |

## Regret (a decoder's LER over the best measured)

29356 measured decodes: median error 0.04, p90 0.29.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 34995 | 0.14 | 0.39 | 94% | 0.99 | 83% |
| mwpm | 1577 | 0.14 | 0.37 | 95% | 0.95 | 86% |
| mwpm_corr | 1676 | 0.19 | 0.47 | 90% | 0.95 | 83% |
| mwpf | 69 | 0.25 | 0.57 | 83% | 0.98 | 86% |
| bposd | 3988 | 0.12 | 0.36 | 95% | 0.99 | 83% |
| bposd_tuned | 3570 | 0.12 | 0.32 | 96% | 0.99 | 82% |
| bplsd | 3197 | 0.13 | 0.37 | 95% | 0.99 | 82% |
| belief_find | 3162 | 0.13 | 0.37 | 95% | 0.99 | 82% |
| relay_bp | 4174 | 0.12 | 0.33 | 98% | 0.99 | 83% |
| tesseract | 5056 | 0.16 | 0.48 | 89% | 0.99 | 83% |
| gari | 4263 | 0.16 | 0.41 | 94% | 0.99 | 84% |
| gari_pp | 4263 | 0.13 | 0.38 | 96% | 0.99 | 84% |

Timed-out decodes: 328, predicted median above their bound for 73%. Decodes without a prediction (decoder not in the fold's training): 168.

## Decoder choice

On 5116 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 93% | 1.00 | 1.14 | 5.25 | 2340 |
| recommend, cheapest | 91% | 1.00 | 1.21 | 3.89 | 2490 |
| always bposd | 72% | 1.02 | 2.02 | 10.38 | 1876 |
| always tesseract | 94% | 1.00 | 1.08 | 6.24 | 1749 |

On the 1950 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 96% | 1.00 | 1.07 | 14.35 |
| recommend, cheapest | 92% | 1.00 | 1.14 | 11.39 |
| always bposd | 53% | 1.27 | 2.62 | 20.24 |
| always tesseract | 100% | 1.00 | 1.02 | 34.90 |

Chosen by recommend, accurate: tesseract 3991, mwpm_corr 833, bposd_tuned 148, mwpm 121, gari_pp 11, relay_bp 11, mwpf 1.
Chosen by recommend, cheapest: tesseract 3289, mwpm_corr 1341, mwpm 253, bposd_tuned 187, relay_bp 45, mwpf 1.
