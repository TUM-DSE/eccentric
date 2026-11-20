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
| all | 34995 | 0.12 | 0.37 | 95% | 0.99 | 84% |
| mwpm | 1577 | 0.06 | 0.19 | 99% | 0.99 | 87% |
| mwpm_corr | 1676 | 0.05 | 0.18 | 100% | 0.99 | 86% |
| mwpf | 69 | 0.27 | 0.54 | 86% | 0.98 | 88% |
| bposd | 3988 | 0.12 | 0.35 | 96% | 0.99 | 83% |
| bposd_tuned | 3570 | 0.12 | 0.33 | 97% | 0.99 | 82% |
| bplsd | 3197 | 0.13 | 0.36 | 96% | 0.99 | 83% |
| belief_find | 3162 | 0.13 | 0.36 | 96% | 0.99 | 83% |
| relay_bp | 4174 | 0.12 | 0.32 | 98% | 0.99 | 83% |
| tesseract | 5056 | 0.16 | 0.51 | 88% | 0.99 | 83% |
| gari | 4263 | 0.17 | 0.42 | 94% | 0.99 | 84% |
| gari_pp | 4263 | 0.13 | 0.38 | 95% | 0.98 | 85% |

Timed-out decodes: 328, predicted median above their bound for 73%. Decodes without a prediction (decoder not in the fold's training): 168.

## Decoder choice

On 5116 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 91% | 1.00 | 1.17 | 1.75 | 1511 |
| recommend, cheapest | 89% | 1.00 | 1.27 | 1.04 | 1475 |
| always bposd | 72% | 1.02 | 2.02 | 10.62 | 1876 |
| always tesseract | 94% | 1.00 | 1.08 | 6.40 | 1749 |

On the 1950 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 95% | 1.00 | 1.06 | 1.88 |
| recommend, cheapest | 92% | 1.00 | 1.15 | 1.30 |
| always bposd | 53% | 1.27 | 2.62 | 20.27 |
| always tesseract | 100% | 1.00 | 1.02 | 54.63 |

Chosen by recommend, accurate: tesseract 3893, gari_pp 597, bposd_tuned 373, mwpm 147, relay_bp 86, mwpm_corr 11, belief_find 3, mwpf 3, bposd 2, bplsd 1.
Chosen by recommend, cheapest: tesseract 3246, gari_pp 762, bposd_tuned 535, mwpm 389, mwpm_corr 115, relay_bp 65, mwpf 2, gari 2.

Measured decodes that `applicable()` rules out, so never recommended: mwpm_corr 81.
