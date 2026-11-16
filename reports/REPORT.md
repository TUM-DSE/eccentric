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
| all | 33319 | 0.14 | 0.39 | 94% | 0.99 | 84% |
| mwpm | 1577 | 0.14 | 0.36 | 95% | 0.95 | 87% |
| mwpf | 69 | 0.23 | 0.49 | 88% | 0.98 | 87% |
| bposd | 3988 | 0.12 | 0.36 | 95% | 0.99 | 83% |
| bposd_tuned | 3570 | 0.12 | 0.33 | 96% | 0.99 | 82% |
| bplsd | 3197 | 0.13 | 0.37 | 95% | 0.99 | 83% |
| belief_find | 3162 | 0.13 | 0.38 | 95% | 0.99 | 82% |
| relay_bp | 4174 | 0.12 | 0.33 | 98% | 0.99 | 83% |
| tesseract | 5056 | 0.16 | 0.51 | 88% | 0.99 | 82% |
| gari | 4263 | 0.16 | 0.43 | 93% | 0.99 | 85% |
| gari_pp | 4263 | 0.13 | 0.38 | 95% | 0.99 | 86% |

Timed-out decodes: 328, predicted median above their bound for 73%. Decodes without a prediction (decoder not in the fold's training): 168.

## Decoder choice

On 5089 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.15 | 1.47 | 1557 |
| recommend, cheapest | 90% | 1.00 | 1.23 | 1.00 | 1466 |
| always bposd | 72% | 1.02 | 2.01 | 9.45 | 1849 |
| always tesseract | 95% | 1.00 | 1.07 | 5.16 | 1722 |

On the 1950 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 95% | 1.00 | 1.06 | 1.51 |
| recommend, cheapest | 93% | 1.00 | 1.10 | 1.17 |
| always bposd | 53% | 1.27 | 2.62 | 17.66 |
| always tesseract | 100% | 1.00 | 1.02 | 16.47 |

Chosen by recommend, accurate: tesseract 3917, gari_pp 615, bposd_tuned 397, mwpm 142, relay_bp 15, bposd 2, bplsd 1.
Chosen by recommend, cheapest: tesseract 3348, gari_pp 720, bposd_tuned 631, mwpm 325, relay_bp 53, mwpf 9, gari 3.
