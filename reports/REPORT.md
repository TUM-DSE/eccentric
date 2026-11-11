# Matchmaker evaluation

Held-out codes: GroupKFold over code, 10 folds over 67 codes (all circuits of one are held out together); 5044 circuits, 32814 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 5044 | 18% | 98% | 93% | 94% |

## Applicability of decoders that refuse circuits (rule from the detector types, else learned)

| decoder | circuits | refused | accuracy | refusals caught | wrongly refused |
|---|---|---|---|---|---|
| gari | 5044 | 781 | 100% | 100% | 0% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 3482 | 0.10 | 0.40 | 0.98 | 84% |

## Regret (a decoder's LER over the best measured)

26468 measured decodes: median error 0.04, p90 0.28.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 31311 | 0.13 | 0.39 | 94% | 0.99 | 84% |
| mwpm | 1359 | 0.13 | 0.35 | 95% | 0.96 | 86% |
| mwpf | 69 | 0.24 | 0.55 | 86% | 0.98 | 83% |
| bposd | 3394 | 0.13 | 0.37 | 94% | 0.98 | 82% |
| bposd_tuned | 3570 | 0.11 | 0.31 | 97% | 0.99 | 81% |
| bplsd | 3196 | 0.13 | 0.36 | 95% | 0.99 | 84% |
| belief_find | 3157 | 0.13 | 0.36 | 95% | 0.99 | 84% |
| relay_bp | 3597 | 0.12 | 0.33 | 98% | 0.99 | 83% |
| tesseract | 4443 | 0.18 | 0.58 | 85% | 0.98 | 82% |
| gari | 4263 | 0.15 | 0.39 | 95% | 0.99 | 85% |
| gari_pp | 4263 | 0.12 | 0.37 | 96% | 0.99 | 86% |

Timed-out decodes: 193, predicted median above their bound for 65%. Decodes without a prediction (decoder not in the fold's training): 168.

## Decoder choice

On 4637 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.13 | 1.34 | 1427 |
| recommend, cheapest | 90% | 1.00 | 1.23 | 1.00 | 1278 |
| always bposd | 75% | 1.01 | 1.90 | 9.49 | 1865 |
| always tesseract | 95% | 1.00 | 1.06 | 5.43 | 1728 |

On the 1820 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 97% | 1.00 | 1.05 | 1.84 |
| recommend, cheapest | 94% | 1.00 | 1.11 | 1.30 |
| always bposd | 52% | 1.28 | 2.62 | 17.63 |
| always tesseract | 100% | 1.00 | 1.02 | 14.47 |

Chosen by recommend, accurate: tesseract 3528, gari_pp 586, bposd_tuned 384, mwpm 132, relay_bp 6, mwpf 1.
Chosen by recommend, cheapest: tesseract 2967, gari_pp 717, bposd_tuned 560, mwpm 327, relay_bp 64, gari 1, mwpf 1.
