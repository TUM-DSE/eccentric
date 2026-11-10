# Matchmaker evaluation

Held-out codes: GroupKFold over code, 10 folds over 67 codes (all circuits of one are held out together); 5044 circuits, 23507 decodes. Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 5017 | 18% | 98% | 92% | 94% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 3439 | 0.10 | 0.38 | 0.98 | 84% |

## Regret (a decoder's LER over the best measured)

18203 measured decodes: median error 0.03, p90 0.22.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 22785 | 0.13 | 0.38 | 94% | 0.99 | 82% |
| mwpm | 1359 | 0.15 | 0.36 | 95% | 0.95 | 85% |
| mwpf | 69 | 0.25 | 0.56 | 80% | 0.98 | 81% |
| bposd | 3394 | 0.13 | 0.36 | 95% | 0.99 | 82% |
| bposd_tuned | 3570 | 0.12 | 0.30 | 98% | 0.99 | 81% |
| bplsd | 3196 | 0.12 | 0.36 | 95% | 0.99 | 82% |
| belief_find | 3157 | 0.12 | 0.36 | 96% | 0.99 | 83% |
| relay_bp | 3597 | 0.12 | 0.33 | 97% | 0.99 | 83% |
| tesseract | 4443 | 0.18 | 0.52 | 88% | 0.99 | 82% |

Timed-out decodes: 193, predicted median above their bound for 60%. Decodes without a prediction (decoder not in the fold's training): 168.

## Decoder choice

On 3728 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 92% | 1.00 | 1.15 | 2.43 | 816 |
| recommend, cheapest | 84% | 1.01 | 1.51 | 1.00 | 820 |
| always bposd | 69% | 1.06 | 2.06 | 10.01 | 961 |
| always tesseract | 94% | 1.00 | 1.09 | 6.99 | 1025 |

On the 1820 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|
| recommend, accurate | 98% | 1.00 | 1.05 | 1.67 |
| recommend, cheapest | 81% | 1.04 | 1.56 | 1.00 |
| always bposd | 52% | 1.28 | 2.62 | 14.07 |
| always tesseract | 100% | 1.00 | 1.01 | 12.36 |

Chosen by recommend, accurate: tesseract 2751, bposd_tuned 705, relay_bp 138, mwpm 132, bplsd 1, mwpf 1.
Chosen by recommend, cheapest: bposd_tuned 1221, relay_bp 955, tesseract 888, mwpm 636, mwpf 28.
