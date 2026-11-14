# Matchmaker evaluation

Extrapolation: every logical operation (surgery measurements, transversal CNOT, lattice surgery) held out, predicted from the memory experiments; scores on the 580 held-out circuits (1984 decodes). Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 577 | 10% | 96% | 64% | 92% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 232 | 0.15 | 0.60 | 0.93 | 74% |

## Regret (a decoder's LER over the best measured)

834 measured decodes: median error 0.03, p90 0.16.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 1823 | 0.15 | 0.45 | 92% | 0.99 | 83% |
| mwpm | 218 | 0.18 | 0.43 | 92% | 0.94 | 80% |
| bposd | 547 | 0.14 | 0.44 | 92% | 0.99 | 85% |
| relay_bp | 577 | 0.14 | 0.37 | 95% | 0.99 | 84% |
| tesseract | 481 | 0.19 | 0.55 | 87% | 0.99 | 80% |

Timed-out decodes: 135, predicted median above their bound for 87%. Decodes without a prediction (decoder not in the fold's training): 0.

## Decoder choice

On 264 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 88% | 1.00 | 1.40 | 3.75 | 97 |
| recommend, cheapest | 86% | 1.00 | 1.46 | 3.58 | 103 |
| always bposd | 66% | 1.20 | 1.84 | 12.49 | 36 |
| always tesseract | 88% | 1.00 | 1.40 | 3.92 | 79 |

On the 0 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|

Chosen by recommend, accurate: tesseract 224, gari_pp 16, bposd_tuned 15, mwpm 7, mwpf 2.
Chosen by recommend, cheapest: tesseract 194, bposd_tuned 35, mwpm 16, gari_pp 16, relay_bp 2, mwpf 1.

## Per held-out operation (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| measure_x | 77 | 1% | 99% | 0.48 | 0.75 | 84% | 77% |
| measure_z | 77 | 7% | 93% | 0.46 | 0.82 | 90% | 74% |
| joint_xx | 71 | 7% | 93% | 0.35 | 0.48 | 93% | 84% |
| ls_cnot | 42 | 12% | 93% | 0.34 | 0.65 | 87% | 100% |
| joint_zz | 66 | 2% | 98% | 0.27 | 0.54 | 92% | 71% |
| ls_merge | 46 | 0% | 100% | 0.19 | 0.51 | 96% | 100% |
| transversal_cnot | 75 | 35% | 93% | 0.16 | 0.40 | 92% | 85% |
| ls_memory | 48 | 0% | 100% | 0.09 | 0.20 | 99% | 88% |
| memory_ec | 78 | 17% | 96% | 0.03 | 0.19 | 93% | 94% |
