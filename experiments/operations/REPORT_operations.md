# Matchmaker evaluation

Extrapolation: every logical operation (surgery measurements, transversal CNOT, lattice surgery) held out, predicted from the memory experiments; scores on the 580 held-out circuits (1984 decodes). Errors are in decades (|log10 error|). Intervals [q10, q90] conformalized within each fold.

## Regime (above threshold: best LER > 0.5)

| circuits | above (base rate) | accuracy | recall | precision |
|---|---|---|---|---|
| 577 | 10% | 96% | 63% | 92% |

## Best LER (circuits whose best decode is measured)

| circuits | median error | p90 error | Spearman | in [q10, q90] |
|---|---|---|---|---|
| 441 | 0.23 | 0.80 | 0.94 | 68% |

## Regret (a decoder's LER over the best measured)

1400 measured decodes: median error 0.04, p90 0.22.

## Time per shot

| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |
|---|---|---|---|---|---|---|
| all | 1823 | 0.13 | 0.45 | 91% | 0.99 | 81% |
| mwpm | 218 | 0.16 | 0.43 | 95% | 0.95 | 78% |
| bposd | 547 | 0.12 | 0.42 | 92% | 0.98 | 79% |
| relay_bp | 577 | 0.12 | 0.38 | 94% | 0.99 | 83% |
| tesseract | 481 | 0.18 | 0.54 | 86% | 0.99 | 80% |

Timed-out decodes: 135, predicted median above their bound for 89%. Decodes without a prediction (decoder not in the fold's training): 0.

## Decoder choice

On 451 held-out circuits with at least two measured decoders. Good enough: not significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |
|---|---|---|---|---|---|
| recommend, accurate | 90% | 1.00 | 1.35 | 3.72 | 153 |
| recommend, cheapest | 87% | 1.00 | 1.44 | 3.56 | 167 |
| always bposd | 53% | 1.34 | 2.76 | 9.71 | 22 |
| always tesseract | 91% | 1.00 | 1.26 | 3.92 | 122 |

On the 0 of them on which all of belief_find, bplsd, bposd, relay_bp, tesseract were measured, so that no choice falls back:

| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough |
|---|---|---|---|---|

Chosen by recommend, accurate: tesseract 398, gari_pp 19, bposd_tuned 17, mwpf 9, mwpm 8.
Chosen by recommend, cheapest: tesseract 327, bposd_tuned 82, gari_pp 20, mwpm 17, relay_bp 4, mwpf 1.

## Per held-out operation (worst LER first)

| held out | circuits | above threshold | regime correct | LER median error | LER p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|---|
| measure_x | 77 | 4% | 96% | 0.54 | 0.97 | 83% | 81% |
| measure_z | 77 | 7% | 93% | 0.48 | 1.14 | 88% | 83% |
| ls_cnot | 42 | 12% | 93% | 0.35 | 0.99 | 91% | 100% |
| joint_xx | 71 | 6% | 94% | 0.28 | 0.60 | 91% | 90% |
| ls_merge | 46 | 0% | 100% | 0.27 | 0.60 | 95% | 98% |
| transversal_cnot | 75 | 35% | 93% | 0.25 | 0.59 | 91% | 92% |
| joint_zz | 66 | 2% | 98% | 0.22 | 0.64 | 92% | 83% |
| ls_memory | 48 | 0% | 100% | 0.10 | 0.35 | 99% | 100% |
| memory_ec | 78 | 17% | 96% | 0.06 | 0.27 | 94% | 90% |
