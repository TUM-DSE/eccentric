# Extrapolation: scaling laws under the trees

Gradient-boosted trees cannot extrapolate: past the edge of the training data they keep the value at that
edge. A circuit larger or quieter than any in training gets the LER and decode time of the largest or
quietest one seen. That was the main failure in [GENERALIZATION.md](GENERALIZATION.md): with Apollo held out,
nothing in training had LERs as low as Apollo's, and its LERs were off by 1.3 decades.

The LER and time models now fit a physical scaling law first, and the trees model only what the law leaves
(`models.py`, `train(parametric=True)`, the default; `evaluate --no-laws` turns it off):

- **LER**: `log10 LER = a + b log10 p + c d log10 p + e d + f log10 detectors`, where `p` is the error mass
  per detector and `d` is the code distance. The law is fitted on circuits whose best LER is below 0.05,
  because below threshold the LER falls as `(p / p_th)^(d/2)`. It does not hold above threshold, so the
  prediction blends the law (plus trees on its residual) with the trees alone, weighted by the trees' own
  median: the law where the trees predict below 0.02, the trees above 0.2, and linearly in log10 between.
- **Code distance as an input, not a feature**: the law takes the code's distance when it is known
  (ECCentric's `distance` in training, `advise --distance` / `advise(code_distance=...)` for users). The
  trees never see it. Without it, the same law applies without the `d` terms.
- **Time, per decoder**: `log10 ms per shot = a + b log10 mechanisms + c log10 det_fire_mean`, with
  non-negative slopes. Size and noise enter separately because a large circuit at low noise is not a small
  circuit at high noise: BP converges in a few iterations below threshold and runs to its limit above.

## Evaluation

Three new single splits (`evaluate --group-by largest | quietest | next_distance`) each predict circuits
beyond the range of the training data:

- **largest**: the 20% of circuits with the most detectors, predicted from the rest;
- **quietest**: the 20% with the least error mass per detector, from the rest;
- **next distance**: the largest distance of every series of circuits that differ only in distance, from
  the smaller distances.

Same table as before (1,810 circuits, 6,673 decodes, ECCentric 1649beb), calibrated intervals. Reports:
`extrapolation/REPORT_<split>.md` (laws) and `extrapolation/REPORT_<split>_no_laws.md` (trees alone).
Errors are in decades (|log10 error|).

| split | model | LER median / p90 error | LER in [q10, q90] | time within 3x | per decoder within 3x |
|---|---|---|---|---|---|
| held-out codes (`reports/`) | trees (committed model) | 0.09 / 0.50 | 83% | 86% | MWPM 95, BP-OSD 90, Relay-BP 65, Tesseract 86 |
| | **laws** | **0.09 / 0.47** | 82% | **88%** | MWPM 95, BP-OSD 92, Relay-BP 67, Tesseract 90 |
| largest 20% | trees | 0.11 / 0.60 | 87% | 74% | MWPM 85, BP-OSD 71, BP-LSD 74, Tesseract 73 |
| | **laws** | **0.10 / 0.56** | 87% | **83%** | MWPM 100, BP-OSD 83, BP-LSD 90, Tesseract 67 |
| quietest 20% | trees | 0.74 / 2.21 | 28% | 36% | MWPM 56, BP-OSD 35, Relay-BP 48, Tesseract 28 |
| | **laws** | **0.30 / 1.06** | 50% | **43%** | MWPM 70, BP-OSD 42, Relay-BP 37, Tesseract 49 |
| next distance | trees | 0.16 / 0.46 | 79% | 88% | |
| | laws | 0.18 / 0.51 | 68% | 87% | |

Regime accuracy and decoder choice stay the same on every split, except the cheapest choice on quietest
(86% -> 89%). On held-out codes, the median predicted
time of a timed-out decode now lies above its bound for 74% of the 23 timeouts, against 0% before: the time
law follows the decode time past the training data.

**Whole devices held out** (leave-one-device-out, 25 folds, uncalibrated):

| held out | trees: LER median / p90 | laws: LER median / p90 | time within 3x, trees -> laws | regime correct |
|---|---|---|---|---|
| all devices | 0.07 / 0.61 | 0.06 / 0.57 | 88% -> 88% | 94% -> 94% |
| Apollo (108 circuits) | 1.27 / 1.95 | **0.47 / 0.98** | 53% -> 46% | 100% |
| Harvard (93) | 0.56 / 0.81 | 0.58 / 0.94 | **79% -> 60%** | 63% -> 63% |
| custom_full (all-to-all SD6, 299) | 0.39 / 1.14 | 0.33 / 0.85 | 75% -> 82% | 97% |
| Willow (50) | 0.09 / 0.27 | 0.06 / 0.17 | 87% -> 92% | 90% |

**Circuits from outside ECCentric** ([EXTERNAL.md](EXTERNAL.md): 27 Stim-generated and 7 decoder-bench
circuits, each given its code distance):

| model | LER median / p90 error | time within 3x | regime correct | choice good enough (accurate / cheapest) |
|---|---|---|---|---|
| trees | 0.29 / 0.64 | 53% | 100% | 100% / 72% |
| **laws** | **0.15 / 0.55** | **69%** | 100% | 100% / 72% |

## What this says

- **The laws fix extrapolation in noise.** With the quietest circuits held out, the LER error falls from
  0.74 to 0.30 decades; with Apollo held out, from 1.27 to 0.47. That is the regime the noise-scaling sweep
  (device noise ×0.1–1) will fill with data; the law covers what is still beyond it.
- **They fix extrapolation in size for the time.** For the largest circuits, decode time is within 3× for
  83% instead of 74%, and MWPM for all of them, because the trees no longer stay at the largest training
  circuit. Circuits from Stim's generators, which ECCentric never produced, gain as much: LER 0.29 -> 0.15
  decades, time 53% -> 69%.
- **Inside the training range nothing gets worse:** held-out codes improve slightly (LER p90 0.50 -> 0.47,
  time 86% -> 88%).
- **The code distance carries the LER law.** Without it (the law without `d` terms), quietest is at 0.54
  decades instead of 0.30 (ablation below).

## Limitations

- **Harvard held out: time 79% -> 60% within 3x.** Atom loss keeps 70% of its circuits above threshold
  while its detectors fire less than average, so the time law, which reads a low firing rate as fast
  decoding, underestimates its decode times (Tesseract by 14×, the BP decoders by 2–3×). The shipped model
  trains on Harvard, so this matters only for a new noise model with loss.
- **Quietest is still the hardest split**: 0.30 decades median error, and the 80% intervals hold only 50% of
  the truths. They are calibrated on held-out codes, not on extrapolation.
- **Next distance gets slightly worse** (0.16 -> 0.18 decades, 78 circuits with a measured best). 56% of
  these have a best LER above 0.05 (median 0.09), near or above threshold, where the law does not apply.
- **Relay-BP's time** is still the weakest prediction (67% within 3x on held-out codes, 37% on quietest).

## What did not work

Uncalibrated ablation (T: trees alone; L0: laws without the code distance; L1: laws with it, the shipped
model; L2: L1 plus the circuit distance below as a tree feature):

| split | T | L0 | L1 | L2 |
|---|---|---|---|---|
| held-out codes: LER median / p90 | 0.094 / 0.50 | 0.089 / 0.48 | 0.089 / 0.47 | 0.082 / 0.44 |
| largest | 0.106 / 0.60 | 0.104 / 0.57 | 0.099 / 0.56 | 0.118 / 0.53 |
| quietest | 0.74 / 2.21 | 0.54 / 1.55 | 0.30 / 1.06 | 0.32 / 1.15 |
| next distance | 0.165 / 0.46 | 0.155 / 0.51 | 0.175 / 0.51 | 0.154 / 0.53 |
| Harvard held out: regime / LER median / time 3x | 63% / 0.56 / 79% | | 63% / 0.58 / 60% | 44% / 0.74 / 45% |
| Apollo held out: LER median / p90 | 1.27 / 1.95 | | 0.47 / 0.98 | 0.38 / 0.97 |

- **A circuit distance feature.** The plan was to compute each circuit's distance (the fewest faults that
  flip a logical without a detection) and give it to the law. On 63 sample circuits, Stim's searches over
  hyperedges (`search_for_undetectable_logical_errors`, also with the cheap
  `dont_explore_edges_increasing_symptom_degree` mode, on a minimal circuit rebuilt from the error model)
  ran out of memory at 3 GB on every large circuit: Bacon-Shor d27, BPC180, BB98, BB180, color d17,
  Steane d9 and d27. A beam search over hyperedges found no logical error on BB, GB, BPC108, LCS175 or
  Mitten circuits. Stim's `shortest_graphlike_error` (at most 7 s per circuit) finds one for 62% of the
  circuits, but for none of the GB codes, 4% of the BB codes and 31% of the BPC codes. As the law's
  distance it did no better than no distance (quietest 0.54 decades against 0.30 with the code distance).
  As a tree feature (L2) it helps on held-out codes, is worse on largest and quietest, and breaks the
  threshold call on Harvard held out (63% -> 44%): loss is Harvard's failure mode, so its distances look
  healthy while the circuits are above threshold. It is not used.
- **An LER law without the gate.** Fitted below threshold and applied everywhere, it predicted tiny LERs for
  large circuits above threshold: on the largest split, the concatenated Steane circuits were off by 3.5
  decades (median), and circuits above LER 0.1 by 0.20 instead of 0.07. Gating on the
  trees' median fixed it; the gate thresholds (0.2 / 0.02) were chosen on the extrapolation splits among
  0.05 / 0.005 to 0.5 / 0.05.
- **A time law in expected detection events** (detectors × firing probability) extrapolated with slope 2–2.5
  learned above threshold, and overestimated large quiet circuits by 1–2 decades (Tesseract within 3x on
  largest fell to 56%). Separating size (mechanisms) from noise (firing probability) fixed it (83% overall).
