# Decoding: which decoder, at what cost, and can the choice be automated?

From the sweep table as of 4 November 2026 (`experiment_results/Sweep/results.csv`); every number
comes from `python analysis/decoding.py`. 1,336 circuits were decoded by at least two decoders on the
same shots. Decoders are compared only where both saw ≥ 100 logical errors, and one counts as better
only when the 95% intervals do not overlap ("tie" otherwise). Time is per shot on one core, decoder
construction included. Tesseract ran on most slices but not on the projected multi-QPU and
chained-module slices, and its 10 timeouts (3–9 h each, no result) are not in its averages, which
flatters it on large codes.

## Summary

1. **Accuracy ceiling: Tesseract.** Wherever it finishes, no decoder beats it significantly (1 loss in
   606 qLDPC circuits). Next: BP-OSD; the rest trail.
2. **BeliefFind and BP-LSD are dominated.** Against BP-OSD they never win (0 of 808–865 circuits) and
   lose in 9–27%, at about the same time. They can be dropped from a portfolio.
3. **The Pareto front has two ends and one middle.** MWPM/Chromobius (µs, where applicable) at one end,
   Tesseract (best, up to minutes per shot) at the other, Relay-BP in between: it is on the front for
   73% of circuits, BP-OSD for 12%.
4. **Cost grows with circuit size and, far more, with noise.** Time grows as detectors^0.8 (MWPM) to
   detectors^2.6 (Tesseract); above threshold every decoder slows 40–4,750×, and 48% of all decode
   CPU went to configurations with LER > 50%.
5. **The choice can be automated, without decoding first.** A depth-4 tree on a few circuit features
   picks a good-enough decoder for 81% of unseen codes (leave-one-code-out) at 1/5.6 of
   always-Tesseract's time; trained for accuracy instead, 91% at 58% of it. Its regime input can be a
   pilot run's LER or, as well, the mean detector firing probability computed from the circuit's error
   model in seconds (section 4).
6. **The LER itself is predictable from the error model.** Without decoding, a linear model predicts
   the LER of unseen codes within a factor of 1.9 (median; rank correlation 0.90), and whether a
   configuration is above threshold for 94% of circuits; mean detector firing alone (saturated above
   ~23%) gets 91%.
7. **Real-time decoding in software only works for slow platforms.** Median per-round decode time is
   ~5 µs for MWPM/Chromobius and ~0.1–0.3 s for BP-OSD, Relay-BP and Tesseract; 1% of circuits have
   any decoder under 1 µs per round (superconducting cycles), 29% under 1 ms (ions).

## 1. Which decoder for which code

| Family group | Most accurate | Runner-up (median LER vs best) | Notes |
|---|---|---|---|
| Surface, heavy-hex, Bacon–Shor (device noise) | Tesseract | BP-OSD 1.0–1.3×, MWPM 1.0–1.4× | Tesseract is also 14× faster than BP-OSD on these small circuits |
| Repetition, toric, XZZX (SD6) | MWPM ≈ BP-OSD | — | ties |
| Color | Tesseract | BP-OSD 1.3×, Chromobius 2.0× | Chromobius ties BP-OSD in median (1.04) at 1/3,500 the time |
| Concatenated Steane | Tesseract | BP-OSD 1.4× | |
| Floquet (honeycomb, planar, hyperbolic, semi-hyperbolic) | BP-OSD | MWPM 1.1–1.4× | Tesseract not yet run on these (in the running Floquet sweep) |
| Bivariate bicycle, BPC, GB, HGP, Mitten | Tesseract | BP-OSD 1.0×, Relay-BP 1.0–1.1× (long tail: p90 up to 8.8×) | |
| Lift-connected surface, lifted product, quantum Tanner | Tesseract | BP-OSD 1.1–1.3×, Relay-BP 1.0–1.4× | Relay-BP beats BP-OSD on lifted product and quantum Tanner |

Head to head on qLDPC circuits: Tesseract beats BP-OSD significantly in 245 of 606 circuits (median
1.18×), BP-OSD beats Relay-BP in 316 of 1,036 (31%) and loses in 64 (median 0.99), Relay-BP is
2.8× faster than BP-OSD. On matchable codes BP-OSD beats MWPM in 58 of 203 and loses in 19
(median 0.91), at 1,400× MWPM's time. MWPM gains nothing from Willow's or IBM's device noise over
SD6: its LER relative to BP-OSD is 1.0–1.2 under every noise model.

## 2. Accuracy against time

Median time per shot (ms) by circuit size:

| Decoder | < 100 det. | 100–300 | 300–1k | 1k–3k | > 3k | Exponent |
|---|---|---|---|---|---|---|
| MWPM | 0.01 | 0.04 | 0.09 | 0.44 | — | 0.84 |
| Chromobius | 0.01 | 0.04 | — | — | — | 0.59 |
| Relay-BP | 25 | 435 | 2,519 | 4,987 | 3,899 | 1.53 |
| BP-OSD | 11 | 517 | 4,949 | 24,468 | 28,006 | 2.10 |
| Tesseract | 3 | 2,310 | 33,252 | 68,916 | — | 2.63 |

Relay-BP's fixed iteration budget keeps it the cheapest BP decoder on large circuits (1/7 of BP-OSD
above 3,000 detectors). Tesseract is the fastest on small circuits and the slowest on large ones.

Time against the noise regime (median ms per shot, LER < 1% / 1–30% / > 30%): BP-OSD 6.5 / 125 /
5,224; Tesseract 6.5 / 52 / 30,799; Relay-BP 33 / 156 / 1,516; MWPM 0.01 / 0.05 / 0.43. Dense
syndromes stop BP from converging and send OSD and Tesseract's search into their worst case.

Under a time budget per shot, the best decoder that fits:

| Budget | Circuits with any decoder in budget | Median / p90 LER vs best |
|---|---|---|
| 0.1 ms | 13% | 1.14 / 2.96 |
| 1 ms | 22% | 1.00 / 2.41 |
| 10 ms | 33% | 1.00 / 1.90 |
| 100 ms | 44% | 1.00 / 1.23 |
| 1 s | 68% | 1.00 / 1.05 |
| 10 s | 94% | 1.00 / 1.01 |

## 3. Choosing the decoder automatically

Target per circuit: the cheapest decoder not significantly worse than the best. Policies evaluated on
all 1,336 circuits; the trees are tested leave-one-code-out (trained without the code they decide for).

| Policy | Good enough | p90 LER vs best | Time vs cheapest good-enough | Total time |
|---|---|---|---|---|
| Always Tesseract | 96% | 1.04 | 2.6× | 6.8 h |
| Always BP-OSD | 75% | 1.88 | 2.8× | 3.5 h |
| Always Relay-BP | 60% | 3.51 | 1.0× | 0.9 h |
| Fastest applicable (MWPM / Chromobius / Relay-BP) | 57% | 4.04 | 1.0× | 0.9 h |
| Rule: Tesseract < 300 detectors, else MWPM if matchable, else BP-OSD | 87% | 1.46 | 2.3× | 4.7 h |
| Tree for the cheapest good-enough decoder | 81% | 1.60 | 1.0× | 1.2 h |
| Tree for the most accurate decoder | 89% | 1.37 | 2.3× | 4.2 h |
| Oracle (cheapest good-enough) | 100% | 1.11 | 1.0× | 1.6 h |

The tree (depth 3, all circuits) reads:

```
if pilot LER > 0.96:                         -> Relay-BP     (above threshold, all decoders equally bad)
elif not matchable and uniform noise (SD6 or modsi1000): -> BP-OSD
elif not matchable and device noise:         -> Tesseract
elif matchable and rounds <= 8:              -> MWPM
else:                                        -> Tesseract
```

The single most useful input is the **noise regime**: whether the configuration is above threshold,
where accuracy no longer separates decoders and only cost matters. A pilot run with the fastest
decoder tells it, and so, without decoding, does the circuit's error model (section 4).

## 4. Predicting the LER without decoding

`python analysis/predict_ler.py` (features from `analysis/dem_features.py`, saved in
`experiment_results/Sweep/dem_features.json`). From each circuit's detector error model, without
sampling or decoding: the probability that each detector fires (exact for independent error
mechanisms), the total error probability, and the number and weight of the error mechanisms; a few
seconds per circuit. Target: the best measured LER of 1,519 circuits; all models leave-one-code-out.

| Features | Median / p90 error of log10 LER | Rank correlation | Above threshold (LER > 0.5): accuracy, recall, precision |
|---|---|---|---|
| Table (size, distance, k, rounds, SWAPs, nominal gate error) | 0.41 / 1.27 | 0.79 | 89%, 83%, 87% |
| Error model | **0.27 / 0.82** | **0.90** | **94%, 88%, 94%** |
| Both | 0.29 / 0.88 | 0.89 | 91%, 85%, 90% |
| Mean detector firing alone (one threshold) | — | — | 91%, 80%, 94% |

| Mean detector firing per round | Circuits | Median LER | Above threshold |
|---|---|---|---|
| < 2% | 67 | 0.05% | 0% |
| 2–5% | 122 | 0.5% | 0% |
| 5–10% | 374 | 3.4% | 8% |
| 10–20% | 403 | 17% | 12% |
| 20–50% | 553 | 97% | 87% |

So a pilot run is not needed to decide what to do with a configuration: whether it is worth
decoding at all, which decoder to use (the tree with the detector firing probability in place of
the pilot LER reaches 81% / 91% good enough, against 81% / 89% with the pilot), and how many shots to
plan (≈ 100 / predicted LER). It does not replace decoding for the benchmark's numbers: a factor of
1.9 is far wider than the ±20% interval of a measured rate.

## 5. Systems observations

- **Spend on hopeless configurations.** 48% of decode CPU went to configurations with LER > 50% (37%
  to LER > 90%). Skipping configurations predicted above threshold (section 4), or stopping after the first batch shows LER ≈ 1, would have
  halved the sweep's cost. The 14 configurations no decoder finished are all of this kind.
- **Skew.** The top 10% of decodes used 55% of the decode CPU; BP-OSD (32%) and Tesseract (29%) used
  most of it, MWPM and Chromobius none to speak of.
- **Memory, not only time.** BP-OSD's OSD-CS of order 7 needs 8–90 GB per decoder on the largest
  circuits (0.2 GB at order 0), which caps how many decodes share a machine; building it per batch
  dominated runtime until it was cached per circuit.
- **Decoder failures are compilation signals.** "Not applicable" came from routing: SWAPs create
  errors MWPM and Chromobius cannot decompose (hh, color codes on IBM devices), and errors that trip no
  detector (circuit distance 1), which BeliefFind refused outright.
- **Real time.** Median per-round decode time on one core: MWPM 5.9 µs, Chromobius 4.3 µs, Relay-BP
  0.12 s, BP-OSD 0.13 s, Tesseract 0.31 s. Against syndrome cycles of ~1–3 µs (IBM, Google), a few
  hundred µs (Apollo) and ~10–20 ms (neutral atoms): 1% of circuits have a decoder under 1 µs per round,
  11% under 10 µs, 29% under 1 ms, 40% under 10 ms. Superconducting qLDPC needs hardware decoders
  (Relay-BP was designed for that); ions and atoms leave 100–10,000× more time per round.

## Caveats

- Times are single-core Python/C++ library calls including decoder construction, not optimized
  real-time implementations; compare decoders relative to each other.
- Not every decoder ran on every slice (Relay-BP, BP-LSD, BeliefFind on qLDPC slices only; MWPM on
  matchable codes; Chromobius on the color code; Tesseract not on the multi-QPU projections), so the
  family table compares the decoders each family had.
- Floquet codes with Tesseract and Relay-BP, and the 13 codes added on 2 November, are not in this
  analysis yet.
