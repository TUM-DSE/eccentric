# How do readout discriminators affect the LER?

**Question.** Holding the noise model fixed (copy-pasted ibm_boston Heron r3 gates +
coherence), how much does the choice of mid-circuit-measurement **discriminator**
change the logical error rate across six QEC codes and readout lengths?

## Setup

- **Codes** (one panel each): surface, bacon-shor, color, heavy-hex, gross, steane.
- **Lines = discriminators** (7), from `../oraqle_reports/master_fidelity.csv`:
  Linear Threshold, Baseline FNN, HERQULES, QubiCML, KLiNQ, MCMit-T, **MCMit-CNN**.
- **Readout length**: 200/400/600/800/1000 ns (sets both the flip error and the
  M-gate duration that drives readout decoherence).
- **Distances**: surface/bacon/color/hh run at **d=5, 7, 9, 12** (four figures).
  gross is the fixed [[144,12,12]] code (d=12); steane is pinned to d=9
  (concat-Steane supports only d∈{3,9,27}) — those two panels are identical in all
  four figures. **color has no even-distance code, so its d=12 panel is blank.**
- **Noise model**: ibm_boston gates SX=1.637e-4, CZ=1.191e-3; T1=284.95µs,
  T2=322.68µs; idle×1.0; noiseless reset. **Only the readout flip differs by
  discriminator**: `measure_error = 1 − geomean(Q1,Q3,Q4,Q5)` (Q2 excluded; *not*
  the F5Q_gmean column). This is exactly the `heron_cnn` model of
  run_ler_noise_models.py with the readout table swapped.
- **MCMit-CNN ("CNN") line.** d=5/7/9: reused from the existing `heron_cnn` model in
  `ler_noise_models.csv` (same noise model, the 0.910-gmean5 CNN readout table) —
  NOT re-run. d=12: `heron_cnn` has no d=12 data, so CNN was re-run at d=12 with the
  **same 0.910 table** (`CNN_ACC_0910`, identical to the table in `run_tradeoff_final.py`;
  the stale master_fidelity MCMit-CNN row was *not* used), via `LER_INCLUDE_CNN=1`.
- Decoders: surface/hh → MWPM (20k shots); bacon/color/steane → BP+OSD (5k);
  gross → BP+OSD (1k). osd_0, max_iter=100.

## Files

- Runner: `run_ler_discriminators.py` (`LER_DIST=5|7`, in the `eccentric` docker image).
- Data: `ler_discriminators_d5.csv`, `ler_discriminators_d7.csv` (180 rows each).
- Plot: `plot_discriminators.py` → `ler_discriminators_d5.pdf/png`, `ler_discriminators_d7.pdf/png`.

## Findings

1. **Discriminator choice is a second-order knob.** Best-vs-worst LER ratio across
   the 7 discriminators is at most **~1.7×** (surface d=5, 1000 ns) and usually
   1.0–1.4×. By contrast the readout-**length** sweep moves LER ~5–7× within one
   panel. Readout length dominates; the discriminator is the fine adjustment.

2. **Ranking tracks readout fidelity exactly.** The high-fidelity discriminators
   (MCMit-CNN, MCMit-T, Baseline FNN — gmean4 measure_error ~0.043–0.045 at 1000 ns)
   give the **lowest** LER; the threshold-style, low-parameter ones (QubiCML 0.061,
   Linear Threshold 0.062) give the **highest**. HERQULES and KLiNQ sit in between.

3. **The gap opens at long readout.** At 200 ns every discriminator is near
   saturation (LER ~0.4–0.5, ratio ~1.0–1.15) — readout is poor for all of them.
   By 1000 ns, where LER drops below saturation, the fidelity differences become
   visible (ratio up to 1.4–1.7× on surface/hh/color).

4. **Over-threshold codes are insensitive.** Bacon-shor (LER ~0.49–0.51, ratio
   ~1.03) and gross (LER = 1.0) are over threshold under this noise model, so the
   discriminator is irrelevant there — consistent with the noise-model sweep.

5. **The discriminator effect SHRINKS with distance — because the codes go over
   threshold.** This noise model is readout-flip-dominated, so increasing distance
   pushes LER *toward* 0.5 (over threshold), not down. As that happens the
   discriminator spread collapses:
   - surface 1000 ns best/worst ratio: **1.69× (d=5) → 1.36× (d=7) → 1.08× (d=9)**.
   - At **d=9** every tunable code is near saturation (LER ~0.43–0.51) and all
     discriminators are within ~1.02–1.08× of each other.
   - At **d=12** surface/bacon/hh are flat at ~0.5 — but this is the **even-distance
     degeneracy** (these codes require odd d; an even-distance code has a weight-d/2
     undetectable logical and behaves ~randomly), not a noise effect. The d=12 panels
     are honest but uninformative for the discriminator question; color is blank
     (no even-distance color code). gross (d=12) = 1.0 and steane (d=9) ~0.3 are
     unchanged across figures.
   The only consistently sub-threshold code is **steane (d=9)** (LER ~0.3, with the
   familiar U-shape vs readout length); there the high-fidelity discriminators
   (Baseline FNN, MCMit-T/CNN) stay visibly below the threshold-style ones.

**Takeaway.** For these surface-like codes the readout discriminator buys you up to
~1.5–1.7× LER at the long-readout operating point, and the better the readout
fidelity (gmean4) the lower the LER — but the effect is small next to readout
length and vanishes for codes already over threshold.
