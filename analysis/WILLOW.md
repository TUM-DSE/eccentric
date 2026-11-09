# The Willow model against Google's experiment (2026-11-09)

Google measured the surface code memory on Willow at 0.143% logical error per cycle at d = 7 and a
suppression factor Λ = 2.14 with its neural-network decoder, and 0.71%, 0.35% and 0.171% at d = 3, 5, 7
(Λ = 2.04) with its ensembled matching decoder (arXiv:2408.13687, Table S1). Our best results (Tesseract)
on the same code, placed on Willow without SWAPs over 10 cycles, are 0.41%, 0.14% and 0.056%: about 2.5
times below the neural-network decoder at d = 7. This note explains the gap.

## 1. Most of it is the decoder

Tesseract is much stronger than matching on these circuits: on our d = 7 circuit it is 2.3 times below
MWPM, while Google's neural-network decoder is 1.2 times below its matching decoder. Matching against
matching (our MWPM, no correlations; theirs ensembled matching):

| | d = 3 | d = 5 | d = 7 | Λ (5 -> 7) |
|---|---|---|---|---|
| Google, Willow, ensembled matching | 0.71% | 0.35% | 0.171% | 2.04 |
| ECCentric `real_willow` (MWPM) | 0.57% | 0.28% | 0.138% | 2.05 |
| ECCentric `real_willow_spec` (MWPM, sweep table) | 0.54% | 0.32% | 0.150% | 2.11 |

The model is 1.15 to 1.25 times optimistic at every distance and gets Λ right.

## 2. Error budget of the model

Fraction of the per-cycle error (MWPM) that goes away when one error source is switched off (the shares
overlap, since the logical error rate is not linear in the physical ones):

| | CZ | data idle | single-qubit | readout | reset |
|---|---|---|---|---|---|
| d = 3 | 53% | 63% | 21% | 25% | 6% |
| d = 5 | 76% | 75% | 36% | 31% | 5% |
| d = 7 | 87% | 83% | 46% | 36% | 8% |

The order matches Google's budget for the same parameters (its 72-qubit processor, Table S4: CZ 41%,
data idle 20%, readout 11%, CZ crosstalk 11%, single-qubit 9%, CZ leakage 5%, reset 2%, heating
leakage 1%); ours weighs idle and single-qubit errors somewhat more.

## 3. What the model leaves out

- **Crosstalk** (11% of Google's budget) was never applied (CHANGELOG fix 16; removed). Added as a
  two-qubit depolarizing error of 5.5e-4 on every pair of neighbouring qubits in different CZs of the
  same layer, it gives 0.76%, 0.52% and 0.337% (Λ = 1.55): twice the measured rate at d = 7. Google's
  stray coupling is mostly ZZ and swap-like, much of which a Z-basis memory does not see, so the true
  effect lies between the two rows; modelling it properly needs that channel.
- **Leakage** (6%) is frozen into each circuit when it is built (Known issue 10): whether a circuit
  carries a leaked qubit is chance (the sweep's d = 7 circuit has none, a fresh build two).
- **Correlated errors** (17% by Google's estimate): Google's own simulation, with crosstalk and leakage,
  still overpredicts Λ (2.25 against 2.14 measured).
- The parameters are those of Google's 72-qubit processor; the 105-qubit Willow chip has similar ones
  (spec sheet: CZ 0.33%, readout 0.77%; `real_willow_spec`).
- The basis does not matter on Willow (BASIS.md, section 1).

Scripts: the budget and crosstalk runs zero or add channels in the stored circuits (sweep2, `willow_surface`)
and decode with PyMatching through sinter (2 million shots or 3,000 errors each).
