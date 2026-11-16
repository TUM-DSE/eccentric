# Correlated matching and level-by-level decoding (2026-11-16)

Two decoders taken from the hierarchical-decoding survey (arXiv 2611.16271, preprint), items 3 and 4:

- `mwpm_corr` (`eccentric/decoders/correlated_matching.py`): PyMatching's two-pass correlated matching, the
  approach of Cain et al., "Correlated decoding of logical algorithms with transversal gates", Phys. Rev.
  Lett. 133, 240602 (2024). Errors stim cannot decompose into edges are split by matching their own detectors
  on the graph of the model's graph-like errors; errors more likely than not are taken to have happened (exact).
- `hier_bposd` (`eccentric/decoders/hierarchical.py`): BP on the level-1 detectors of a concatenated code,
  its posteriors as the priors of BP on the level-2 detectors, and so on, then BP-OSD (OSD-CS order 7, 100
  min-sum iterations) on all detectors. This is Poulin's soft level-by-level decoding (Phys. Rev. A 74, 052333,
  2006), used by Goto's many-hypercube codes (Science Advances, 2024), applied heuristically to the circuit's
  detector error model. `hier_bposd_hard` passes hard decisions between levels instead (ablation). Only
  for mhc36, mhc216 and concatenated Steane at d = 9, 27.

Numbers: `python analysis/new_decoders.py`. A ratio is new / reference on the configurations where both have a
rate above 0 and the reference is below 0.5 (below 1 is better); "significant" means the 95% Wilson intervals
do not overlap. Runs `sweep_mwpm_corr`, `sweep_hier`, `sweep_hier_d27`.

## 1. Correlated matching

Every configuration with an MWPM row that fits its device: 1,682 decodes. 6 are refused: toric-code operations,
where some faults flip an odd number of detectors in a part of the matching graph without a boundary.

| | configurations | median mwpm_corr / mwpm | 10th-90th percentile | significantly better / worse |
|---|---|---|---|---|
| all | 1,415 | 0.88 | 0.42-1.03 | 404 / 16 |
| hyperbolic and semi-hyperbolic Floquet (codes with 5 or more configurations) | 52 | 0.54-0.81 by code | | 26 / 0 |
| lattice-surgery CNOT | 34 | 0.69 | 0.32-0.97 | 17 / 1 |
| surface (memory and operations) | 495 | 0.76 | 0.35-1.03 | 196 / 5 |
| transversal CNOT | 5 | 0.77 | 0.75-0.81 | 0 / 0 |
| heavy-hex | 293 | 0.93 | 0.60-1.07 | 53 / 8 |
| Bacon-Shor, toric, repetition | 185 | 1.00 | | 2 / 0 |

- 99 configurations that plain MWPM could not decode now have a rate: heavy-hex 46, honeycomb and planar
  honeycomb 30, surface 23 (5 transversal CNOTs, 10 joint measurements, 5 logical measurements, 3 memories).
- mwpm_corr is the most accurate decoder of the sweep on 216 configurations.
- Correlations help most where single faults flip both an X- and a Z-type detector pair (Floquet codes,
  surgery and CNOT gadgets) and not at all on codes decoded one basis at a time (Bacon-Shor, repetition).
- On the surface-code transversal CNOT (d = 3 and 5, SD6 at p = 2e-3), correlated matching is still behind
  BP-OSD, Relay-BP and Tesseract (d = 5: 0.035 against 0.025, 0.020, 0.014).

## 2. Level-by-level BP-OSD

112 memory configurations (mhc36 56, mhc216 23, Steane d = 9 30, d = 27 3), soft and hard.

| code | against | configurations | median hier_bposd / reference | significantly better / worse |
|---|---|---|---|---|
| Steane d = 9 | hard decisions | 25 | 0.65 | 15 / 0 |
| | BP-OSD | 26 | 0.88 | 4 / 0 |
| | Relay-BP | 25 | 0.99 | 6 / 3 |
| | Tesseract | 25 | 1.44 | 0 / 9 |
| mhc36 | hard decisions | 10 | 0.56 | 10 / 0 |
| | BP-OSD | 14 | 0.92 | 0 / 0 |
| | tuned BP-OSD | 24 | 1.12 | 0 / 2 |
| | Relay-BP | 14 | 1.05 | 0 / 0 |
| | Tesseract | 25 | 1.16 | 0 / 2 |
| mhc216 | hard decisions | 4 | 0.08 | 4 / 0 |
| | Relay-BP | 3 | 9.2 | 0 / 2 |
| | Tesseract | 6 | 79 | 0 / 6 |

- Soft information is what helps: hard decisions between levels are worse everywhere.
- On Steane d = 9 the soft decoder has a lower rate than every BP-type decoder of the configuration on 12 of 22,
  but significantly so only on Infleqtion (0.021 against tuned BP-OSD's 0.038; Tesseract 0.011); the other
  wins are near threshold (rates 0.09-0.43). It is the most accurate decoder on 2 configurations, both near threshold.
- On mhc216 Relay-BP and Tesseract are far better (Apollo: 0.0033 against 5.4e-4 and 4e-5).
- Steane d = 27: no rate. All 6 decodes timed out or gave an upper bound (about 36 GB of memory per process).
- The Harvard/MIT atom circuits of mhc36 and mhc216 (4 configurations) are refused: their detector coordinates
  do not carry the check index the level assignment reads.

Pilot (`hier_pilot` on the cluster, same seed per configuration; "flat" = the final BP-OSD alone, without the
level-by-level stages, "10k" = 10,000 iterations in the final stage):

| configuration | soft | flat | soft 10k | flat 10k |
|---|---|---|---|---|
| Steane d = 9, SD6 1e-3 | 0.017 | 0.027 | 0.015 | 0.021 |
| Steane d = 9, Infleqtion | 0.023 | 0.059 | 0.024 | 0.043 |
| Steane d = 9, grid | 0.035 | 0.098 | 0.031 | 0.074 |
| mhc216, Apollo | 0.0043 | 0.0050 | 0.0068 | 0.0031 |

The schedule beats the same BP-OSD without it by 1.4-3x on Steane d = 9 even when the flat version gets 10,000
iterations; on mhc216 the flat version with 10,000 iterations is better. The sweep keeps 100 final iterations.
