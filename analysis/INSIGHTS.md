# Insights from the ECCentric sweep

State of 2 November 2026: the sweep table `experiment_results/Sweep/results.csv` (8,170 rows,
5,937 with a logical error rate or bound, 58 codes, 25 devices, 12 noise models, 7 decoders),
after the waiting-qubit fix (CHANGELOG 9) and the refinement of weak results, plus the routing
and variance experiments in `experiment_results/`. Every number below comes from
`python analysis/insights.py [<table from before fix 9>]`.

Conventions. LER is the logical error rate per shot over all rounds (d rounds unless stated);
"best" is the lowest LER over decoders and schedules of a configuration; where no logical error was
seen, the 95% upper bound stands in (Apollo's many 3.8 × 10⁻⁵ entries are such bounds). "÷10"
is the Flamingo noise model with error rates divided by 10 and T1/T2 × 3 (`real_flamingo_downscaled`).

## Summary

1. **qLDPC codes routed onto IBM heavy-hex devices do not work, even at ÷10 noise.** None of 151
   code/device/noise combinations on Heron, Loon or Flamingo reaches LER < 1%; SWAPs are 64–93% of
   all two-qubit gates.
2. **A native layout recovers 2–5 orders of magnitude.** At ÷10 noise, 7 of 9 bivariate bicycle codes
   reach LER < 10⁻⁴ on their native layout and ≈ 0.5–1 when routed onto Flamingo. At today's noise,
   even the native layout stays above 2%.
3. **IBM's bicycle module is a native home for the gross code only.** At today's noise the gross code
   reaches 9.8% on it while most other qLDPC codes sit at 50–100% (the d = 3 ones at 21–49%); at ÷10 noise it reaches 8 × 10⁻⁶.
   Chained modules and multi-QPU projections stay at 6–100%.
4. **Trapped ions dominate.** On Apollo 31 of 32 qLDPC codes reach LER < 10⁻³ (bb98: 1.7 × 10⁻⁷);
   on Infleqtion 10 of 30. Our Harvard neutral-atom model gives LER ≥ 14% for every code, driven by
   decoherence during its 10 ms readout (see caveats).
5. **Waiting qubits matter.** Letting idle qubits decohere (fix 9) raises the LER of the IBM models
   3.4–3.6× (median over unsaturated rows) and halves the share of ÷10 configurations below 1%
   (30% → 14%). Shortening readout from 1.56 µs to 700 ns lowers LER 1.2–1.9×.
6. **Distributing a code over several QPUs does not pay off yet.** Larger codes spread over 2–3
   linked QPUs are worse than smaller codes on one QPU for color, hh, Bacon–Shor and Steane codes;
   only the surface code gains, and little.
7. **Routing can make a code useless, not just noisier.** 44 compiled circuits (lcs15, lcs52, lp,
   bpc54) have circuit distance 1: a single SWAP fault flips a logical qubit undetected.
8. **Placement before routing cuts SABRE's SWAPs 1.6–7× and removes them where code and device match.**
9. **No decoder wins everywhere.** Tesseract is most accurate (13% below BP-OSD) at 3.6× the time;
   Relay-BP matches BP-OSD at a third of the time; MWPM and Chromobius are ~1,000× faster, with
   MWPM 10% less and Chromobius 4% more accurate.
10. **Our Willow model reproduces Google's error suppression:** Λ ≈ 2.4 (d5 → d7) on the Willow
    layout, against Λ = 2.14 measured.

## 1. qLDPC codes on IBM devices

Codes reaching an LER threshold (best per code), routed with Qiskit onto each device:

| Device | Noise | Codes | < 1% | Best LER (code) |
|---|---|---|---|---|
| Flamingo (3 QPUs) | Flamingo | 30 | 0 | 49% (lp) |
| Flamingo (3 QPUs) | ÷10 | 30 | 0 | 3.3% (lp) |
| Flamingo, 1 QPU | ÷10 | 11 | 0 | 2.7% (lp) |
| Loon (2 QPUs) | ÷10 | 19 | 0 | 2.4% (lp) |
| Loon, 1 QPU | ÷10 | 10 | 0 | 2.4% (lp) |
| Torino (Heron) | Heron | 11 | 0 | 57% (lcs15) |

The best results come from the smallest codes (d = 3); every larger code saturates near LER 1.
SWAP share of two-qubit gates (median over circuits): Loon 1 QPU 0.64, Loon 0.73, Flamingo 1 QPU
0.78, Torino 0.79, Flamingo 0.86, projected 4–5 QPU chains 0.82–0.93.

## 2. Native layout against routing

Bivariate bicycle codes on a layout with a coupler for every pair the syndrome circuit entangles
(`native`), against the same codes routed onto Flamingo:

| Code | Native, ÷10 | Routed, ÷10 | SWAPs routed | Native, Flamingo noise |
|---|---|---|---|---|
| bb54 | 2.6 × 10⁻³ | 0.47 | 11,701 | 26% |
| bb90 | 2.7 × 10⁻⁵ | 0.85 | 33,572 | 5.1% |
| bb98 | 2.0 × 10⁻⁶ | 0.98 | 49,790 | 3.7% |
| gross (bb144) | 1.6 × 10⁻⁵ | 1.00 | 78,744 | 7.7% |
| bb162 | 3.8 × 10⁻⁵ | 0.99 | 112,798 | 2.4% |
| bb180 | 9 × 10⁻⁶ | 0.98 | 156,322 | 3.2% |

At ÷10 noise the native layout is 180–490,000× better. At today's Flamingo noise no code reaches 1%
even natively: qLDPC on IBM hardware needs both the native layout and lower noise. With long-range
couplers as noisy as IBM's remote couplers (3% at today's noise, `native_lr`), every code fails at
today's noise (≥ 92%) and 7 of 9 still reach < 10⁻³ at ÷10.

## 3. Bicycle modules, chains and projections

On one bicycle module (IBM's planned architecture), best LER:

| Code | Flamingo noise | ÷10 |
|---|---|---|
| surface | 0.81% | 2.2 × 10⁻⁵ |
| hh | 6.3% | 1.6 × 10⁻³ |
| color | 7.7% | 1.4 × 10⁻³ |
| gross | 9.8% | 8 × 10⁻⁶ |
| bb98 / bpc126 | 98% / 99% | 1.9 × 10⁻³ / 1.2 × 10⁻³ |
| gb126 / mitten150 | 100% | 24% / 27% |

At ÷10 noise, 14 of 29 codes reach < 1%. With noisy long-range couplers (`real_bicycle_lr`) the
gross code loses 4× (3.2 × 10⁻⁵) and most others 2–30×. Two chained modules: best 5.7% (lcs150).
Projected chains of 3–5 Loon or Flamingo QPUs: best 34% (lcs150 on 3 Loon QPUs), the large codes ≈ 1.

## 4. Ions and neutral atoms

| Device | Codes | < 1% | < 10⁻³ | < 10⁻⁴ | Best |
|---|---|---|---|---|---|
| Apollo (trapped ions) | 32 | 32 | 31 | 26 | 1.7 × 10⁻⁷ (bb98) |
| Infleqtion (neutral atoms) | 30 | 17 | 10 | 4 | 6 × 10⁻⁶ (bb98) |
| Harvard (neutral atoms) | 31 | 0 | 0 | 0 | 14.6% (lp) |

On Apollo, most entries are upper bounds (no error in up to 10⁵ shots). The Harvard result is a
modelling consequence: with T1 = T2 = 1 s and 10 ms imaging, every waiting qubit takes ~1% error per
round from readout alone, which puts all codes above threshold. Whether real devices hide this
(separate readout zones, dynamical decoupling) decides the comparison; see caveats.

## 5. Idle decoherence and readout time

New (fix 9) against old LER, same configuration and decoder, both measured:

| Noise | Rows | Median ratio | 90th percentile | Median, old LER < 0.3 |
|---|---|---|---|---|
| Flamingo | 886 | 1.13 | 3.2 | 3.4 (153 rows) |
| ÷10 | 777 | 3.1 | 9.8 | 3.6 (622 rows) |
| Heron | 126 | 1.06 | 1.9 | 2.6 (6 rows) |
| SD6, Willow (controls) | 511 | 1.00 | 1.00 | 1.00 |

Configurations below 1%: Flamingo 7% → 0%, ÷10 30% → 14%. Readout of 700 ns instead of 1.56 µs
lowers LER by 1.2× (Flamingo, unsaturated rows) and 1.9× (÷10): as gate errors fall, the time qubits
wait for readout becomes the limit. The models without T1/T2 idling (SD6, Willow with its own idle
error) are unchanged, as they must be.

## 6. Distributed codes (paper DQC slice, BP-OSD, ÷10 noise)

| Code | 1 QPU (Flamingo) | 3 Flamingo QPUs | 1 QPU (Loon) | 2 Loon QPUs |
|---|---|---|---|---|
| surface | 3.6 × 10⁻³ (d7) | 2.1 × 10⁻³ (d13) | 1.3 × 10⁻³ (d7) | 1.4 × 10⁻³ (d9) |
| color | 3.2% (d7) | 23% (d13) | 1.1% (d7) | 7.1% (d9) |
| hh | 2.6% (d7) | 15% (d11) | 1.9% (d7) | 5.9% (d9) |
| Steane | 33% (d9) | 46% (d9) | 26% (d9) | 38% (d9) |

The links between QPUs cost more than the extra distance gains; only the surface code holds even.

## 7. Routing and circuit distance

44 compiled circuits contain error mechanisms that flip a logical observable and trip no detector,
all of them lcs15, lcs52, lp and bpc54 routed onto IBM or bicycle devices, with a total probability
of 5 × 10⁻⁵ to 2 × 10⁻³ per shot. These schedules have circuit distance 2 even without routing
(validation report); a SWAP between two data qubits turns one of those weight-2 faults into a
single fault. No other compiled circuit in the sweep has circuit distance 1.

## 8. Placement before routing

Mean SWAPs with SABRE (100 samples), Qiskit's placement against ours:

| Code | Grid | Grid, placed | Diamond, placed | Heavy-hex | Heavy-hex, placed |
|---|---|---|---|---|---|
| surface | 9,310 | 3,215 | 2,035 | 8,532 | 4,427 |
| hh | 10,223 | 1,402 | 6,165 | 11,850 | **0** |
| Bacon–Shor | 716 | **0** | 1,064 | 1,595 | 895 |
| color | 10,480 | 5,325 | 5,309 | 8,834 | 5,269 |
| Steane | 7,234 | 3,639 | 3,639 | 10,896 | 6,933 |

Basic routing (the paper's setting) needs 1.5–3.7× more SWAPs than SABRE. Placement removes all
SWAPs where the code matches the device (hh on heavy-hex, Bacon–Shor on the grid) and cuts the rest
1.6–7×. On the paper's sparse topologies at p = 0.002 (modsi1000), the gross code falls from
1.5 × 10⁻⁴ (all-to-all) to 0.62 (cube) and 1.0 (grid); the surface code from 5.7 × 10⁻⁵ to
1.4–1.7 × 10⁻³.

## 9. Decoders

On circuits decoded by several decoders (measured rows):

| Decoder | Lowest LER on | Median LER / BP-OSD | Median time / BP-OSD |
|---|---|---|---|
| Tesseract | 572 of 809 | 0.87 | 3.6 |
| Relay-BP | 144 of 1,100 | 1.01 | 0.33 |
| BP-OSD | 524 of 1,443 | 1 | 1 |
| BP-LSD | 56 of 831 | 1.02 | 0.93 |
| BeliefFind | 135 of 923 | 1.05 | 0.83 |
| MWPM | 77 of 226 | 1.10 | 0.0006 |
| Chromobius (color code) | 8 of 20 | 0.96 | 0.0005 |

Tesseract times out on large codes far above threshold (gb126, mitten150), as BP-OSD does on bb288;
BP-OSD's OSD-CS of order 7 needs 8–90 GB per decoder on the largest circuits.

## 10. Reference: all-to-all, SD6 at p = 10⁻³

| Code | n, k, d | Qubits | LER |
|---|---|---|---|
| gross | 144, 12, 12 | 288 | 1 × 10⁻⁶ |
| bb162 | 162, 8, 14 | 324 | 1 × 10⁻⁶ |
| bpc144 | 144, 8, 12 | 288 | 1.2 × 10⁻⁵ |
| semi-hyperbolic Floquet 400 | 400, 4, 7 | 400 | 2.9 × 10⁻⁵ |
| surface d7 | 49, 1, 7 | 118 | 9.6 × 10⁻⁵ |
| honeycomb d6 | —, 1, 6 | 180 | 5.0 × 10⁻⁴ |
| hh d7 | 49, 1, 7 | 115 | 9.6 × 10⁻³ |
| hgp_hamming | 58, 16, 3 | 100 | 18% |

The gross code protects 12 logical qubits 100× better than the surface code protects one, in 2.4×
the qubits. Per-round Λ at p = 10⁻³: surface 5.8 (d3→5) and 8.7 (d5→7), color 7.0 and 3.9,
Bacon–Shor 6.8 and 1.5, hh 1.7 and 1.4. hh has no threshold above p ≈ 2 × 10⁻³ (its LER grows with
distance at 4 × 10⁻³).

## 11. Willow

Surface code, LER per shot, Willow's measured noise:

| d | Rounds | Native layout | Willow layout |
|---|---|---|---|
| 3 | 10 | 2.8% | 4.0% |
| 5 | 10 | 0.90% | 1.35% |
| 7 | 10 | 0.22% | 0.56% |

Λ (d5 → d7, 10 rounds) is 2.4 on the Willow layout and 4.1 on the native layout; Google measured
2.14. The spec-sheet noise gives the same within about 20%.

## Caveats and open checks

- **Harvard** (section 4): the result hinges on idle decoherence during 10 ms imaging with
  T2 = 1 s. Check the readout model against the experiment (zoned readout, decoupling) before
  claiming anything about neutral atoms from it.
- **Willow, d7 over 30 rounds**: BP-OSD gives 4.1% (long run, 2,550 shots) where MWPM gives 1.7%;
  the difference is unexplained.
- **Upper bounds**: most Apollo entries and several reference entries are bounds, not rates.
- **Paper DQC**: different codes run at different distances per device, so the comparison is between
  the largest code each device holds, not at equal distance.
- **Running**: 9 long jobs (until Wednesday) refine the Willow d7, semi-hyperbolic Floquet 400 and
  Apollo rows; the 13 codes added on 2 November (coprime and trivariate bicycle, many-hypercube,
  Golay) and the Floquet codes on devices are not in the sweep yet.
