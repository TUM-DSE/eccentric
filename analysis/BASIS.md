# X- against Z-basis memory (2026-11-08, whole sweep 2026-11-09)

Until now every ECCentric circuit was a Z-basis memory: logical |0> prepared and measured, so only X
and Y errors on the data qubits can flip the result. The `basis_x` slice runs 272 configurations in the
X basis (logical |+>), compiled with the same seed as their Z partners, and `python analysis/basis.py`
pairs them with the Z results of the same code, distance, rounds, device and noise. A pair compares
each basis's best decoder among the decoders run in both, by per-round logical error rate; "significant"
means the 95% intervals do not overlap.

210 pairs. 29 are above threshold in both bases (per-round rate 0.5) and say nothing; of the rest, 49
differ significantly, 40 of them worse in X.

## 0. Whole sweep (2026-11-09)

`sweep.add_basis` then ran the X basis for every other configuration below threshold in Z (1,639
configurations; the most accurate decoder in Z and the cheapest within 1.5x of it; run `sweep_basis_all`,
35 Tesseract timeouts). With the pairs of sections 1-4, 1,521 pairs; leaving out the 29 above threshold in
both bases, 791 of 1,492 are significantly worse in X and 100 better. X/Z ratio of the per-round rate:

| devices | code kind | pairs | median | 90th percentile | worse / better in X |
|---|---|---|---|---|---|
| IBM (Heron, Flamingo, Loon, bicycle module), today's noise | topological | 162 | 1.32 | 3.2 | 51 / 6 |
| | qLDPC | 40 | 2.70 | 21 | 31 / 0 |
| IBM, gate errors / 10 and T1, T2 x 3 | topological | 68 | 2.31 | 9.9 | 48 / 2 |
| | qLDPC | 383 | 2.43 | 8.6 | 319 / 1 |
| IBM, scaled noise (x0.1 to x0.5) | topological | 241 | 1.74 | 5.7 | 130 / 14 |
| | qLDPC | 25 | 2.20 | 8.5 | 20 / 0 |
| Infleqtion | topological | 69 | 4.26 | 14 | 65 / 3 |
| | qLDPC | 100 | 7.64 | 183 | 97 / 0 |
| Willow | topological | 89 | 0.95 | 1.7 | 11 / 9 |
| | qLDPC | 11 | 0.79 | 1.3 | 0 / 1 |
| Harvard/MIT | topological | 67 | 1.03 | 1.7 | 9 / 0 |
| | qLDPC | 34 | 0.90 | 1.2 | 0 / 5 |
| Apollo | qLDPC | 28 | 0.95 | 1.2 | 0 / 8 |
| all-to-all, SD6 | topological | 65 | 0.99 | 1.4 | 6 / 8 |
| | qLDPC | 110 | 0.78 | 1.2 | 4 / 43 |

The devices whose idling has T2 < T1 (IBM, Infleqtion) are worse in X wherever the codes are below
threshold, and the gap grows with the code's size (bb: 2.4 at bb54 to 31 at bb288; bpc: 1.5 at bpc36 to 170
at bpc180, on the devices above). With symmetric noise (SD6) the bivariate bicycle and BPC codes are the same
in both bases (0.86 to 1.15, none significant), so their X circuits are not to blame. Some qLDPC circuits are
asymmetric themselves: the coprime bivariate bicycle, generalized bicycle and lifted product codes from the
qLDPC library are better in X under SD6 (cbb30 at p = 5e-4: 0.25), so their Z-basis results are the
pessimistic ones. Willow, Harvard/MIT and Apollo are basis-neutral. Every Z-basis result on IBM devices and
Infleqtion below threshold is thus optimistic, typically by a factor of 2 to 8 and for large qLDPC codes by
one to two orders of magnitude.

## 1. The basis does not explain the Willow gap

The surface code on Willow is the same in both bases: median X/Z 0.94 (0.56 to 1.26), 2 of 28 pairs
significant, both in favour of X. Our Willow numbers stay about 2.5 times below Google's measured 0.143%
per cycle at d = 7; the missing errors are elsewhere (correlated errors, leakage as a rate, crosstalk).

## 2. Long idles make the Z basis optimistic: Infleqtion and the bicycle module

| device | surface d = 3 / 5 / 7 | heavy-hex d = 3 / 5 / 7 | qLDPC (X/Z) |
|---|---|---|---|
| Infleqtion | 2.7 / 5.1 / 11 | 3.0 / 8.0 / 29 | bb72 23, bpc72 16, bpc90 37, bb90 120, gross 505 |
| IBM bicycle module | 2.6 / 4.1 / 4.2 (d = 9: 4.7) | 2.2 / 3.3 / 2.5 | gross 21 (above threshold in X) |
| Willow, Heron, Flamingo, Loon, Harvard/MIT | 0.6 to 1.4 | 0.9 to 3.1 (section 3) | 1.0 |

The X-basis rate is up to 500 times the Z-basis rate, and the gap grows with distance. The reason is
dephasing: Infleqtion's qubits have T1 = 26 s and T2 = 2.8 s, and each round spends about 18 ms in
measurement and reset, so idling gives a Z error of about 0.3% per qubit and round against about 0.02%
for X. The Z basis sees only the X and Y part. The logical rate goes as a power of the physical rate, so
the factor of ten in physical errors becomes two to three orders of magnitude at the larger codes.
Bacon-Shor shows the mirror image (X/Z 0.5 on Infleqtion), because its circuit toggles the data qubits'
frame each half-round. On the bicycle module the codes are below threshold and idle while others are
measured, and the same T2 < T1 asymmetry appears (surface 2.6 to 4.7, gross from 0.024 per round in Z
to above threshold in X). On the other IBM devices the codes are near or above threshold, where the
basis matters little.

**Consequence:** the Z-basis results on Infleqtion and the bicycle module, including the noise-scaling
effective distances there, are optimistic. A memory's error rate should be the worse of the two bases
(or their combination), as hardware experiments report it.

## 3. Heavy-hex is worse in X, more so at larger distances

Median X/Z 1.7; 16 of 30 pairs significantly worse in X, none better. The ratio grows with distance on
every device that is below threshold (Harvard/MIT 1.3, 1.7, 3.1; Willow 1.0, 1.7, 2.8; SD6 1.05, 1.13,
1.56). The heavy-hex code protects one logical with surface-code checks and the other with Bacon-Shor
checks, which have no threshold, so the X-basis memory does not improve with distance the way the
Z-basis one does.

## 4. Codes that are basis-symmetric

Surface, color and XZZX on Willow, Heron, Flamingo, Loon and Harvard/MIT, and every code under SD6
noise (median X/Z 0.92 to 1.05; heavy-hex excepted, section 3), are the same in both bases within a few
tens of percent.

## Caveats

- Willow's leakage is decided when the circuit is built (CHANGELOG, Known issue 10), and the X circuit
  draws a different pattern than its Z partner: the d = 3 X circuit has one probability-1 error, the Z
  circuit none, and MWPM cannot decode it (2 failures). The comparison above takes the best decoder, so
  it is not affected.
- 3 Tesseract decodes of the color code at d = 7 timed out in X; 1 MWPM decode of heavy-hex failed to
  decompose the error model.
- The repetition, Floquet and concatenated Steane circuits exist only in the Z basis.
