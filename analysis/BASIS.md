# X- against Z-basis memory (2026-11-08)

Until now every ECCentric circuit was a Z-basis memory: logical |0> prepared and measured, so only X
and Y errors on the data qubits can flip the result. The `basis_x` slice runs 272 configurations in the
X basis (logical |+>), compiled with the same seed as their Z partners, and `python analysis/basis.py`
pairs them with the Z results of the same code, distance, rounds, device and noise. A pair compares
each basis's best decoder among the decoders run in both, by per-round logical error rate; "significant"
means the 95% intervals do not overlap.

210 pairs. 29 are above threshold in both bases (per-round rate 0.5) and say nothing; of the rest, 49
differ significantly, 40 of them worse in X.

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
