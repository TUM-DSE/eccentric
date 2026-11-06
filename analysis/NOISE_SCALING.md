# Noise scaling (2026-11-06)

The `noise_scaling` slice compiles codes onto eight devices and simulates them with the device's own
noise model, every error probability multiplied by 0.1, 0.25, 0.5 or 1 (the leakage rate too). The
compiled circuit is the same at every scale (`stable_seed` of the configuration without its noise), so
only the noise changes. Codes: surface (d = 3-9), color, heavy-hex and planar honeycomb (d = 3-7), at
d and 2d rounds; bb72, bb90, gross, bpc36, bpc72, bpc90 at d rounds. Up to a million shots per decode;
892 configurations with a rate, all decoders (Tesseract, BP-OSD, BP-OSD tuned, BP-LSD, Relay-BP after
its fix, MWPM, MWPF where run). Numbers from `python analysis/noise_scaling.py` (rates per round, best
decoder per configuration; Λ = LER(d) / LER(d + 2), * where the 95% intervals do not overlap).

## 1. Below threshold, at what noise? (Λ from d = 3 to 5)

| device | surface Λ at x0.1 / x0.25 / x0.5 / x1 | crosses 1 at | color Λ at x0.1 / x1 | crosses 1 at |
|---|---|---|---|---|
| Google Willow | 19.7 / 11.1 / 5.8 / 3.5 | below at x1 | 19.9 / 1.8 | below at x1 |
| Infleqtion | – / 16.9 / 7.6 / 4.1 | below at x1 | 45.0 / 3.2 | below at x1 |
| Harvard/MIT | 11.4 / 7.5 / 3.3 / 2.2 | below at x1 | 14.3 / 1.2 (not significant) | about x1 |
| IBM bicycle module | 22.6 / 16.6 / 5.5 / 3.3 | below at x1 | 6.4 / 0.64 | x0.67 |
| IBM Loon, 2 QPUs | 39.3 / 12.1 / 4.8 / 2.4 | below at x1 | 3.7 / 0.56 | x0.39 |
| IBM Flamingo, 3 QPUs | 7.6 / 3.2 / 2.0 / 1.1 (not significant) | about x1 | 5.6 / 0.86 | x0.79 |
| IBM Flamingo, 1 QPU | 9.0 / 3.1 / 1.3 / 0.69 | x0.66 | 5.1 / 0.51 | x0.39 |
| IBM Heron (Torino) | 6.1 / 1.8 / 0.76 / 0.56 | x0.40 | 4.1 / 0.48 | x0.39 |

At today's noise the surface code is below threshold everywhere except on Heron and a single Flamingo
QPU, which need 40% and 66% of their noise, and three Flamingo QPUs, where it is at threshold. The color code needs a third to two thirds of today's IBM
noise. Heavy-hex and planar honeycomb follow the same order (full table: section 2 of the script).

## 2. Larger codes get worse once they need SWAPs, at any noise

On devices with fixed couplers, suppression holds only while the code is placed without SWAPs. From
the first distance that needs routing, the larger code is worse, also at a tenth of today's noise:

| device, code | placed without SWAPs | first routed distance | Λ across that step at x0.1 / x1 |
|---|---|---|---|
| IBM Loon, surface | d = 3, 5 | d = 7 (1,315 SWAPs) | 0.22* / 0.13* |
| Google Willow, surface | d = 3, 5, 7 | d = 9 (1,808 SWAPs) | – / 0.12* (0.15* at x0.5) |
| IBM Loon, heavy-hex | d = 3, 5 | d = 7 (1,885 SWAPs) | 0.25* / 0.23* |
| IBM bicycle module, heavy-hex | d = 3, 5 | d = 7 (608 SWAPs) | 0.03* / 0.51* |
| color code on IBM devices and Willow | none | d = 3 (26-132 SWAPs) | d 5 -> 7: 0.37-0.59 / 0.13-0.72 (Willow 1.16) |

The SWAP count grows faster than the code (color code on the bicycle module: 26, 361, 1,446 SWAPs at
d = 3, 5, 7), so each routed distance step adds more error than the extra distance removes. The surface
code on the bicycle module, which fits its layout without SWAPs up to d = 9, keeps Λ > 1 from d = 7 to 9
at every scale measured (1.3-2.1). The devices that move qubits instead of swapping them (Harvard/MIT,
Infleqtion: no SWAPs at any size) suppress further in most cases but not all: moves cost errors and time
too (Harvard: color d = 5 -> 7 Λ = 0.28* at x1, surface d = 7 -> 9 0.32* at x0.5; Infleqtion: heavy-hex
d = 5 -> 7 0.22* at x0.1).
**Lower noise does not rescue routed circuits; the layout does.** This matches the native-layout
comparison of bivariate bicycle codes (`bb_native`).

## 3. qLDPC codes: how fast the LER falls with noise

Below threshold LER ~ p^((d_eff + 1)/2), so the slope s of log LER against log noise gives an effective
distance d_eff = 2 s - 1 (from the lowest pair of scales with both per-round rates below 1%):

| code (d) | bicycle module | Willow | Harvard/MIT | Infleqtion |
|---|---|---|---|---|
| bb72 (6) | 7.5 | 6.5 | 6.4 | 5.5 |
| bb90 (10) | 7.9 | 8.4 | 12.0 | 6.1 |
| gross (12) | 8.0 | 9.7 | – | – |
| bpc72 (8) | 7.2 | 6.7 | 6.4 | 5.5 |
| bpc90 (10) | 7.6 | 5.5 | 10.9 | 4.7 |

Rough estimates (two noise levels each, finite-size effects), but they point the same way: the large
qLDPC codes do not reach their distance on the routed devices (gross: 8-10 instead of 12; bpc90 and bb90
5.5-8.4 instead of 10), while the Harvard/MIT array, which needs no SWAPs, comes close to it. IBM Flamingo
and Loon give too few rates below 1% per round for an estimate.

## 4. The best decoder depends on the noise

Configurations where each decoder gave the lowest rate:

| scale | Tesseract | Relay-BP | BP-OSD tuned | MWPM | BP-LSD | BP-OSD |
|---|---|---|---|---|---|---|
| x0.1 | 114 | 35 | 21 | 19 | 1 | 1 |
| x0.25 | 138 | 39 | 15 | 17 | 10 | 0 |
| x0.5 | 142 | 27 | 23 | 13 | 16 | 5 |
| x1 | 131 | 12 | 20 | 18 | 30 | 16 |

Tesseract is best on most configurations at every noise level. Relay-BP gains as noise falls (12 -> 35
configurations), BP-LSD and BP-OSD as it rises (1 -> 30, 1 -> 16): a decoder ranking measured at one
noise level does not carry over to another.

## Caveats

- Willow's and Apollo's leakage is sampled when the circuit is built (CHANGELOG, Known issue 10); the
  scaled models scale the leakage rate, but a leaked qubit's error is frozen into the circuit. Willow's
  surface code at d = 9 is routed, which already explains its loss of suppression (section 2).
- "Crosses 1 at" interpolates Λ(3 -> 5) in log noise between the measured scales; where larger codes need
  SWAPs, the threshold of the larger distances is lower (section 2).
