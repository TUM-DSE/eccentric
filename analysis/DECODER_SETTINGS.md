# Decoder settings (2026-11-05)

How fair ECCentric's decoder settings are, prompted by the cross-check against decoder-bench
(`experiment_results/DecoderBench`, where our Relay-BP and BP decoders differed from decoder-bench's).

## Relay-BP decoded padded syndromes

relay-bp 0.2.1's sinter `decode_shots_bit_packed` unpacks the detection events without their count,
so a circuit whose number of detectors is not a multiple of 8 got up to 7 extra syndrome bits, and
Relay-BP then rarely converged: on decoder-bench's bivariate bicycle d = 6 circuit (252 detectors),
the same decoder on the same shots took 201 instead of 5.4 ms per shot and gave LER 0.120 instead of
0.095. This explains the cross-check (our Relay-BP 1.2-7x worse on BB d6 and color d7, 252
detectors each; equal on surface d7, 336). It is known upstream (trmue/relay#34, fixed in 0.2.2).
ECCentric decodes the exact syndromes since `5708faf`; 1,413 of the sweep's 2,080 Relay-BP rows
(68%) were affected and are decoded again in `sweep_relay_fix`. Tesseract's sinter wrapper ignores
the padding bits; MWPF's too.

## The settings pilot

`sweep/decoder_settings.py` sampled 80 sweep circuits (26 code families, many devices; best LER
2e-3 to 0.3), sampled each circuit's shots once, and decoded them with ECCentric's settings and
alternatives, so every comparison is paired (McNemar, p < 0.01). Each decode had 30 CPU minutes.
Data: `experiment_results/DecoderSettings/pilot.csv`. Median LER and time relative to ECCentric's
settings (min-sum BP, 10,000 iterations, scaling 1.0; OSD-CS / LSD-CS of order 7):

| decoder | setting | topological / low-weight (41) | qLDPC, high-weight checks (37-38) | time |
|---|---|---|---|---|
| BP-OSD | 100 iterations | 1.00 | 1.13 (worse on 18) | 0.05-0.10x |
| BP-OSD | 1,000 iterations | 1.00 | 1.03 (worse on 6) | 0.14-0.20x |
| BP-OSD | 100 iterations, scaling 0.625 | **0.80** (better on 27, worse on 0) | 1.00 (worse on 9, better on 9) | 0.09-0.16x |
| BP-OSD | 1,000 iterations, scaling 0.625 | 0.80 (better on 27, worse on 0) | 0.98 (worse on 8, better on 10) | 0.28-0.37x |
| BP-LSD | 100 iterations | 1.03 | 1.24 (worse on 34) | 0.03-0.04x |
| BP-LSD | 100 iterations, scaling 0.625 | 0.85 (better on 19) | 1.37 (worse on 27) | 0.03-0.06x |
| BP-LSD | 1,000 iterations, scaling 0.625 | 0.85 (better on 19) | 1.36 (worse on 28) | 0.20-0.29x |
| BP-LSD | 100 iterations, LSD order 10 | 1.02 | 1.21 (worse on 32) | 0.03-0.05x |
| BeliefFind | 1,000 iterations | 1.02 | 1.14 (worse on 31) | 0.13-0.15x |
| BeliefFind | 100 iterations | 1.07 | 1.39 (worse on 36) | 0.02-0.03x |
| Relay-BP | decoder-bench's parameters | 1.05 | 1.06 | 0.67-0.70x |

qLDPC here: bb, gross, bpc, mitten, gb, golay, cbb, tb, lcs, qt, hgp, rm. Min-sum scaling 0.625 helps
the topological and other low-weight codes (heavy-hex 0.50, surface 0.64, Bacon-Shor 0.72, color
0.81 for BP-OSD) but hurts some qLDPC codes badly (BP-OSD at 1,000 iterations: gross 1.7x, BPC90 on
Harvard 3.1x, Mitten150 3.3x). No single setting is best everywhere.

**Decision:** `bposd` keeps the paper's settings; `bposd_tuned` (100 iterations, scaling 0.625) runs
next to it on the topological and low-weight codes (`sweep_bposd_tuned`, 1,503 decodes). BP-LSD,
BeliefFind and Relay-BP keep their settings.

## MWPF

MWPF (`mwpf` 0.2.18, `cluster_node_limit=50` as in decoder-bench) on 83 sweep configurations
(`experiment_results/DecoderSettings/mwpf_pilot.csv`, `sweep_mwpf_pilot`), against the other
decoders' rows of the same configurations:

| configurations | MWPF finished | MWPF LER relative to | time |
|---|---|---|---|
| 23 where MWPM is not applicable (routed honeycomb, heavy-hex, planar honeycomb on IBM and Willow devices) | 14 (9 timeouts) | 1.16x BP-OSD, 2.0x Tesseract | median 8.5 s per shot |
| 40 matchable | 35 | 0.95x MWPM, 0.96x BP-OSD, 1.12x Tesseract | 260x MWPM, 0.26x BP-OSD, ~Tesseract |
| 20 small qLDPC | 18 | 0.99x BP-OSD, 1.37x Tesseract | ~BP-OSD |

MWPF is about as accurate as BP-OSD everywhere and slightly better than MWPM on matchable codes, but
best nowhere, and on the circuits MWPM cannot decode it is slow and less accurate than Tesseract,
which already covers them. It stays available (`mwpf`); no sweep-wide run.
