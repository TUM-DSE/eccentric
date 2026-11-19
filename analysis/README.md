# Analyses of the ECCentric dataset

Each report states the snapshot of the sweep table it was computed from; its script regenerates every number
(`python analysis/<script>.py`). The dataset itself is described in [docs/dataset.md](../docs/dataset.md).

| Report | Question | Script | Snapshot |
| --- | --- | --- | --- |
| [INSIGHTS](INSIGHTS.md) | What does the sweep say about codes on today's devices: routing vs native layouts, technologies, waiting qubits? | `insights.py` | 2 Nov 2026 |
| [DECODING](DECODING.md) | Which decoder, at what cost, and can the choice be automated? | `decoding.py` | 4 Nov 2026 |
| [DECODER_SETTINGS](DECODER_SETTINGS.md) | Are the decoders' settings fair? Paired pilot of BP-OSD, BP-LSD and Relay-BP settings | `../sweep/decoder_settings.py` | 5 Nov 2026 |
| [NOISE_SCALING](NOISE_SCALING.md) | At what fraction of today's noise do codes cross threshold on each device? | `noise_scaling.py` | 6 Nov 2026 |
| [BASIS](BASIS.md) | How much worse is X-basis memory than Z-basis memory? | `basis.py` | 9 Nov 2026 |
| [WILLOW](WILLOW.md) | Does the Willow model reproduce Google's surface-code experiment? | — | 9 Nov 2026 |
| [LATTICE_SURGERY](LATTICE_SURGERY.md) | Do the tqec lattice-surgery operations agree with lattice-sim? | `lattice_surgery_crosscheck.py` | 13 Nov 2026 |
| [NEW_DECODERS](NEW_DECODERS.md) | Correlated matching and level-by-level BP-OSD against the other decoders | `new_decoders.py` | 16 Nov 2026 |
| [GPU](GPU.md) | Do the GPU decoders agree with the CPU ones, and how much faster are they? | `gpu_verification/` | 18 Nov 2026 |

Earlier snapshots cover fewer decoders and configurations than the current table; rerun their scripts for
current numbers. Other scripts: `predict_ler.py`, `predict_time.py` and `dem_features.py` (predicting logical
error rates and decoding times from the circuit), `crosscheck_decoder_bench.py` (against decoder-bench).
