# Noise models

The `error_types` field of a configuration takes these names.

| Value | Description |
| --- | --- |
| `sd6`, `pm3`, `em3_1`, `em3_2`, `si1000`, `modsi1000`, `constant` | Artificial noise models parameterised by `error_probabilities` |
| `erasure`, `erasure_undetected` | Pure erasure noise: every gate and measurement loses each of its qubits with probability `p`, heralded right away to the decoder, or not detected at all |
| `real_willow`, `real_apollo`, `real_infleqtion`, `real_heron` | Hardware-inspired noise models |
| `real_willow_spec` | Willow with the gate and measurement errors of Google's Willow spec sheet (Dec 2024) instead of arXiv:2408.13687; the other parameters, which the spec sheet does not report, are shared with `real_willow`. Use with the `real_willow` backend |
| `real_harvard` | Harvard/MIT neutral atoms from arXiv:2506.20661: 0.1% single-qubit and 0.5% CZ error, 0.46% readout error, atom loss of 0.25% per move and 0.24% per readout that is detected when the atom is next imaged, measured gate, readout and move durations, T2 = 1 s (T1 is not reported and set to T2) |
| `real_harvard_undetected` | `real_harvard` with undetected atom loss |
| `real_flamingo`, `real_flamingo_downscaled` | IBM Flamingo noise (1.56 us readout), and the same with error probabilities divided by 10 and T1/T2 multiplied by 3 |
| `real_flamingo_m700`, `real_flamingo_downscaled_m700` | The same with 700 ns readout |
| `<model>_x<factor>` | Any hardware noise model with every error probability multiplied by `factor`, e.g. `real_willow_x0.25` |
| `variance` | Decoherence driven by the T1/T2 values of a `variance_*` backend |
| `variance_low`, `variance_mid`, `variance_high` | Readout error drawn from a log-normal distribution with increasing variance |

Hardware noise models must be paired with a matching backend, for example `real_infleqtion` with `real_infleqtion`.

Loss (erasure) is modeled per qubit: a lost qubit is replaced by a maximally mixed one. When it is detected, a noiseless flag qubit records the loss and is read out as an extra detector, either right after the loss (`erasure`) or at the qubit's next measurement (`real_harvard`), so decoders can use where losses happened. Decoders that work on the full detector error model, such as `bposd`, `bplsd` or `tesseract`, make use of it.
