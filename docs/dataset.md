# The ECCentric dataset

`experiment_results/Sweep/results.csv` holds the sweep: one row per configuration and decoder. Its headline numbers
are in `experiment_results/Sweep/stats.json` (`python -m sweep.stats`), and the website
([eccentric-benchmark.github.io](https://eccentric-benchmark.github.io)) shows it.

Schema version: 1

The schema version changes when a column is renamed, removed or changes meaning; adding a column does not change it.
Tools should read the table through `eccentric.dataset`, which checks it against this schema:

```python
from eccentric import dataset

table = dataset.load()              # every row, checked against the schema
rates = dataset.with_rates(table)   # rows with a logical error rate or an upper bound
```

Rows are produced by `sweep/` (`python -m sweep.collect <run dirs> <out>`). Each records the commit that built and
decoded it, so results from different code versions can be told apart.

## Configuration

Together with `decoder`, these columns identify a row.

| Column | Meaning |
| --- | --- |
| `slice` | Part of the sweep the row belongs to (sweep/slices.py) |
| `code` | Code name (docs/codes.md) |
| `schedule` | Requested syndrome-extraction schedule; empty for the code's default |
| `distance` | Code distance |
| `cycles` | Requested syndrome-extraction rounds; empty for the default (usually the distance) |
| `backend` | Device (docs/devices.md) |
| `backend_size` | Qubits of a synthetic (`custom_*`) device |
| `error_type` | Noise model (docs/noise-models.md) |
| `error_probability` | Physical error rate of an artificial noise model |
| `layout_method` | Qiskit layout method |
| `routing_method` | Qiskit routing method |
| `translating_method` | Gate-set translation method |
| `basis` | Memory basis, Z or X |
| `operation` | `memory`, or the fault-tolerant operation (docs/codes.md) |

## Decoder

| Column | Meaning |
| --- | --- |
| `decoder` | Decoder (docs/decoders.md) |

## Compiled circuit

| Column | Meaning |
| --- | --- |
| `schedule_used` | Schedule the circuit was built with |
| `rounds` | Syndrome-extraction rounds the circuit runs |
| `code_qubits` | Qubits of the code's circuit before compilation |
| `device_qubits` | Qubits of the device |
| `circuit_qubits` | Qubits of the compiled circuit |
| `detectors` | Detectors of the compiled circuit |
| `observables` | Logical observables of the compiled circuit |
| `two_qubit_gates` | Two-qubit gates of the compiled circuit, SWAPs included |
| `swaps` | SWAPs inserted by routing |
| `layers` | Layers (moments) of the compiled circuit |
| `measurements` | Measurements of the compiled circuit |
| `undetectable_faults` | Single faults that flip an observable without firing a detector |
| `undetectable_probability` | Total probability of those faults |

## Provenance

| Column | Meaning |
| --- | --- |
| `commit` | ECCentric commit that built the circuit or decoded it |
| `run` | Sweep run directory |
| `job_id` | Job in that run |
| `result_file` | Result file of the decode |

## Result

| Column | Meaning |
| --- | --- |
| `shots` | Shots decoded |
| `errors` | Logical errors among them |
| `decode_seconds` | CPU seconds of the decode, decoder construction included |
| `refined_chunks` | Extra chunks of shots added by refinement runs |
| `status` | How far the rate can be trusted (STATUSES) |
| `logical_error_rate` | Logical errors / shots over all rounds; empty for an upper bound |
| `ler_low` | Lower end of the 95% Wilson interval |
| `ler_high` | Upper end of the 95% Wilson interval; for an upper bound, the bound |
| `ler_per_round` | Per-round rate (1 - (1 - 2 LER)^(1/rounds)) / 2; empty at LER >= 0.5 |
| `ler_per_round_high` | Per-round rate of ler_high |
| `note` | Why a row has no rate (error message, timeout) |

## Statuses

The `status` column says how far a row's rate can be trusted. Rows with a rate or bound: `measured`, `low statistics`, `upper bound`.

| Status | Meaning |
| --- | --- |
| `measured` | At least the target number of logical errors (100) seen |
| `low statistics` | Stopped by the shot or time limit before the target number of errors |
| `upper bound` | No logical error seen; ler_high is the 95% upper bound |
| `does not fit` | The code does not fit on the device |
| `decoder not applicable` | The decoder refused the circuit (e.g. matching on non-decomposable errors) |
| `timeout` | Not a single batch of shots decoded within the time limit |
| `failed` | The decode failed for another reason |
| `pending` | Not decoded yet |

The 95% intervals are Wilson score intervals. Rates are per shot over all `rounds`; `ler_per_round` converts them.

## Re-timed matching decoders

`experiment_results/Sweep/matching_timing.csv` re-times `mwpm` and `mwpm_corr`, which take microseconds per shot:
for each of their configurations, both decoders on the same circuit, back to back on one host, 10^5 shots each
(`sweep/retime_matching.py`). `call_seconds` times the whole decode call (sampling, building the decoder and
checking the predictions, as the sweep times every decoder), `decode_only_seconds` PyMatching's decoding alone,
and `ms_per_shot` is the call per shot. The sweep table's own `decode_seconds` for these decoders are mostly
per-batch overhead; `eccentric.dataset.with_matching_times(table)` adds the re-timed `ms_per_shot`.
