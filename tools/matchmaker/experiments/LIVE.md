# Live run of the pilot runtime (2026-11-10)

`qec_matchmaker.runtime.execute` with real decoders on 300 circuits sampled from the end-to-end replay set
(`experiments/live/`: `prepare.py` samples the slice and records, per circuit, its held-out predictions and what
the replay simulates; `run_live.py` runs it on the cluster with ECCentric's adaptive decode in
`decode_worker.py`). Objective "cheapest"; 56 of the 300 circuits are above threshold. Each circuit is predicted
by a model that never saw its code. Pilots are real: 100 shots under the time cap, including the decoder's
construction; committing continues the pilot's decode to its error target; switching or stopping kills it.

| | live | simulated (same circuits) |
|---|---|---|
| CPU-hours | 92 | 129 |
| same outcome (decoded / skipped / stopped as above threshold) | 292 of 300 | |
| same decoder, where both decoded | 228 of 237 | |
| cost per decoded circuit, live / simulated | median 0.97 (25-75%: 0.72-1.06; 10-90%: 0.57-1.42) | |
| below threshold: good enough | 93.8% (of 244) | |
| useful circuits (best rate < 0.3) not decoded | 0% | |
| above threshold: caught (skipped or stopped) | 98.2% (of 56) | 94.6% |

Good enough: the live decode's rate is not significantly worse than the best rate measured in the sweep (its 95%
Wilson interval reaches the best one's upper bound). All 8 disagreements are borderline circuits with a best rate
between 0.43 and 0.57, where 100 pilot shots cannot tell 45% from 55%.

The simulated replay is therefore a faithful stand-in for the runtime: same choices, same per-circuit cost to
within tens of percent in the middle of the distribution, and outcomes that agree everywhere except at the
threshold itself. Data: `experiments/results/live_vs_simulated.csv` (one row per circuit) and
`experiments/results/live_slice_keys.json` (the sample).
