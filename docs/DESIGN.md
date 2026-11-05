# Design

Matchmaker predicts, from a noisy Stim circuit (or its detector error model, or a check matrix with
priors), how decoding it will go: above or below threshold, the logical error rate, each decoder's
time per shot, and which decoder to use. It is trained on measurements from
[ECCentric](https://github.com/TUM-DSE/eccentric) and kept up to date as ECCentric produces more.

## Package layout

| Module | Role |
|---|---|
| `qec_matchmaker/features.py` | Features of a circuit, DEM or check matrix; the only place features are defined |
| `qec_matchmaker/decoders.py` | The decoders Matchmaker knows, and when each applies |
| `qec_matchmaker/dataset.py` | Training table from ECCentric's results and a feature store; versioned on disk |
| `qec_matchmaker/models.py` | Training, saving, loading and predicting with the models |
| `qec_matchmaker/recommend.py` | Choosing a decoder for an objective from the predictions |
| `qec_matchmaker/evaluate.py` | Leave-one-code-out evaluation and its report |
| `qec_matchmaker/api.py` | `advise(...)`: the public entry point |
| `qec_matchmaker/cli.py` | `qec-matchmaker` command line |
| `qec_matchmaker/pipeline.py` | Incremental feature extraction over circuit files; the update cycle |
| `qec_matchmaker/io/decoder_bench.py` | Reading decoder-bench HDF5 traces |
| `qec_matchmaker/calibrate.py` | Scaling predicted times to another machine |

## Features (`features.py`, `FEATURE_VERSION = 1`)

All from the detector error model, built with `approximate_disjoint_errors=True` and without
decomposition, so a circuit, a DEM and a check matrix with priors give the same numbers.
A mechanism is one `error(p)` instruction (after flattening); a detector fires with probability
`(1 - prod(1 - 2p)) / 2` over the mechanisms that flip it (exact for independent mechanisms).

| Name | Meaning |
|---|---|
| `detectors`, `observables`, `mechanisms` | sizes |
| `mechanism_weight_mean`, `mechanism_weight_max` | detectors flipped per mechanism |
| `hyperedge_fraction` | share of mechanisms flipping more than 2 detectors |
| `error_mass` | sum of mechanism probabilities |
| `error_mass_per_detector` | `error_mass / detectors` |
| `det_fire_mean`, `det_fire_median`, `det_fire_p90`, `det_fire_max` | detector firing probabilities |
| `logical_mass` | sum of probabilities of mechanisms that flip an observable |
| `undetectable_faults`, `undetectable_probability` | mechanisms that flip an observable and no detector |
| `graphlike` | 1 if every mechanism flips at most 2 detectors, else 0 |
| `color_annotated` | 1 if every detector has a 4th coordinate (Chromobius' color/basis), else 0 |
| `qubits`, `two_qubit_gates`, `swaps`, `ticks`, `measurements` | circuit only (NaN from a DEM) |
| `rounds` | distinct values of the detectors' 3rd (time) coordinate; NaN if absent |
| `decomposable` | circuit only: 1 if `decompose_errors=True` succeeds (MWPM applies), else 0 |

`FEATURE_NAMES` lists the model inputs in a fixed order. Functions:
`features_from_dem(dem)`, `features_from_circuit(circuit, check_decomposition=True)`,
`features_from_matrices(check_matrix, priors, observables_matrix=None)`; each returns a `dict`.

## Decoders (`decoders.py`)

Names as in ECCentric: `mwpm` (PyMatching), `bposd`, `bplsd`, `belief_find` (ldpc), `relay_bp`,
`tesseract`, `chromobius`. `applicable(features) -> dict[str, bool]`: `mwpm` needs `decomposable`
(graph-like errors in a DEM settle it in favour), `chromobius` needs `color_annotated`, the rest always
apply; `None` when the features cannot tell (e.g. matching from a DEM alone), which the recommendation
treats as not applicable and `advise` reports.

## Training table (`dataset.py`)

One row per circuit and decoder. Columns:

- keys: `circuit_key` (`<run>/<circuit job id>` in ECCentric), `code`, `family`, `distance`, `cycles`,
  `backend`, `error_type`, `error_probability`, `slice`, `decoder`
- features: every name in `FEATURE_NAMES` (per circuit, repeated on its decoder rows)
- labels: `status` (`measured`, `low statistics`, `upper bound`, `timeout`, `decoder not applicable`,
  `failed`), `shots`, `errors`, `ler` (errors / shots, NaN without shots), `ler_low`, `ler_high`
  (95% Wilson), `ms_per_shot` (decode time per shot on one core, decoder construction included, NaN
  when unknown), `timeout_s` (the limit a timed-out decode hit, else NaN)

Stored as `data/dataset.csv` with `data/MANIFEST.json`: feature version, source (ECCentric commit
and file hashes), row and circuit counts, date. `load_dataset(dir)` returns the frame and manifest.

## Models (`models.py`)

Gradient-boosted trees (scikit-learn `HistGradientBoosting*`), inputs `FEATURE_NAMES` (+ the decoder
for per-decoder models), never the code name, so that unseen codes are predicted from their circuits.

- **Regime**: per circuit, P(best LER > 0.5); best LER = lowest over decoders with a rate (an upper
  bound counts as its bound).
- **LER**: per circuit, quantiles 0.1 / 0.5 / 0.9 of log10 best LER (circuits with a measured best).
- **Regret**: per decoder, log10(LER / best LER) on circuits with at least two measured decoders.
- **Time**: per decoder, quantiles 0.1 / 0.5 / 0.9 of log10 ms per shot. Timed-out decodes have no shots;
  they are right-censored at `timeout_s * 1000 / 100` ms per shot (they did not finish a first batch of
  100 shots): after a first fit, each joins one refit at max(predicted median, its bound).

**Scaling laws under the trees** (`train(..., parametric=True)`, the default; `evaluate --no-laws` turns
them off). Trees cannot extrapolate: beyond the training data they repeat the value at its edge, so a
circuit larger or quieter than any in training gets the decode time and LER of the edge. The LER and time
quantile trees therefore model what is left after a law fitted first (bounded least squares, `Law`):

- LER: `log10 LER = a + b log10 p + c d log10 p + e d + f log10 detectors` with `p` the error mass per
  detector and `d` the code distance, fitted on circuits whose best LER is below 0.05 (below threshold
  the LER falls as `(p / p_th)^(d/2)`). The code distance is not a circuit feature: it is an optional input
  (`code_distance`, `LAW_INPUTS`, never seen by the trees; ECCentric's `distance` in training, `advise
  --distance` for users). Without it, the same law without the `d` terms. The law does not hold above
  threshold, so the prediction blends the law (plus trees on its residual) and the trees alone by the
  trees' own median: the law's below 0.02, the trees' above 0.2, linear in log10 between.
- Time, per decoder: `log10 ms = a + b log10 mechanisms + c log10 det_fire_mean` (slopes ≥ 0): size and
  noise apart, because a large circuit at low noise is not a small one at high noise (BP converges in a
  few iterations below threshold and runs to its limit above).

Fitted coefficients are in `meta["laws"]`; `MODEL_VERSION` (2) guards the saved form.

`train(df, manifest=None, calibrate=False, parametric=True) -> ModelBundle`; `ModelBundle.predict(features_frame or dict) -> DataFrame`
(per circuit and decoder: `circuit_key`, `decoder`, `applicable`, `p_above`, `ler_q10/q50/q90`,
`regret_q50`, `ler_decoder_q50`, `ms_q10/q50/q90`); `ModelBundle.save(dir)` / `ModelBundle.load(dir)`
(`models.joblib` plus `meta.json` with the scikit-learn version, feature version and features used,
the training manifest and, from `qec-matchmaker train --metrics`, the evaluation metrics). The inputs are
the `FEATURE_NAMES` columns the training table actually fills; a decoder a model did not see in training
gets no prediction.

With `train(..., calibrate=True)` (what `qec-matchmaker train` and `evaluate` do), the outer LER and time
quantiles are conformalized: models trained without a group of codes (GroupKFold, 4 folds) predict that
group, and [q10, q90] is widened in log10 by the amount that makes it hold 80% of those held-out truths
(per decoder for the time), stored in `meta["conformal"]`. `meta["ranges"]` holds each feature's
training range; predictions name the features of a circuit outside it (`outside_training`).

## Recommendation (`recommend.py`)

Objectives: `accurate` (lowest predicted LER), `cheapest` (cheapest decoder whose predicted LER is
within `tolerance` (default 1.25×) of the best predicted), `budget` (lowest predicted LER with
`ms_q50 <= budget_ms`; with `rounds`, or `objective="budget_per_round"`, `ms_q50 / rounds` is compared).
Ties go to the faster decoder. Only applicable decoders; above
threshold (`p_above > 0.5`) the advice is the cheapest decoder plus a warning that decoding will
not separate decoders.

## Continuous integration with ECCentric

1. On the cluster, `qec-matchmaker extract <run dirs> --store features.jsonl` adds the features of
   circuit files it has not seen (keyed by `<run>/<circuit id>`, with size and mtime, recomputed
   when the file or `FEATURE_VERSION` changes).
2. `qec-matchmaker ingest --results <ECCentric results.csv> --store features.jsonl --out data/`
   joins them into the training table and writes the manifest.
3. `qec-matchmaker train --data data/` (into the package's `qec_matchmaker/model/`, so it ships with
   the package) and `qec-matchmaker evaluate --data data/ --out reports/` retrain and re-evaluate.
4. `scripts/update_from_eccentric.sh` runs all three against ECCentric's committed sweep table (so the
   manifest names the ECCentric commit), with extraction on the cluster that holds the circuit files;
   ECCentric's daily refresh calls it and commits `data/` (training table, manifest, feature store
   `features.jsonl`), `qec_matchmaker/model/` and `reports/` when they changed.
