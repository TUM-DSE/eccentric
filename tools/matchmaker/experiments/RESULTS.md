# Model-quality experiments

Held-out-code metrics for the regime, LER, regret and time models and the decoder choice, under variations of
the HistGradientBoosting setup.

**Setup.** `data/` as of ECCentric 79badf0: 5965 decodes, 1633 circuits, 54 codes, 7 decoders. Same split as
`evaluate.py`: 10-fold `GroupKFold` over codes, no shuffle, `random_state=0`, and the same metric definitions.
The scripts use a frozen copy of `train` / `predict` / the metrics from ec4f622 (`experiments/mq_core.py`), so the
concurrent edits to `models.py` cannot move these numbers. That copy equals `evaluate(calibrate=False)`; the new
conformal widening only changes q10/q90. The baseline reproduces `reports/metrics.json` exactly. HGB without early
stopping is deterministic, so differences come from the configuration and not from a seed; they can still be
specific to this split. **+ / −** marks a metric whose paired bootstrap over codes (300 resamples of the 54 codes,
the same resample for both configurations) gives a 95% interval of the difference from the baseline that excludes 0,
meaning significantly better or worse. Across ~200 comparisons (drop table), a few isolated marks are expected by
chance.

**Only 16 of 24 features are filled.** `mechanism_weight_max`, `hyperedge_fraction`, `det_fire_p90`,
`logical_mass`, `graphlike`, `color_annotated`, `rounds` and `decomposable` are NaN everywhere, so the models ignore
them. Consequences here: two of the five requested LER-monotone features (`det_fire_p90`, `logical_mass`) could not
be constrained; the importance results say nothing about the missing eight; and `applicable()` rules out mwpm (208
measured decodes) and chromobius (18), so the decoder choice never picks them.

Columns: regime acc = accuracy of P(above) > 0.5. LER med / p90 / Spearman = median and p90 |log10 error| of
q50 best LER, and its rank correlation. time med / p90 / ≤3x = the same for q50 ms per shot, and the share
within 3x. regret p90 = p90 |log10 error| of regret. choice good = good-enough share of the `accurate` /
`cheapest` recommendation. LER / time cov = raw share of truths inside [q10, q90], nominal 80%, before conformal
widening.

## 1. Hyperparameters

Baseline: learning_rate 0.05, max_iter 200, max_leaf_nodes 15, min_samples_leaf 20, l2_regularization 1.0.
Stage A changed one parameter at a time; stage B combined the capacity increases. Each row sets the parameters for all four models.

| config | regime acc | LER med | LER p90 | LER Spearman | time med | time p90 | time ≤3x | regret p90 | choice good (accurate) | choice good (cheapest) | LER cov | time cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.964 | 0.111 | 0.566 | 0.968 | 0.171 | 0.699 | 0.834 | 0.259 | 0.904 | 0.866 | 0.592 | 0.694 |
| lr0.03 | 0.964 | 0.105 | 0.567 | 0.968 | 0.184- | 0.757- | 0.819- | 0.265 | 0.899- | 0.874 | 0.646 | 0.749 |
| lr0.1 | 0.961 | 0.105 | 0.546 | 0.970 | 0.167 | 0.674 | 0.839 | 0.271 | 0.901 | 0.869 | 0.559 | 0.638 |
| it400 | 0.961 | 0.107 | 0.562 | 0.971+ | 0.167 | 0.668+ | 0.843+ | 0.259 | 0.906 | 0.870 | 0.545 | 0.635 |
| leaves7 | 0.960 | 0.120 | 0.634- | 0.962- | 0.198- | 0.830- | 0.786- | 0.264 | 0.905 | 0.872 | 0.639 | 0.736 |
| leaves31 | 0.963 | 0.105 | 0.540 | 0.971+ | 0.165+ | 0.705 | 0.840 | 0.270 | 0.904 | 0.865 | 0.539 | 0.648 |
| msl10 | 0.958- | 0.112 | 0.553 | 0.968 | 0.173 | 0.731- | 0.830 | 0.272 | 0.902 | 0.867 | 0.605 | 0.682 |
| msl40 | 0.956- | 0.108 | 0.549 | 0.968 | 0.170 | 0.721 | 0.833 | 0.271- | 0.907 | 0.874 | 0.591 | 0.693 |
| it400_leaves31 | 0.962 | 0.102 | 0.525 | 0.972+ | 0.159+ | 0.692 | 0.843 | 0.275 | 0.900 | 0.865 | 0.491 | 0.579 |
| lr0.1_it400 | 0.960 | 0.103 | 0.546 | 0.972+ | 0.161+ | 0.667 | 0.845+ | 0.267 | 0.904 | 0.873 | 0.542 | 0.610 |
| lr0.1_leaves31 | 0.962 | 0.103 | 0.529 | 0.972+ | 0.156+ | 0.669 | 0.848+ | 0.276- | 0.906 | 0.868 | 0.498 | 0.595 |
| lr0.1_it400_leaves31 | 0.960- | 0.102 | 0.508+ | 0.972+ | 0.154+ | 0.666 | 0.853+ | 0.280- | 0.903 | 0.868 | 0.453 | 0.562 |
| l2_0 | 0.957- | 0.110 | 0.572 | 0.968 | 0.174 | 0.714 | 0.831 | 0.257 | 0.903 | 0.873 | 0.612 | 0.686 |

Stage A and B show that capacity helps the LER and time models but not the regime classifier or the regret
model: regime accuracy and regret p90 drift the wrong way, and changing min_samples_leaf or l2 either way costs
regime accuracy. The final stage therefore raises capacity for the **LER and time models only** and keeps the
regime and regret models at the baseline. `cap` = learning_rate 0.1, max_leaf_nodes 31, max_iter 400; `cap200`
uses 200 iterations; `cap63` uses 63 leaves. `cap_drop_em` and `drop_em_iv` (baseline parameters) drop
`error_mass`, and `cap_drop_em_det` also drops `detectors` (see section 3). `base_iv` is the baseline re-run so that
the interval bounds are kept.

| config | regime acc | LER med | LER p90 | LER Spearman | time med | time p90 | time ≤3x | regret p90 | choice good (accurate) | choice good (cheapest) | LER cov | time cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_iv | 0.964 | 0.111 | 0.566 | 0.968 | 0.171 | 0.699 | 0.834 | 0.259 | 0.904 | 0.866 | 0.592 | 0.694 |
| cap200_ler_time | 0.964 | 0.103 | 0.529 | 0.972+ | 0.156+ | 0.669 | 0.848+ | 0.259 | 0.905 | 0.868 | 0.498 | 0.595 |
| cap_ler_time | 0.964 | 0.102 | 0.508+ | 0.972+ | 0.154+ | 0.666 | 0.853+ | 0.259 | 0.904 | 0.868 | 0.453 | 0.562 |
| cap63_ler_time | 0.964 | 0.096 | 0.500+ | 0.973+ | 0.154+ | 0.677 | 0.848+ | 0.259 | 0.902 | 0.865 | 0.422 | 0.520 |
| cap_drop_em | 0.964 | 0.102 | 0.508+ | 0.972 | 0.153+ | 0.638 | 0.851+ | 0.264 | 0.905 | 0.868 | 0.498 | 0.557 |
| cap_drop_em_det | 0.961 | 0.097 | 0.526 | 0.972+ | 0.152+ | 0.628+ | 0.851+ | 0.262 | 0.904 | 0.871 | 0.459 | 0.557 |
| drop_em_iv | 0.964 | 0.103 | 0.556 | 0.970 | 0.167 | 0.691 | 0.836 | 0.264 | 0.906 | 0.866 | 0.612 | 0.675 |

Interval quality on a common footing: the symmetric widening that makes [q10, q90] hold 80% of the held-out
truths is applied as an oracle conformal step (comparable across configurations only). The table shows the
resulting median width in decades.

| config | LER raw coverage | LER raw median width | LER width at 80% | time raw coverage | time raw median width | time width at 80% |
|---|---|---|---|---|---|---|
| base_iv | 59% | 0.37 | 0.53 | 69% | 0.62 | 0.75 |
| cap200_ler_time | 50% | 0.25 | 0.48 | 59% | 0.41 | 0.64 |
| cap_ler_time | 45% | 0.23 | 0.51 | 56% | 0.38 | 0.63 |
| cap63_ler_time | 42% | 0.21 | 0.51 | 52% | 0.35 | 0.64 |
| cap_drop_em | 50% | 0.23 | 0.49 | 56% | 0.37 | 0.62 |
| cap_drop_em_det | 46% | 0.23 | 0.52 | 56% | 0.38 | 0.62 |
| drop_em_iv | 61% | 0.38 | 0.51 | 67% | 0.62 | 0.75 |

Rough wall time of one 10-fold evaluation with 6 workers (some runs overlapped with others): baseline 31 s,
`cap200` 61 s, `cap` 123 s. Training the LER and time models costs roughly 2x (`cap200`) or 3-4x (`cap`) the
baseline.

## 2. Monotonic constraints

`mono_ler` constrains the LER model to be non-decreasing in det_fire_mean, det_fire_median and
error_mass_per_detector (det_fire_p90 and logical_mass are NaN). `mono_ler_regime` adds the same constraints on
P(above). `mono_time` makes time non-decreasing in mechanisms and detectors. `mono_all` = all three. `mono_full`
constrains every feature that moves when the noise rises (adds det_fire_max and error_mass) or when the circuit
grows (adds qubits, measurements, two_qubit_gates, error_mass and ticks to time).

| config | regime acc | LER med | LER p90 | LER Spearman | time med | time p90 | time ≤3x | regret p90 | choice good (accurate) | choice good (cheapest) | LER cov | time cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.964 | 0.111 | 0.566 | 0.968 | 0.171 | 0.699 | 0.834 | 0.259 | 0.904 | 0.866 | 0.592 | 0.694 |
| mono_ler | 0.964 | 0.111 | 0.596 | 0.961- | 0.171 | 0.699 | 0.834 | 0.259 | 0.904 | 0.866 | 0.608 | 0.694 |
| mono_ler_regime | 0.958 | 0.111 | 0.596 | 0.961- | 0.171 | 0.699 | 0.834 | 0.259 | 0.905 | 0.865 | 0.608 | 0.694 |
| mono_time | 0.964 | 0.111 | 0.566 | 0.968 | 0.173 | 0.709 | 0.829 | 0.259 | 0.904 | 0.865 | 0.592 | 0.692 |
| mono_all | 0.958 | 0.111 | 0.596 | 0.961- | 0.173 | 0.709 | 0.829 | 0.259 | 0.905 | 0.863 | 0.608 | 0.692 |
| mono_full | 0.957 | 0.121 | 0.585 | 0.962- | 0.171 | 0.685 | 0.836 | 0.259 | 0.901 | 0.861- | 0.635 | 0.703 |

How often the models move the wrong way, measured on held-out codes. Single features ×1.25 on the baseline:
raising det_fire_median alone lowers the predicted LER for 33% of circuits, det_fire_mean 12%,
error_mass_per_detector 11%. Raising mechanisms alone lowers the predicted time for 13% of decodes, detectors 19%
(worst −0.41 decades). Single-feature bumps leave the data manifold, so the realistic test scales all noise-rate
features together (as a higher physical error rate does) or all size features together:

| config | LER q50 wrong way (x1.1 / x1.5 noise) | worst LER drop (dec) | P(above) wrong way (x1.1 / x1.5) | time q50 wrong way (x1.1 / x1.5 size) | worst time drop (dec) |
|---|---|---|---|---|---|
| baseline | 18.8% / 16.0% | -0.20 | 22.8% / 23.1% | 18.7% / 6.0% | -0.50 |
| mono_all | 17.4% / 17.9% | -0.11 | 16.0% / 16.4% | 17.4% / 5.4% | -0.49 |
| mono_full | 6.4% / 5.0% | -0.12 | 0.0% / 0.0% | 4.4% / 1.3% | -0.31 |

The requested constraint set barely helps (16-18% of LER predictions still go the wrong way), because the
unconstrained det_fire_max and error_mass still pull them down. Only `mono_full` makes P(above) exactly monotone.
The LER and time models are never exactly monotone, because **scikit-learn 1.9.1 does not enforce `monotonic_cst`
under `loss="quantile"`**: after each tree it refits the leaf values to residual quantiles
(`_update_leaves_values`) without re-applying the bounds. A synthetic check confirms this: squared error gives 0
violations along the constrained feature, while the q50 / q90 quantile models give 3 / 6 decreasing steps on a
400-point grid.

## 3. Feature importance

Permutation importance on each fold's held-out codes (5 repeats; a feature is permuted among the fold's held-out
circuits), as the mean loss of accuracy. Baseline: regime acc 0.964, log-loss 0.122, LER MAE 0.222, time MAE 0.291,
time ≤3x 0.834. The size and noise features are strongly correlated, so single-feature numbers understate shared
information; the group rows show it.

| permuted | regime acc drop | regime log-loss rise | LER MAE rise (dec) | time MAE rise (dec) | time ≤3x drop |
|---|---|---|---|---|---|
| *noise rate group*: det_fire_*, error_mass_per_detector | +0.358 | +1.944 | +0.868 | +0.267 | +0.252 |
| *size group*: detectors, mechanisms, qubits, measurements, two_qubit_gates, error_mass | +0.024 | +0.048 | +0.083 | +0.834 | +0.540 |
| error_mass | +0.003 | -0.001 | +0.015 | +0.588 | +0.422 |
| det_fire_max | +0.058 | +0.183 | +0.310 | +0.139 | +0.156 |
| error_mass_per_detector | +0.015 | +0.031 | +0.272 | +0.089 | +0.120 |
| mechanisms | +0.004 | +0.005 | +0.009 | +0.192 | +0.225 |
| observables | +0.041 | +0.111 | +0.047 | +0.016 | +0.013 |
| mechanism_weight_mean | +0.004 | +0.005 | +0.012 | +0.100 | +0.121 |
| det_fire_mean | +0.017 | +0.030 | +0.069 | +0.005 | +0.005 |
| det_fire_median | +0.017 | +0.034 | +0.031 | +0.003 | +0.004 |
| qubits | +0.020 | +0.037 | +0.016 | +0.002 | +0.002 |
| swaps | +0.001 | +0.001 | +0.004 | +0.042 | +0.053 |
| detectors | +0.001 | +0.003 | +0.013 | +0.001 | +0.001 |
| measurements | -0.000 | -0.001 | +0.016 | +0.002 | +0.002 |
| two_qubit_gates | +0.002 | +0.001 | +0.011 | +0.003 | +0.002 |
| ticks | +0.000 | -0.001 | +0.004 | +0.013 | +0.011 |
| undetectable_probability | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| undetectable_faults | +0.000 | +0.000 | -0.000 | -0.000 | +0.000 |

Drop-one retraining (baseline parameters; full table with all columns in `results/table_drop.md`):

| config | regime acc | LER med | LER p90 | LER Spearman | time med | time p90 | time ≤3x | regret p90 | choice good (accurate) | choice good (cheapest) | LER cov | time cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.964 | 0.111 | 0.566 | 0.968 | 0.171 | 0.699 | 0.834 | 0.259 | 0.904 | 0.866 | 0.592 | 0.694 |
| drop_detectors | 0.961 | 0.107 | 0.548 | 0.969 | 0.165 | 0.696 | 0.835 | 0.264 | 0.903 | 0.877+ | 0.595 | 0.684 |
| drop_observables | 0.946- | 0.150- | 0.592 | 0.940- | 0.173 | 0.734 | 0.830 | 0.264 | 0.909 | 0.873 | 0.551 | 0.683 |
| drop_mechanisms | 0.963 | 0.109 | 0.564 | 0.968 | 0.183- | 0.709 | 0.828 | 0.269 | 0.904 | 0.868 | 0.605 | 0.674 |
| drop_mechanism_weight_mean | 0.963 | 0.115 | 0.575 | 0.966- | 0.181- | 0.763- | 0.818- | 0.259 | 0.905 | 0.868 | 0.573 | 0.678 |
| drop_error_mass | 0.964 | 0.103 | 0.556 | 0.970 | 0.167 | 0.691 | 0.836 | 0.264 | 0.906 | 0.866 | 0.612 | 0.675 |
| drop_error_mass_per_detector | 0.961 | 0.114 | 0.596 | 0.963- | 0.176 | 0.734 | 0.824 | 0.265 | 0.906 | 0.872 | 0.607 | 0.673 |
| drop_det_fire_mean | 0.957- | 0.107 | 0.545 | 0.968 | 0.173 | 0.700 | 0.835 | 0.267 | 0.903 | 0.859 | 0.605 | 0.693 |
| drop_det_fire_median | 0.961 | 0.106 | 0.582 | 0.967 | 0.170 | 0.701 | 0.834 | 0.260 | 0.906 | 0.873 | 0.583 | 0.684 |
| drop_det_fire_max | 0.963 | 0.109 | 0.586 | 0.967 | 0.172 | 0.713 | 0.829 | 0.260 | 0.903 | 0.868 | 0.583 | 0.684 |
| drop_undetectable_faults | 0.964 | 0.111 | 0.566 | 0.968 | 0.172 | 0.701 | 0.835 | 0.259 | 0.904 | 0.866 | 0.592 | 0.694 |
| drop_undetectable_probability | 0.964 | 0.110 | 0.571 | 0.968 | 0.171 | 0.704 | 0.830- | 0.257 | 0.904 | 0.871 | 0.587 | 0.693 |
| drop_qubits | 0.958 | 0.111 | 0.584 | 0.966 | 0.170 | 0.694 | 0.836 | 0.263 | 0.902 | 0.872 | 0.570 | 0.691 |
| drop_two_qubit_gates | 0.963 | 0.108 | 0.562 | 0.969 | 0.169 | 0.700 | 0.835 | 0.261 | 0.906 | 0.868 | 0.582 | 0.680 |
| drop_swaps | 0.964 | 0.110 | 0.572 | 0.968 | 0.175 | 0.705 | 0.830 | 0.260 | 0.903 | 0.874 | 0.580 | 0.685 |
| drop_ticks | 0.962 | 0.110 | 0.561 | 0.969 | 0.170 | 0.698 | 0.833 | 0.262 | 0.901 | 0.871 | 0.602 | 0.683 |
| drop_measurements | 0.962 | 0.108 | 0.554 | 0.969 | 0.171 | 0.714 | 0.832 | 0.264 | 0.903 | 0.877+ | 0.611 | 0.686 |

What carries the predictions: the **noise-rate group** carries the regime and LER models (permuting it costs
36 points of regime accuracy and 0.87 decades of LER MAE); the strongest single features are det_fire_max and
error_mass_per_detector. **observables** is the one feature that cannot be dropped: without it, regime
accuracy falls from 96.4% to 94.6% and LER median error rises from 0.111 to 0.150. The **size group** carries the
time model; the strongest single features are error_mass, mechanisms, det_fire_max and mechanism_weight_mean.
No feature is significantly harmful. Dropping `error_mass` (redundant: error_mass_per_detector × detectors)
gives LER median 0.103 and Spearman 0.970 against 0.111 and 0.968, without a significant mark. `undetectable_faults`
and `undetectable_probability` contribute exactly nothing: they are nearly constant in this data, with 4 and 16
distinct values.

## 4. Learning curve

Each fold trains on a random subset of its training codes, or on all codes but a share of each code's circuits.
The test folds are unchanged. Means over 3 subsamples; across subsamples the spread is at most 0.013 in regime acc,
0.06 decades in LER p90 and 0.02 in time ≤3x.

| training data per fold | regime acc | LER med | LER p90 | LER Spearman | time med | time p90 | time ≤3x | regret p90 | choice good (accurate) |
|---|---|---|---|---|---|---|---|---|---|
| 25% of codes (~12 codes) | 0.925 | 0.176 | 0.716 | 0.931 | 0.237 | 0.961 | 0.740 | 0.318 | 0.903 |
| 50% of codes (~24 codes) | 0.948 | 0.125 | 0.636 | 0.959 | 0.189 | 0.763 | 0.808 | 0.291 | 0.899 |
| 75% of codes (~36 codes) | 0.954 | 0.121 | 0.604 | 0.963 | 0.181 | 0.736 | 0.817 | 0.271 | 0.906 |
| 100% (baseline, ~49 codes) | 0.964 | 0.111 | 0.566 | 0.968 | 0.171 | 0.699 | 0.834 | 0.259 | 0.904 |
| 25% of circuits per code (all codes) | 0.947 | 0.127 | 0.666 | 0.954 | 0.196 | 0.797 | 0.795 | 0.296 | 0.899 |
| 50% of circuits per code (all codes) | 0.954 | 0.109 | 0.572 | 0.964 | 0.182 | 0.732 | 0.821 | 0.282 | 0.905 |

There is no plateau. From 75% to 100% of the codes, regime accuracy rises from 95.4% to 96.4%, LER p90 error falls
from 0.604 to 0.566, time ≤3x rises from 81.7% to 83.4%, and regret p90 falls from 0.271 to 0.259. At the same
number of circuits, more codes beat more circuits per code: 50% of codes gives LER p90 0.636 against 0.572 for 50%
of each code's circuits. For the LER, half of each code's circuits already does as well as all of them (median 0.109,
p90 0.572). The decoder-choice score stays flat at 90% from 25% of the codes up.

**Why the decoder-choice score never moves.** In every experiment it stayed between 0.89 and 0.91 (`accurate`)
and between 0.86 and 0.88 (`cheapest`), even with 25% of the data. Tesseract is good enough on 99.9% of the
circuits where it was run, and the recommender picks it on about 60% of circuits. 32% of the choices (429 of 1336) were not run on the circuit and are scored as
the BP-OSD fallback, which is good enough on 74%. The score measures ECCentric's coverage more than model quality.

## Recommendations

- **LER and time models: adopt learning_rate 0.1, max_leaf_nodes 31, max_iter 400.** Keep min_samples_leaf 20 and
  l2_regularization 1.0. LER median error falls from 0.111 to 0.102, p90 from 0.566 to 0.508 and Spearman rises from
  0.968 to 0.972; time median error falls from 0.171 to 0.154 and ≤3x rises from 83.4% to 85.3%. The LER p90,
  Spearman, time median and ≤3x changes are significant; regime, regret and the decoder choice do not change.
  If training time matters, max_iter 200 keeps most of the gain (LER p90 0.529, time ≤3x 84.8%). Do not go to 63 leaves: LER improves a little more, but time ≤3x and relay-BP time get worse.
- **Regime and regret models: keep the current parameters** (0.05 / 200 / 15 / 20 / 1.0). Extra capacity lowers
  regime accuracy (0.964 to 0.960) and raises regret p90 (0.259 to 0.280), and changing min_samples_leaf (10 or 40)
  or l2 (0) costs 0.6-0.8 points of regime accuracy. This needs per-model parameters in `models.py`, where `PARAMS`
  is currently shared by all models.
- **Ship the capacity change only together with the conformal widening (`calibrate=True`).** Raw [q10, q90]
  coverage falls from 59% to 45% (LER) and from 69% to 56% (time). Once both are widened to 80% coverage, the new
  intervals are narrower than the baseline's: 0.51 against 0.53 decades for LER, 0.63 against 0.75 for time.
- **Do not adopt `monotonic_cst`.** The requested constraints cost accuracy: LER p90 rises from 0.566 to 0.596,
  Spearman falls from 0.968 to 0.961 (significant), regime accuracy falls by 0.6 points and time ≤3x by 0.5. They also
  fix almost none of the joint wrong-way moves, and scikit-learn does not enforce them for the quantile models anyway.
  If monotone outputs matter, for example for p-sweeps in `advise`, constrain only the regime classifier with the
  full noise set: P(above) becomes exactly monotone, at 96.4% to 95.7% accuracy (not significant). Alternatively,
  enforce monotonicity after prediction.
- **Keep the feature set; fill the eight missing features instead.** No feature is significantly harmful
  (`error_mass` could go for simplicity, with no loss). The predictive work is done by noise-rate features, which is
  the kind det_fire_p90 and logical_mass are. `decomposable` and `color_annotated` are needed before mwpm and
  chromobius can be recommended at all.
- **More ECCentric results are worth waiting for, above all new codes.** The learning curve has not flattened:
  each doubling of the training codes buys about 1.6 points of regime accuracy, 0.07 decades of LER p90 and 2.6 points
  of time ≤3x. To make the decoder-choice score useful for tuning, also score it on circuits where every
  applicable decoder was run (599 circuits have all five BP and tesseract decoders), or without fallbacks.

## Files

`experiments/mq_core.py` (frozen, parameterised training, prediction and metrics; paired code bootstrap),
`experiments/model_quality.py` (`run <stage>` for hp, hp_b, final, final2, mono, drop, curve and curve_circuits;
`importance`), `experiments/probe_joint.py` (joint monotonicity probe), `experiments/report.py` (tables).
Per-stage CSVs and full tables are in `experiments/results/`; per-unit held-out results are cached in
`experiments/cache/` (git-ignored).
