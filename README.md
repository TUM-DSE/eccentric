# qec-matchmaker

**Pick a decoder before you decode.** Matchmaker reads a noisy QEC circuit and predicts, in seconds
and without decoding a single shot, whether the configuration is above threshold, roughly what its
logical error rate will be, how long each decoder will take, and which decoder to use.

> **Status: working prototype.** Features, models, recommendation, evaluation, command line and the
> update pipeline from ECCentric work and are tested. The packaged model is trained on a first table
> built from features ECCentric had already computed (16 of the 24 features); the next update replaces
> it with Matchmaker's own extraction of every circuit. See [docs/DESIGN.md](docs/DESIGN.md).

## Why

Choosing a decoder is usually a guess, and a costly one. In the [ECCentric](https://github.com/TUM-DSE/eccentric)
benchmark (58 codes compiled onto 25 devices under 12 noise models, decoded by 7 decoders on the same
shots) no decoder wins everywhere, decoding costs span six orders of magnitude, and 48% of the CPU time
went to configurations with a logical error rate above 50%, where every decoder fails. All of this is
predictable from the circuit alone.

## Install

```bash
pip install -e .                      # add [decoder-bench] to read decoder-bench traces, [test] for the tests
```

## Use

```bash
qec-matchmaker advise circuit.stim                         # cheapest decoder within 1.25x of the most accurate
qec-matchmaker advise circuit.stim --objective accurate
qec-matchmaker advise circuit.stim --objective budget --budget-ms 1 [--per-round]
qec-matchmaker advise trace.h5                             # a decoder-bench trace
qec-matchmaker features circuit.stim                       # the features the models see
```

```
$ qec-matchmaker advise surface_d5_p0.001.stim
above threshold: 0%
logical error rate (best decoder): 5.78e-04 (80% interval 1.7e-04 - 4.7e-03)
recommended decoder: mwpm (cheapest (0.0117 ms per shot) of the decoders predicted within 1.25x of the lowest LER (0.00058); its predicted LER is 0.00062)

    decoder  applicable      LER  ms/shot
       mwpm        True 0.000625   0.0117
      bposd        True 0.000612     4.58
  tesseract        True 0.000578     0.57
       ...
```

```python
import stim
from qec_matchmaker.api import advise

advice = advise(stim.Circuit.from_file("circuit.stim"), objective="accurate")
advice.decoder                 # e.g. "tesseract"
advice.above_threshold         # probability
advice.ler                     # (q10, q50, q90) of the best decoder's logical error rate
advice.time_per_shot_ms        # {decoder: ms}
advice.expected_cost(errors=100)   # shots and core-hours per decoder
```

Times are for one core of the machine ECCentric ran on. `qec_matchmaker.calibrate` measures speed
factors on yours (`advise(..., speed_factors=...)`, `--calibration`).

## How it works

1. **Features from the detector error model** (`qec_matchmaker/features.py`): the probability that
   each detector fires (exact for independent error mechanisms), the total and the logical error mass,
   the number and weight of the error mechanisms, undetectable logical faults, whether errors decompose
   into edges (matching) and carry color annotations (Chromobius), and the circuit's size. Routing,
   remote couplers and device noise show up in these, so the models apply to codes and devices they
   have not seen; they never see the code's name.
2. **Gradient-boosted trees** (`qec_matchmaker/models.py`): P(above threshold); quantiles of the best
   logical error rate; each decoder's LER relative to the best; quantiles of each decoder's time per
   shot, with timed-out decodes as right-censored observations.
3. **Recommendation** (`qec_matchmaker/recommend.py`): the most accurate, the cheapest within a tolerance
   of the most accurate, or the most accurate within a time budget, among the decoders that apply;
   above threshold, the cheapest, with a warning. Whether matching applies needs the circuit (an error
   model alone cannot tell), so pass circuits where you can.

## How good it is

Evaluated on codes held out of training (10 folds over 54 codes, 1,633 circuits, 5,965 decodes; full
report in [reports/REPORT.md](reports/REPORT.md)):

| Prediction | Result |
|---|---|
| Above or below threshold | 96% correct (recall 95%, precision 94%) |
| Logical error rate of the best decoder | median error 0.10 decades (1.26×), p90 0.51, rank correlation 0.97 |
| Time per shot | 85% within 3× (BP-LSD, BeliefFind 96%; Relay-BP 63%), rank correlation 0.97 |
| Decoder choice, all 1,336 circuits ("accurate" / "cheapest") | good enough on 90% / 87%, against 75% for always BP-OSD |
| Decoder choice, the 599 circuits where all five BP-type and Tesseract decoders ran | "accurate": good enough on 92% at the cheapest good-enough decoder's time (median); always Tesseract: 100% at 3.8× that time; always BP-OSD: 60% at 2.6× |

The 80% intervals are conformalized on held-out codes and hold 83% of held-out LERs and 77–90% of each
decoder's times (quantile regression alone held 59% and 69%). Model settings come from the experiments in
[experiments/RESULTS.md](experiments/RESULTS.md) (hyperparameters, monotonic constraints, feature
importance, learning curve); the learning curve shows more codes still improve every prediction.

Known weaknesses: in this first table the matching and color features are missing, so matching and
Chromobius are never recommended in the held-out evaluation (they are for new circuits, whose features
are complete); Relay-BP's time is the hardest to predict; and trees do not extrapolate: for a circuit
larger than any in training (say a distance-41 surface code) the predictions stay near the largest seen,
which `advise` reports as a warning naming the features outside the training range.

## Keeping up with ECCentric

ECCentric keeps producing results. One command brings Matchmaker up to date:

```bash
scripts/update_from_eccentric.sh          # MM_PUSH=1 to push; see the script for settings
```

It extracts the features of new circuits on the cluster that holds ECCentric's run directories
(`qec-matchmaker extract`, incremental), rebuilds the training table from ECCentric's committed sweep
table (`qec-matchmaker ingest`; the manifest records the ECCentric commit and file hashes), retrains
(`train`) and re-evaluates (`evaluate`), and commits `data/`, the packaged model and `reports/` when they
changed. ECCentric's daily refresh runs it.

## Decoders

Trained on: PyMatching (MWPM), BP-OSD, BP-LSD and BeliefFind (ldpc), Relay-BP, Tesseract and
Chromobius. Planned: MWPF.

## Limitations

- Timings are single-core software decoders, including the decoder's construction; hardware (FPGA/ASIC)
  decoders are not modelled.
- Training data are memory experiments; logical operations such as lattice surgery are future work.
- Predicted logical error rates are for planning, not for publication: they are far wider than the
  interval of a measured rate.
- decoder-bench traces keep only unique syndromes, so their syndromes cannot give a physical logical
  error rate; Matchmaker reads only their circuit or check matrix.

## Related work

- [ECCentric](https://github.com/TUM-DSE/eccentric): the end-to-end QEC benchmark whose measurements
  train Matchmaker.
- [decoder-bench](https://github.com/satvikmaurya/decoder-bench) (Maurya et al., IISWC 2025):
  benchmarks decoders on fixed decoding problems; Matchmaker predicts how they will do before running
  them.

## License and citation

To be decided.
