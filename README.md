# qec-matchmaker

**Pick a decoder before you decode.** Matchmaker reads a noisy QEC circuit and predicts, in seconds
and without decoding a single shot, whether the configuration is above threshold, roughly what its
logical error rate will be, how long each decoder will take, and which decoder to use.

> **Status: working prototype.** Features, models, recommendation, evaluation, command line and the
> update pipeline from ECCentric work and are tested. The packaged model is trained on Matchmaker's own
> features of 3,146 ECCentric circuits (20,270 decodes, ECCentric 0e9c091). See [docs/DESIGN.md](docs/DESIGN.md).

## Why

Choosing a decoder is usually a guess, and a costly one. In the [ECCentric](https://github.com/TUM-DSE/eccentric)
benchmark (67 codes compiled onto 25 devices under 27 noise models, decoded by up to 9 decoders on
the same shots) no decoder wins everywhere, decoding costs span six orders of magnitude, and 48% of the CPU time
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
qec-matchmaker advise circuit.stim --distance 11                # the code distance, if known (better beyond the training data)
qec-matchmaker advise circuit.stim --objective budget --budget-ms 1 [--per-round]
qec-matchmaker advise trace.h5                             # a decoder-bench trace
qec-matchmaker features circuit.stim                       # the features the models see
```

```
$ qec-matchmaker advise surface_d5_p0.001.stim --distance 5
above threshold: 0%
logical error rate (best decoder): 7.19e-05 (80% interval 4.3e-05 - 2.3e-04)
recommended decoder: tesseract (cheapest (0.549 ms per shot) of the decoders predicted within 1.25x of the lowest LER (7.2e-05); its predicted LER is 7.2e-05)

    decoder  applicable      LER  ms/shot
       mwpm        True 0.000146  0.00722
bposd_tuned        True 9.36e-05    0.913
   relay_bp        True 9.51e-05    0.746
  tesseract        True 7.19e-05    0.549
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
   shot, with timed-out decodes as right-censored observations. The LER and time trees sit on top of
   physical scaling laws (below threshold the LER falls as (p/p_th)^(d/2) with the code distance d, if
   given; time is a power law in the circuit's size and noise), so that circuits larger or quieter than
   any in training are not predicted at the edge of the training data
   ([experiments/EXTRAPOLATION.md](experiments/EXTRAPOLATION.md)).
3. **Recommendation** (`qec_matchmaker/recommend.py`): the most accurate, the cheapest within a tolerance
   of the most accurate, or the most accurate within a time budget, among the decoders that apply;
   above threshold, the cheapest, with a warning. Whether matching applies needs the circuit (an error
   model alone cannot tell), so pass circuits where you can.

## How good it is

Evaluated on codes held out of training (10 folds over 67 codes, 3,146 circuits, 20,270 decodes; full
report in [reports/REPORT.md](reports/REPORT.md)):

| Prediction | Result |
|---|---|
| Above or below threshold | 98% correct (recall 95%, precision 96%) |
| Logical error rate of the best decoder | median error 0.08 decades (1.20×), p90 0.37, rank correlation 0.98 |
| Time per shot | 94% within 3× (Relay-BP 98%, BP-OSD 95%, Tesseract 85%, MWPF 82%), rank correlation 0.99 |
| Decoder choice, all 2,774 circuits ("accurate" / "cheapest") | good enough on 90% / 83%, against 61% for always BP-OSD |
| Decoder choice, the 1,820 circuits where all five BP-type and Tesseract decoders ran | "accurate": good enough on 97% at 1.7× the cheapest good-enough decoder's time (median); "cheapest": 84% at 1.0×; always Tesseract: 100% at 12×; always BP-OSD: 52% at 14× |

The 80% intervals are conformalized on held-out codes and hold 84% of held-out LERs and 81–87% of each
decoder's times. Harder splits: whole code families, devices or noise models held out
([experiments/GENERALIZATION.md](experiments/GENERALIZATION.md)); circuits beyond the training range
([experiments/EXTRAPOLATION.md](experiments/EXTRAPOLATION.md): the quietest 20% of circuits held out, LER
error 0.30 decades, against 0.74 for trees alone; these two reports are from the 1,810-circuit table);
circuits from Stim's generators and decoder-bench ([experiments/EXTERNAL.md](experiments/EXTERNAL.md): LER
error 0.08 decades, p90 0.28, time within 3× for 84%, decoder choice good enough on 100% / 94%).
Model settings come from the experiments in [experiments/RESULTS.md](experiments/RESULTS.md).

Known weaknesses: Tesseract's time is the hardest to predict (85% within 3×); beyond the training data the 80% intervals
are too narrow (they are calibrated on held-out codes); a noise model unlike any in training can mislead
the time law (Harvard's atom loss held out: time within 3× 79% -> 60%); and `advise` warns, naming the
features, when a circuit is outside the training range.

## Keeping up with ECCentric

ECCentric keeps producing results. One command brings Matchmaker up to date:

```bash
scripts/update_from_eccentric.sh          # MM_PUSH=1 to push; see the script for settings
```

It extracts the features of new circuits on the cluster that holds ECCentric's run directories
(`qec-matchmaker extract`, incremental), rebuilds the training table from ECCentric's committed sweep
table (`qec-matchmaker ingest`; the manifest records the ECCentric commit and file hashes), retrains
(`train`) and re-evaluates (`evaluate`), and commits `data/`, the packaged model and `reports/` when they
changed. It is run by hand after ECCentric's sweep table changes; ECCentric's daily refresh does not run it.

## Decoders

Trained on: PyMatching (MWPM), MWPF, BP-OSD with ECCentric's paper settings and with tuned BP (100
iterations, min-sum scaling 0.625), BP-LSD and BeliefFind (ldpc), Relay-BP, Tesseract and Chromobius.
MWPF has the least data (83 circuits).

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

MIT ([LICENSE](LICENSE)). If you use Matchmaker in your research, please cite it, and ECCentric, whose
measurements train it ([CITATION.cff](CITATION.cff), which GitHub's "Cite this repository" button reads):

```bibtex
@software{giortamis2026matchmaker,
  title  = {qec-matchmaker: pick a decoder before you decode},
  author = {Giortamis, Emmanouil},
  year   = {2026},
  url    = {https://github.com/manosgior/qec-matchmaker}
}

@inproceedings{swierkowska2026eccentric,
  title     = {ECCentric: An Empirical Analysis of Quantum Error Correction Codes},
  author    = {{\'S}wierkowska, Aleksandra and Pflieger, Jannik and Giortamis, Emmanouil and Bhatotia, Pramod},
  booktitle = {Proceedings of the ACM SIGMETRICS Conference (SIGMETRICS'26)},
  year      = {2026},
  note      = {arXiv:2511.01062}
}
```
