# GPU pilots and the GPU time model

With `--gpu` (`advise(..., gpu=True)`), Matchmaker plans for a machine with a GPU: the six decoders that have
GPU versions in ECCentric (`eccentric/decoders/gpu`: `bposd_gpu`, `bposd_tuned_gpu`, `bplsd_gpu`,
`belief_find_gpu`, `relay_bp_gpu`, `gari_gpu`) are predicted, and recommended, as those. They give the CPU
decoders' corrections, exactly or within statistical error (ECCentric's `analysis/GPU.md`), so the accuracy
predictions are the CPU decoders'; only the time model is new (`qec_matchmaker/gpu.py`).

## The pilots (18–19 November 2026)

306 circuits of ECCentric's sweep, below threshold, stratified by code family (`slice.json`), each decoded on
one NVIDIA H200 NVL by every GPU decoder that applies: 4,096 shots, then batches of up to 32,768 until 10^5
shots or 120 s, after building the decoder (timed) and warming it up on 64 shots. `prep.py` (4 CPU processes)
built the decoders' inputs ahead of the GPU; `runner.sh` ran `pilot_circuit.py` per circuit with a one-hour
limit; `results/<i>.jsonl` holds the timings. The scripts are as run on the cluster (paths included).

| | |
| --- | --- |
| Pilots | 1,191: 1,145 measured, 43 skipped (predicted over 1,500 s for 4,096 shots), 1 cut off by the hour limit |
| GPU time | 31 hours of decoding and building |
| Throughput against one CPU core (median) | Relay-BP 202×, BP-OSD 113×, BeliefFind 94×, GARI 93×, BP-LSD 79×, BP-OSD tuned 37× |
| Range | 2× to 4,000×: the speedup depends on the circuit, which is what the model learns |

Times are steady-state ms per shot on that GPU, CPU post-processing (BP-LSD's and BeliefFind's) included.
Building a decoder takes a further 0–0.7 s (median per decoder; `meta.json`, `gpu.build_seconds`).

## The GPU-only decoder (25 November 2026)

`tcbp_jobs_sc_gpu` (TCBP-JOBS, arXiv:2611.21321, preprint; ECCentric's `analysis/GPU.md`) has no CPU version, so
its accuracy cannot come from a CPU decoder's. It decoded the same 306 circuits (`../gpu_tcbp`: the settings and
shots of the circuits' CPU decode jobs, then this directory's timing protocol). `qec-matchmaker train` joins those
306 decodes to the training table (`--gpu-only`; the regret and best-LER models learn from them, the CPU time model
does not) and its 306 timings to the pilots below; without `--gpu` it is not predicted. On codes held out of
training (`../gpu_tcbp/heldout.py`, `heldout.json`), its predicted LER is within 3× of the measured one on 83% of
242 decodes (median error 0.15 decades), as for the CPU decoders on the same circuits (82%, 0.17 decades).

## The model

ECCentric's time law (a power law in error mechanisms and detector firing rate) per decoder, plus quantile
trees on the circuit features and the decoder, fitted to log10 GPU ms per shot; the same model as the CPU times.
On codes held out of training (10 folds over the codes; 1,451 pilots of 306 circuits, TCBP-JOBS's included):

| | within 3× | median error | 10–90% interval coverage |
| --- | --- | --- | --- |
| Time law + trees (packaged) | 93% | 1.29× | 50% raw, 80% after widening by 10^0.119 |
| Trees alone | 89% | 1.36× | 54% raw |
| The circuit's measured CPU time ÷ its decoder's median speedup (needs a CPU decode; CPU-version decoders only) | 81% | 1.79× | |

Per decoder, within 3×: TCBP-JOBS 100%, BP-OSD tuned 98%, BP-OSD 97%, Relay-BP 93%, GARI 89%, BP-LSD 84%,
BeliefFind 83%. (Before TCBP-JOBS and the corrected slice, 22 November: 87% within 3×, 1.47×.) The widening is
calibrated on the same held-out predictions it is reported on. `qec-matchmaker train-gpu` refits the time model
on the pilots only (not the GPU-only decoders, whose accuracy needs `train`); `qec-matchmaker train` does both
whenever these directories exist.

## What it changes

`experiments/gpu_decisions.py` (`results/gpu_decisions.json`): the recommendation for each of the 5,624
training circuits, without and with a GPU.

| | accurate | cheapest |
| --- | --- | --- |
| Circuits recommended a GPU decoder | 5,306 (94%) | 5,084 (90%) |
| of which TCBP-JOBS | 5,227 | 4,811 |
| of which the CPU choice's algorithm, on the GPU | 66 | 77 |
| Recommended decoder's time per shot, CPU choice ÷ GPU choice (median; 10th–90th percentile) | 583× (15–36,000×) | 433× (13–18,000×) |
| Circuits still recommended Tesseract | 154 | 123 |

Before TCBP-JOBS (22 November) a GPU decoder was recommended for 1,087 / 1,781 circuits and Tesseract kept 4,349 /
3,377: TCBP-JOBS matched Tesseract's error rate on all 181 slice circuits Tesseract measured, at GPU speed, so it
takes Tesseract's place. Its accuracy was measured on the 306 slice circuits only, all below threshold: the
predictions for the other 5,318 circuits extrapolate from them (the held-out-code check covers new codes, not
other regimes or larger circuits); decoding the whole table with it is planned. Decoders predicted to time out
(over 108 s per shot) are not recommended, on either device. Times compare one GPU against one CPU core, so they
are latency per shot, not cost.

Limits: one GPU model (H200), single-GPU runs, and the 43 slowest pilots were skipped, so the model has not seen
the largest circuits on the GPU; `advise` warns when a circuit's features are outside the training range.

## Correction (24 November 2026)

`slice.json` named the wrong circuit for 39 of the 306 circuits: another run's circuit of the same file name
(sweep1, sweep_bicycle or sweep_atoms, without idle noise) instead of sweep_idle's, which the CPU decoders decoded
(decode jobs of runs such as sweep_bposd_tuned name their circuit in `circuit_file`). `fix_slice.py` set each
entry's `file` to its decode jobs' circuit (42 entries changed, 3 of them to an identical copy; the old path is in
`file_before_fix`). The pilots of the 39 are redone on the corrected file; the first results are kept in
`results/stale_wrong_circuit/` on the cluster.
