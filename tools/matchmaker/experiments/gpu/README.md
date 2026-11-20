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

## The model

ECCentric's time law (a power law in error mechanisms and detector firing rate) per decoder, plus quantile
trees on the circuit features and the decoder, fitted to log10 GPU ms per shot; the same model as the CPU times.
On codes held out of training (10 folds over 61 codes):

| | within 3× | median error | 10–90% interval coverage |
| --- | --- | --- | --- |
| Time law + trees (packaged) | 87% | 1.47× | 43% raw, 80% after widening by 10^0.195 |
| Trees alone | 85% | 1.63× | 47% raw |
| The circuit's measured CPU time ÷ its decoder's median speedup (needs a CPU decode) | 77% | 1.82× | |

Per decoder, within 3×: BP-OSD tuned 97%, BP-OSD 95%, Relay-BP 88%, GARI 84%, BP-LSD 79%, BeliefFind 78%.
The widening is calibrated on the same held-out predictions it is reported on. `qec-matchmaker train-gpu`
refits the model and adds it to a trained bundle; `qec-matchmaker train` does so whenever this directory exists.

## What it changes

`experiments/gpu_decisions.py` (`results/gpu_decisions.json`): the recommendation for each of the 5,624
training circuits, without and with a GPU.

| | accurate | cheapest |
| --- | --- | --- |
| Circuits recommended a GPU decoder | 1,090 (19%) | 1,995 (35%) |
| of which the same algorithm, on the GPU | 165 | 407 |
| Recommended decoder's time per shot, CPU choice ÷ GPU choice (median; 10th–90th percentile) | 188× (9–745×) | 133× (6–643×) |
| Circuits still recommended Tesseract (no GPU version) | 4,449 | 3,398 |

Under "accurate", 917 of the 925 changes of algorithm are circuits predicted above threshold, where every
objective takes the fastest decoder; the other 8 are ties in predicted accuracy. Times compare one GPU against
one CPU core, so they are latency per shot, not cost.

Limits: one GPU model (H200), single-GPU runs, and the 43 slowest pilots were skipped, so the model has not seen
the largest circuits on the GPU; `advise` warns when a circuit's features are outside the training range.
