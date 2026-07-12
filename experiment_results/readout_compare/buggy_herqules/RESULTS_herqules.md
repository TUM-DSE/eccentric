# Readout length vs logical-CNOT LER — HERQULES (realistic) readout model

Companion to `RESULTS.md`. Here the readout **flip error and readout duration move
together**: each readout length uses the HERQULES per-length flip error AND that same
length as the M-gate duration (decoherence). This is the realistic case — you cannot
have arbitrarily short readout, because short readout has poor fidelity.

Runners: `compare_readout_eccentric.py`, `synchronization-artifact/compare_readout_artifact.py`
with `READOUT_MODE="herqules"`. Data: `combined_readout_ler_herqules.csv`.

Everything else identical to RESULTS.md: SOTA ibm_boston gates/coherence, current vs
futuristic (errors/10, T1/T2×3), idle_mult ECCentric=1.0 / Artifact=3.0, distances 5–13,
MWPM, logical CNOT (fail = any of 3 observables wrong).

HERQULES readout flip error (measure_error = 1 − geomean(Q1,Q3,Q4,Q5), Q2 excluded;
1000 ns row updated to Q1,Q3,Q4,Q5 = 0.974,0.955,0.958,0.982):

| ns | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| measure_error | 0.2210 | 0.0507 | 0.0340 | 0.0337 | 0.0328 |

## LER vs readout length — `current`
```
ECCENTRIC (idle×1)                              ARTIFACT (idle×3)
 d |  200    400    600    800   1000            d |   200      400      600      800     1000
 5 | .7485  .5177  .4403  .4515  .4545           5 | 0.4156  0.01067  0.00669  0.00899  0.01221
 7 | .7481  .7376  .7280  .7294  .7260           7 | 0.4551  0.00484  0.00259  0.00398  0.00580
 9 | .7513  .7479  .7518  .7461  .7466           9 | 0.4763  0.00231  0.00086  0.00157  0.00248
11 | .7540  .7473  .7468  .7526  .7509          11 | 0.4899  0.00104  0.00030  0.00059  0.00108
13 | .7516  .7544  .7453  .7482  .7498          13 | 0.4954  0.00048  0.00009  0.00026  0.00050
```

## LER vs readout length — `futuristic`
```
ECCENTRIC (idle×1)                              ARTIFACT (idle×3)
 d |  200    400    600    800   1000            d |  200    400    600    800   1000
 5 | .0517  .0202  .0195  .0207  .0213           5 | 7e-5   2e-5   4e-5   7e-5   8e-5
 7 | .1427  .0721  .0694  .0729  .0748           7 | 1e-5   2e-6   0      4e-6   1e-5
 9 | .4860  .3321  .3220  .3314  .3351           9 | 0      0      0      0      2e-6
11 | .7255  .6828  .6769  .6816  .6852          11 | 0      0      0      0      0
13 | .7529  .7501  .7537  .7481  .7443          13 | 0      0      0      0      0
```

## Key takeaways
1. **Readout length now has an OPTIMUM (~600 ns), not "shorter is better".** With the realistic
   HERQULES model the curve is U-shaped: at 200 ns the flip error (0.221) is catastrophic and
   saturates the LER; 400→600 ns drops sharply as fidelity improves; beyond 600 ns LER creeps
   back up as decoherence during the longer window dominates. The 1000 ns flip error is actually
   the lowest (0.0328) but 600 ns still wins because of the decoherence penalty. Best length is
   **600 ns** in almost every (tool, distance, noise) cell.
2. **This flips the conclusion vs the fixed-error run (RESULTS.md).** There (flip error fixed,
   "magically clean readout") the curve was monotonic — shorter always better, optimum at the
   shortest length. The realistic model says you cannot go short.
3. **Magnitude is large where not saturated.** Artifact current d=11: 200 ns → 600 ns is
   0.49 → 0.0003, a ~1600× improvement; the 600 ns curve is sub-threshold and falls with
   distance (d5→d13: 6.7e-3 → 9.4e-5).
4. **Tools still diverge** as in RESULTS.md: artifact deeply sub-threshold and very
   readout-sensitive; ECCentric's tqec CNOT over threshold (≈0.75) for d≥7 (current) so its
   U-shape is only visible at low d / futuristic. But where both are unsaturated they agree the
   optimum is ~600 ns.

## How to read the two result sets together
- `RESULTS.md` (fixed 3.54e-3): "if we could magically have fast, accurate readout, how low can
  LER go?" → shorter readout is strictly better (e.g. artifact d=11 @250 ns ≈ 6e-6).
- `RESULTS_herqules.md` (this file): realistic readout where speed costs fidelity → optimum ≈600 ns
  (artifact d=11 best ≈ 3e-4). The gap between the two (~50×) is the prize for improving readout
  fidelity-at-speed.
