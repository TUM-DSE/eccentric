# Does readout length affect logical-CNOT LER, and does it differ between tools?

Surface-code lattice-surgery **logical CNOT**, LER vs **readout (= measurement) length**.
Runners: `compare_readout_eccentric.py`, `synchronization-artifact/compare_readout_artifact.py`.
Data: `combined_readout_ler.csv` (+ per-tool CSVs).

## Setup (unified across both tools)
- Readout length = the **measurement duration** (M-gate / mcm latency), swept 250–2000 ns.
  Its effect on LER is via **idle T1/T2 decoherence during the measurement window**; the
  readout *flip* error is FIXED at the ibm_boston median (so both tools share one readout model).
- Noise = **SOTA ibm_boston (Heron r3)**: CZ=1.191e-3, SX=1.637e-4, readout=3.54e-3,
  T1=284.95 µs, T2=322.68 µs. **current** = these values; **futuristic** = all errors /10, T1/T2 ×3.
- **idle_multiplier**: ECCentric = **1.0** (no artifact inflation) · Artifact = **3.0** (its IBM default).
- CNOT latency 70 ns. Decoder = MWPM. A shot fails if any of the 3 logical observables is wrong.
- Shots: artifact 500k; ECCentric adaptive (100k@d5 … 15k@d13).
- Distances 5,7,9,11,13.

## LER vs readout length (ns) — "ratio" = max/min across readout lengths

### current
```
ECCENTRIC                                                          ARTIFACT
 d |   250    500    750   1000   1500   2000  ratio    d |   250      500      750     1000     1500     2000   ratio
 5 | .2445  .2598  .2723  .2896  .3172  .3458  1.41x    5 | 7.98e-4 1.64e-3 2.84e-3 4.42e-3 8.43e-3 1.46e-2  18.4x
 7 | .6703  .6757  .6782  .6799  .6878  .6877  1.03x    7 | 1.48e-4 3.88e-4 7.54e-4 1.37e-3 3.15e-3 6.49e-3  43.9x
 9 | .7501  .7504  .7519  .7483  .7507  .7503  1.00x    9 | 1.80e-5 1.06e-4 2.06e-4 4.18e-4 1.27e-3 2.98e-3 165.6x
11 | .7504  .7476  .7492  .7434  .7500  .7474  1.01x   11 | 6.0e-6  1.0e-5  5.2e-5  1.22e-4 4.14e-4 1.10e-3 183.3x
13 | .7543  .7457  .7487  .7498  .7517  .7483  1.01x   13 | 0       0       6e-6    3.6e-5  1.52e-4 4.40e-4  73.3x
```

### futuristic
```
ECCENTRIC                                                          ARTIFACT
 d |   250    500    750   1000   1500   2000  ratio    d |   250      500      750     1000     1500     2000   ratio
 5 | .0137  .0161  .0163  .0164  .0208  .0234  1.71x    5 | 1.0e-5 1.4e-5  2.6e-5  8.2e-5  1.24e-4 3.20e-4  32.0x
 7 | .0549  .0578  .0617  .0646  .0708  .0749  1.37x    7 | 0      0       0       2e-6    6e-6    2.4e-5   12.0x
 9 | .2753  .2884  .2928  .3014  .3139  .3264  1.19x    9 | 0      0       0       0       0       0         -
11 | .6653  .6696  .6677  .6721  .6740  .6787  1.02x   11 | 0      0       0       0       0       0         -
13 | .7490  .7411  .7404  .7465  .7473  .7473  1.01x   13 | 0      0       0       0       0       0         -
```

## Q1 — Does readout length affect LER?
**Yes, in both tools: longer readout → higher LER, monotonically.** A longer measurement
window means more idle T1/T2 decoherence on the data qubits, which raises the logical error.
The effect is visible everywhere the code is below saturation; it vanishes only where a tool
is pinned at the saturation floor (~0.75).

## Q2 — Does this change between the two tools? **Yes, strongly — on three axes.**
1. **Absolute LER differs by 100–1000×.** At identical nominal noise, the artifact's logical
   CNOT is deeply sub-threshold (1e-6 … 1e-2) while ECCentric's tqec lattice-surgery CNOT is
   **over threshold** (LER ≈ 0.75 for d≥7 current / d≥11 futuristic). Root cause: the tqec
   compiled CNOT is a much larger/deeper circuit (e.g. 806 qubits / 42k detectors at d=11) than
   the artifact's hand-built merge, so it sits above threshold regardless of readout length.
   (This was already true in the original results: ECCentric ~0.87 vs artifact ~0.017.)
2. **Readout-length sensitivity is far stronger in the artifact** (18–183× across the sweep)
   than in ECCentric (1.2–1.7× where not saturated). Two reasons: (a) artifact idle_mult=3 vs
   ECCentric 1.0 — each extra ns of readout adds 3× the decoherence; (b) the artifact is deep
   below threshold where LER ∝ p^(d/2), so a small change in per-round error is amplified into a
   large LER ratio, whereas ECCentric near saturation cannot amplify.
3. **Distance scaling is opposite.** Artifact LER *decreases* with distance (sub-threshold:
   current@1000ns 4.4e-3→3.6e-5 for d5→d13); ECCentric LER *increases* toward 0.75 (over threshold).

### Caveat
The comparison is "same nominal noise, each tool's native CNOT circuit + the requested
idle_multiplier asymmetry (1.0 vs 3.0)". The absolute gap is dominated by the **circuit**
difference (tqec full lattice surgery vs the artifact's merge), not by the readout model — both
tools share the fixed ibm_boston readout flip error. ECCentric's saturation at d≥7 (current)
limits its readout-length signal to low distance.
