# Changelog: fixes to existing functionality (branch `new-codes`)

This file lists changes to functionality that existed before the `new-codes` branch: bug fixes,
changed behaviour and changed dependencies. The new codes, decoders, noise models, platforms and
features of the branch are documented in the README. Each entry says what was wrong, what changed,
which experiments and paper figures it affects, and how it was checked.

"Paper" refers to the ECCentric camera-ready version (SIGMETRICS'26). Figure numbers follow it.

## Fixed

### 1. Two-qubit gate decoherence was applied to every qubit of a gate layer

- **File:** `eccentric/noise/noise.py`, `NoiseModel.noisy_op`. **Commit:** `9390dc7`.
- **Problem:** Stim merges parallel two-qubit gates into one instruction (e.g. `CX` on 72 pairs).
  For each pair, the T1/T2 decoherence of that gate's duration was added to *all* qubits of the
  instruction instead of the two qubits of the pair, so every qubit of a layer of *n* parallel
  gates decohered for *n* gate durations.
- **Fix:** the decoherence of a pair is applied to that pair only.
- **Affects:** noise models that derive decoherence from T1/T2 and gate times (no fixed `idle`
  error): `real_infleqtion`, `real_flamingo`, `real_flamingo_downscaled`, `real_heron`, `variance`
  and the `variance_*` readout models (and the new `real_harvard`). Not affected: `real_willow`,
  `real_apollo` and all artificial models (`sd6`, `si1000`, ...). Configurations with many parallel
  gates (all-to-all or shuttling devices) are affected most.
- **Paper:** Fig. 15 (variance), Fig. 17 (distributed QPUs) and the Infleqtion bars of Fig. 19.
  Re-runs with BP-OSD and the paper's settings (1000 shots):
  - Fig. 15: almost all bars at about 0.5 were caused by this bug. With the fix, at 1× T1/T2,
    surface 0.49 → 0.000, heavy-hex 0.50 → ≤ 0.011, color 0.50 → ≤ 0.012, gross 1.00 → 0.000.
    Codes stay nearly flat across variance levels, so Takeaway #3 still holds, but for a different
    reason (the codes now work).
  - Fig. 17 (Nighthawk): most error rates drop; single-QPU execution beats the distributed one for
    every code under normal noise.
  - Fig. 19 (Infleqtion): no significant change for surface, heavy-hex and Bacon-Shor.
- **Checked by:** `tests/test_noise_invariants.py::test_parallel_gates_decohere_each_qubit_once`;
  the affected experiments were re-run with and without the fix on identical code.

### 2. Concatenated Steane circuits did not measure the logical qubit

- **File:** `eccentric/codes/concat_steane.py` (`normal_steane_circuit_f`, `concat_steane_circuit_f`,
  `extended_steane_circuit`), `eccentric/codes/utils.py`. **Not committed yet.**
- **Problem:** the circuits ended by CNOTing a readout ancilla to all data qubits and took that
  ancilla's measurement as the logical result. The ancilla was the control of the CNOTs and
  started in |0>, so the CNOTs did nothing and the "logical" result did not depend on the data:
  a logical error on the data neither flipped the result nor fired a detector. Every Steane
  logical error rate therefore measured noise on the readout ancilla, not the code. The circuit
  version used before August 2025 (see Known issue 4) had the same flaw.
- **Fix:** standard destructive readout. All data qubits are measured, a final detector compares
  each Z stabilizer recomputed from the data with its last syndrome measurement, and the logical
  observable is the parity of all data qubits (the transversal logical Z). The readout ancilla is
  gone, so the circuits use 13 / 97 / 685 qubits for d = 3 / 9 / 27 instead of 14 / 98 / 686
  (`get_min_n` and `get_max_d` updated).
- **Also fixed:** the d = 27 circuit (`extended_steane_circuit`) ran one more syndrome round than
  requested; it now runs `cycles` rounds like d = 3 and 9.
- **Changed:** `get_concat_steane_code(basis="X")` now raises `NotImplementedError`. The circuits
  always prepare the data in |0>, so the X basis never was a valid memory experiment. ECCentric
  only uses the Z basis.
- **Affects:** every result of the `steane` code, in every experiment.
- **Paper:** every figure and table that includes the Steane code, and statements that rely on
  them, e.g. §4.5 "the concatenated Steane code performs best on average" and the
  Steane-specific statements of §4.4. Steane results are being regenerated with the fix.
- **Checked by:** injecting a logical error (`X_ERROR(1)` on all data qubits) now flips the result
  without firing detectors, a single data error fires detectors, no single fault flips the
  result undetected, and the circuit distance is 3 for d = 3 and 9 for d = 9.

### 3. Results with new columns were appended to existing CSV files misaligned

- **File:** `eccentric/utils/saving.py`, `save_results_to_csv`. **Commit:** `0dfecab`.
- **Problem:** the header of `results.csv` was written once. Rows with additional or reordered
  columns were appended by position, so their values ended up under the wrong headers.
- **Fix:** values are written by column name; when a row has new columns, the file is rewritten
  with the extended header (missing values left empty).
- **Checked by:** `tests/test_features.py::test_new_columns_extend_existing_results`.

### 4. Detector coordinates were lost on the way through Qiskit

- **Files:** `eccentric/experiments/base.py` (`with_detector_coordinates`),
  `eccentric/codes/color_code_stim.py`. **Commits:** `0dfecab`, `c7eec62`.
- **Problem:** the Stim → Qiskit → Stim conversion dropped detector coordinates, so decoders that
  need them (Chromobius) could not be used in the normal pipeline. The paper's `color` code also
  had no color/basis annotation on its detectors.
- **Fix:** coordinates are copied back from the code's Stim circuit after the conversion, and the
  `color` code's detectors carry a 4th coordinate (0–2: X checks, 3–5: Z checks, per color).
- **Affects:** only decoders that read coordinates; all other results are unchanged. `chromobius`
  now works on the `color` code in `logical_error` experiments.
- **Checked by:** `tests/test_features.py::test_detector_coordinates_survive_the_pipeline`.

### 5. MWPM decoding graph built from an undecomposed error model

- **File:** `eccentric/decoders/mwpm.py`.
- **Problem:** the detector error model was generated without `decompose_errors=True`. PyMatching
  then builds its matching graph without the error mechanisms that flip more than two detectors
  (for example Y errors and correlated two-qubit errors), so most edge probabilities were too low
  (surface code d = 5: 466 of 502 edges, mean probability 2.0e-3 instead of 3.3e-3).
- **Fix:** the error model is decomposed into graph edges, the standard way to use PyMatching. A
  circuit whose errors cannot be decomposed now makes MWPM fail instead of decoding with a wrong graph.
- **Affects:** all `mwpm` results. On the circuits checked (surface code d = 5, heavy-hex d = 3 at
  p = 0.001) the logical error rates changed only within the statistical error, so this does not
  explain the gap between MWPM and BP-OSD reported in the paper.

### 6. "1 QPU" devices were compiled onto the full multi-QPU device

- **File:** `eccentric/backends/utils.py` (with `fake_loon.py` and `fake_flamingo.py`, which now take
  the number of QPUs).
- **Problem:** `real_loon_1_qpu` and `real_flamingo_1_qpu` loaded the full two- and three-QPU
  devices. The single-QPU size (120 and 133 qubits) only chose the code distance, so the compiler
  was free to place and route the code across all QPUs, over the slow and noisy inter-QPU links. In
  the sweep, 84 of 102 "1 QPU" circuits used more qubits than one QPU has. The paper-era `main.py`
  did the same.
- **Fix:** the "1 QPU" devices are built with a single QPU. The distance choice is unchanged.
- **Affects:** every `real_loon_1_qpu` and `real_flamingo_1_qpu` result, including the
  single-QPU baselines of the distributed-QPU comparison in the paper. In the sweep the effect on
  the logical error rate is small: of 467 decodes on the same configurations (qLDPC codes, IBM
  noise and tenfold lower), 441 agree within 3 standard errors, 14 are lower and 12 higher after
  the fix (median ratio 1.00), so the circuits that spilled over used few inter-QPU links.
- **Checked by:** `tests/test_platforms.py::test_single_and_projected_ibm_qpus`; after the fix none
  of the 102 circuits uses more qubits than one QPU. The two- and three-QPU devices are unchanged
  (identical couplers and inter-QPU links).

### 7. Willow idle error applied in every layer instead of once per cycle

- **Files:** `eccentric/noise/noise.py` (new `idle_window` option), `eccentric/noise/willow_noise.py`.
- **Problem:** the Willow models (`real_willow`, `real_willow_spec`) take their idle error, 0.9%,
  from Google's "data qubit idle error during measurement and reset" (arXiv:2408.13687), an error
  per cycle on the data qubits. `NoiseModel` applied it as a depolarizing error to every waiting
  qubit in every layer, about seven times per surface-code cycle. Willow was therefore about eight
  times too noisy: the surface code at d = 7 placed on the grid gave 1.1% logical error per cycle,
  against 0.143% in Google's experiment, and an error suppression factor Λ of 1.6 instead of 2.14.
  Switching off each error source in turn showed the idle error to account for almost all of it.
- **Fix:** `NoiseModel(idle_window="measurement")` applies the idle error only in layers with
  measurements, to the qubits that wait meanwhile; the Willow models use it. All other models keep
  the idle error per layer (`idle_window="layer"`, the default), which is how SD6, SI1000 and the
  other synthetic models define it.
- **Affects:** every Willow result, including the paper's (`experiment_results/Willow`, the Willow
  rows of `Real_Topology`).
- **Checked by:** `tests/test_noise_invariants.py::test_idle_window`, and against Google's
  experiment: the surface code memory placed on the grid, 10 cycles, MWPM, now gives 0.44%, 0.20%
  and 0.076% per cycle at d = 3, 5, 7 (Λ = 2.4) with the paper's Willow model and 0.50%, 0.23%,
  0.097% (Λ = 2.3) with the spec-sheet model, against about 0.65%, 0.31% and 0.143% (Λ = 2.14)
  measured. The remaining difference (the model is slightly optimistic) fits what the model leaves
  out, such as leakage accumulation and stray couplings.

### 8. Codes were placed qubit by qubit onto sparse devices

- **File:** `eccentric/transpilers/transpiler.py`.
- **Problem:** `run_transpiler` compiles with Qiskit's `optimization_level=0` (so that no gate
  optimization changes the syndrome circuit). Without an explicit layout method, level 0 places
  circuit qubit i on device qubit i (`TrivialLayout`) and leaves the rest to routing; it never looks
  for a placement, not even one needing no SWAPs, which Qiskit's levels 1 to 3 find with `VF2Layout`.
  The surface code at d = 7 on Willow fits the grid exactly but was routed with about 740 SWAPs per
  two cycles; where no exact fit exists, a chosen placement still needs far fewer SWAPs: 198 instead
  of 582 for the surface code at d = 5 on Torino, 2162 instead of 3311 for bb72 on Loon, 12346
  instead of 19364 for the gross code on Flamingo (two cycles). On the Willow surface code memory,
  the routed d = 7 code gave 1.1% logical error per cycle (MWPM), against 0.077% placed on the grid.
- **Fix:** without a layout method, `run_transpiler` first looks for a placement under which every
  two-qubit gate acts on coupled qubits (with rustworkx's VF2, Qiskit's level-2 search budget),
  using only local couplers if it can and otherwise the fewest remote ones. Without one, SabreLayout
  chooses the placement; when the circuit fits on one QPU, it is placed and routed on that QPU's
  local couplers, so that codes do not spread over the noisy links between QPUs without need. Compilation stays at level 0 and routing is
  unchanged; an explicit layout method (`trivial`, `dense`, `sabre`) is used as given, and
  `layout_method="trivial"` reproduces the old behaviour.
- **Affects:** every result on a sparse device compiled without a layout method: in the paper
  `Real_Topology`, `Willow`, `DQC_Flamingo`, `DQC_Nighthawk`, `Connectivity` and
  `Program_stats_topology`, and on Apollo and Infleqtion (no SWAPs there, but fewer shuttles); in the
  sweep all IBM-device results. Not affected: all-to-all devices (`custom_full`, `variance_*`), the
  routing experiments (`Routing_grid`, `Routing_hh`, which set the layout method) and the `native`
  devices. Exact fits are rare: of 2049 earlier results on sparse devices, 40 had one (18 of them in
  the routing experiments); the change comes mostly from choosing a placement at all.
- **Checked by:** `tests/test_platforms.py::test_layout_of_codes_that_fit_the_device`.

### 9. Qubits did not decohere while they waited, and IBM readout took 70 ns

- **Files:** `eccentric/noise/noise.py`, `eccentric/noise/flamingo_noise.py`,
  `eccentric/noise/flamingo_noise_downscaled.py`, `eccentric/noise/utils.py`.
- **Problem:** T1/T2 decoherence was added only to the qubits an operation acts on, for that
  operation's duration. A qubit that waited (a data qubit while the ancillas are measured, or any
  qubit beside a slower operation) got no noise, so the noise models without a fixed `idle` error
  had no idle noise at all. On top, the IBM Flamingo/Loon models (also used for the bicycle module)
  gave measurement the duration of a two-qubit gate, 70 ns; IBM Heron's calibration data gives
  1.56 us, which our Heron model already used. The tenfold-lower model divided the error
  probabilities by 10 but kept T1/T2, unlike the "futuristic" IBM model of our other analyses
  (errors / 10, T1/T2 x 3).
- **Fix:** in every layer, each qubit the circuit uses decoheres for the time it waits for the
  slowest operation of the layer (measurement and reset layers included), with its own T1/T2. The
  IBM models measure for 1.56 us; `real_flamingo_m700` and `real_flamingo_downscaled_m700` are
  variants with 700 ns readout. `real_flamingo_downscaled` multiplies T1/T2 by 3
  (`NoiseModel(coherence_scale=3)`).
- **Effect:** with the median T1/T2 of the IBM models (about 175 and 143 us), a data qubit waiting
  out one readout gets about 0.8% error, four times the two-qubit gate error. Surface code d = 5 on
  one Loon QPU (MWPM, 5 rounds): 0.42% -> 1.5% logical error at IBM noise (0.91% with 700 ns
  readout); 0.021% -> 0.020% at tenfold lower noise with T1/T2 x 3.
- **Affects:** every noise model that derives decoherence from T1/T2 and gate times (no fixed
  `idle`): `real_flamingo`, `real_flamingo_downscaled`, `real_heron`, `real_infleqtion`,
  `real_harvard`, `variance` and `variance_*`, i.e. all IBM results of the sweep, the paper's
  distributed-QPU (Fig. 17), variance (Fig. 15) and Infleqtion results, and the neutral-atom slice.
  Not affected: `real_willow` (fixed idle error), `real_apollo` and the artificial models.
- **Checked by:** `tests/test_noise_invariants.py::test_waiting_qubits_decay`.

## Changed

### 10. BP-OSD decoders run on ldpc v2

- **File:** `eccentric/decoders/bp_osd.py`; `requirements.txt`. **Commit:** `d55f942`.
- **Why:** the new code libraries (qLDPC, QUITS) need `ldpc` ≥ 2, which breaks `stimbposd`
  (it imports `ldpc.osd`, removed in v2) and the v1 call signature used by `bposd_chk`.
- **Change:** `bposd_chk` and `bposd_batch` use `ldpc.BpOsdDecoder` with the same settings as
  before (min-sum BP, 10,000 iterations, scaling factor 1, OSD-CS of order 7). `bposd_batch`
  reproduces `stimbposd.BPOSD` (error mechanisms merged with XOR-combined priors, OSD order capped
  by the matrix rank). `stimbposd` is no longer used.
- **Checked by:** the same shots decoded with the old (ldpc 0.1.60 + stimbposd) and the new
  decoders, for the validation circuits of all codes at two error rates: 214 paired comparisons, mean |z| = 0.13. The only two
  differences beyond 3σ (`lcs15`) favour the new version.

### 11. Dependencies

- `ldpc` 0.1.60 → 2.4.1, `stim` 1.14.0 → 1.16.0, `sinter` 1.14.0 → 1.16.0, `PyMatching`
  2.2.1 → 2.3.1.
- Removed `stimbposd` (replaced, see 5) and `mqt.qecc` (not imported anywhere; it requires
  `ldpc` < 2).
- Added for the new features: `qldpc`, `quits`, `relay-bp`, `tesseract-decoder`, `beliefmatching`.

### 12. More columns in result rows

- Rows with a logical error rate (`logical_error`, `decoder`, `program_stats`, `chromobius`) now
  also contain `schedule`, `rounds` (the syndrome rounds the circuit runs) and
  `logical_error_rate_per_round`; `accumulated_error`, `routing` and `translation` rows contain
  `schedule`. Existing columns are unchanged.

### 13. Minor

- `create_bivariate_bicycle_codes` accepts the known code distance, so it is not recomputed by
  brute force (same circuits).
- `pytest` skips tests marked `slow` by default (`pytest -m slow` runs them).

## Known issues (found, not fixed)

1. **Fig. 17 c/d legend is swapped.** The Nighthawk results are stored as `real_loon` /
   `real_loon_1_qpu`, which `backend_rename_map` in `eccentric/plots/utils.py` does not rename,
   so the bars sort in the opposite order to the Flamingo panels and the shared legend labels
   them the wrong way round. The data says single-QPU execution is better for every code except
   Steane under normal noise and for every code under scaled noise; §4.4 states the opposite.
2. **The `chromobius` experiment (Fig. 28) has an error floor under SI1000-type noise.** Its color
   code circuit prepares and reads out the logical qubit with one multi-qubit measurement of the
   logical operator. ECCentric's noise model makes these measurements noisy, and a single fault
   on them flips the result undetected: probability 0.013 / 0.029 / 0.051 at d = 3 / 5 / 7 for
   p = 0.001. The phenomenological comparison (Fig. 29) uses noiseless logical measurements and
   is not affected. The CNOT-based `color` code (now decodable with Chromobius, see 4) is a sound
   replacement.
3. **Three entries of `BBCODE_DICT` are inconsistent** (`eccentric/codes/gross_code.py`):
   (126, 8, 10) and (150, 16, 8) give k = 0 with the convention of the gross code, and
   (360, 12, 24) builds a 240-qubit code. They are not used.
4. **The paper's Steane results come from two different circuits.** The Infleqtion and Apollo
   Steane results of Fig. 19 (run in July 2025) and the routing and translation results predate
   an August 2025 change of the Steane circuits; all results from December 2025 use the later
   circuits. Both versions have the readout flaw fixed in 2. With today's pipeline and Infleqtion
   noise, the two versions have undetectable-error floors of 0.004 and 0.109, which is why the
   paper reports 0.002 for Steane on Infleqtion while the later circuit gives about 0.1.
5. **Paper text vs data:** the introduction says distributing codes across QPUs increases the
   logical error rate by 69.14%, Takeaway #4 says single-QPU execution is 34.21% better; and the
   9.0×10⁻³ quoted in §4.5 for Flamingo at 10% noise does not appear in the stored results.
6. **TKET translation test crashes** (`tests/test_transpilers.py::test_tket_translation_returns_circuit`)
   with a segmentation fault depending on test order, also in the environment of `main`.
7. **Circuit-level distance is often below the code distance**, including for the paper's `color`
   code (≤ 2 at d = 3, ≤ 3 at d = 5, from hook errors with one ancilla per check). This is a
   property of the syndrome-extraction circuits, not a bug; `validation/` reports it per code.
8. **Some codes cannot be placed without SWAPs on the devices they target.** The heavy-hex code
   (`hh`) does not fit the heavy-hex lattice: its circuit's coupling graph has cycles of 10 and 16
   qubits, the lattice only cycles of 12 and more, so even on Heron it is routed (46 SWAPs at d = 3
   over two cycles). The Loon model gives long-range c-couplers only to the first row of each QPU,
   where IBM's layout gives every qubit two (arXiv:2308.07915), so bivariate bicycle codes on Loon
   are routed; the `native` devices give the placed comparison, with the long-range couplers as good
   as local ones (`native`) or as noisy as remote ones (`native_lr`).
9. **MWPM on heavy-hex gets worse with distance** at SD6 noise (p = 5×10⁻⁴: 0.21%, 0.62%, 0.69% at
   d = 3, 5, 7 over d rounds) while BP-OSD improves; decomposing the error model (entry 5) did not
   change this. Not yet explained.

## Validation

`validation/` runs the checks behind these entries on a cluster (circuit validity, code and circuit
distance, cross-decoder comparison including old vs. new BP-OSD); `tests/test_noise_invariants.py`
pins down the noise model's behaviour.
