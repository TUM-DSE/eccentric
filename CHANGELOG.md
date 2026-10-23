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

## Changed

### 5. BP-OSD decoders run on ldpc v2

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

### 6. Dependencies

- `ldpc` 0.1.60 → 2.4.1, `stim` 1.14.0 → 1.16.0, `sinter` 1.14.0 → 1.16.0, `PyMatching`
  2.2.1 → 2.3.1.
- Removed `stimbposd` (replaced, see 5) and `mqt.qecc` (not imported anywhere; it requires
  `ldpc` < 2).
- Added for the new features: `qldpc`, `quits`, `relay-bp`, `tesseract-decoder`, `beliefmatching`.

### 7. More columns in result rows

- Rows with a logical error rate (`logical_error`, `decoder`, `program_stats`, `chromobius`) now
  also contain `schedule`, `rounds` (the syndrome rounds the circuit runs) and
  `logical_error_rate_per_round`; `accumulated_error`, `routing` and `translation` rows contain
  `schedule`. Existing columns are unchanged.

### 8. Minor

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

## Validation

`validation/` runs the checks behind these entries on a cluster (circuit validity, code and circuit
distance, cross-decoder comparison including old vs. new BP-OSD); `tests/test_noise_invariants.py`
pins down the noise model's behaviour.
