# Changelog

All notable changes to ECCentric are listed here. Versions follow [semantic versioning](https://semver.org);
releases are tagged on GitHub.

## [Unreleased]

### Added

- Plugins: codes, decoders, devices, noise models and command-line commands from other packages, registered
  through entry points (`eccentric.plugins`, docs/plugins.md). `eccentric plugins` lists them.
- `eccentric.dataset`: the sweep table's schema (columns, statuses, schema version 1) and a loader that checks it
  (docs/dataset.md). `stats.json` records the schema version.
- `tools/` for tools built on ECCentric; an index of the analyses (analysis/README.md).
- Figures (`python -m eccentric.figures`): featured findings of the dataset and the paper's topology,
  technology, distributed, routing, translation and accumulated-error experiments redrawn from the current
  results, each with its data, in `site/figures/` for the website.
- `eccentric.dataset.family()`: the code families, used by the website export and the figures.

## [1.1.0] – 2026-11

### Added

- **Codes:** about 60 new codes: bivariate, trivariate, coprime and generalized bicycle, balanced product,
  hypergraph and lifted product, lift-connected surface, quantum Tanner, many-hypercube, Mitten, Golay and
  Reed-Muller codes; honeycomb, hyperbolic and semi-hyperbolic Floquet codes; XZZX, toric and repetition codes.
  Several syndrome-extraction schedules per code (`schedules`).
- **Operations:** X-basis memory (`bases`); lattice surgery on surface-code patches with tqec (`ls_memory`,
  `ls_merge`, `ls_cnot`); fault-tolerant measurement, joint measurement and transversal CNOT on CSS codes
  (`operations`).
- **Decoders:** BP-LSD, BeliefFind, union-find, belief matching, Relay-BP, Tesseract, MWPF, GARI, GARI++ (beta),
  correlated matching, level-by-level BP-OSD for concatenated codes, BP-OSD with tuned settings, and GPU
  versions of the BP-based decoders.
- **Devices:** IBM's bicycle-architecture module, single-QPU versions of Flamingo and Loon, Harvard/MIT neutral
  atoms, and code-native layouts.
- **Noise models:** erasure and atom loss, Willow from its spec sheet, 700 ns IBM readout, and a scaling knob for
  any hardware model (`<model>_x<factor>`).
- **Dataset:** the sweep over codes, devices, noise models and decoders (`experiment_results/Sweep/`), the
  tooling that produces it (`sweep/`, `validation/`), and analyses of it (`analysis/`).
- Result rows carry more columns (circuit size, schedule, basis, rounds, per-round logical error rate).
- Every configuration compiles to the same circuit in every run (`eccentric/seeding.py`).

### Changed

- More accurate noise modelling: decoherence of qubits during parallel gates and while waiting, idle errors,
  and readout durations of the hardware models.
- More accurate circuit construction: the concatenated Steane code's logical readout, placement of codes on
  sparse devices, single-QPU devices, and detector coordinates through compilation.
- Willow noise model: idle error once per cycle; the crosstalk term, which had no effect, was removed.
- Accumulated-error experiment: the idle reference circuit now idles.
- MWPM decodes the decomposed error model.
- Results with new columns are merged into existing `results.csv` files by column name.
- BP-OSD decoders run on ldpc v2. Updated dependencies (stim, sinter, PyMatching), and `stimbposd` and
  `mqt.qecc` removed.

Details of each change: [docs/DETAILED_CHANGES.md](docs/DETAILED_CHANGES.md).

## [1.0.0] – 2026

The version evaluated in the SIGMETRICS'26 paper, packaged as `eccentric` with a single experiment runner
(tag `v1.0.0-sigmetrics26`, October 2026). The code as of the camera-ready version (March 2026), before the
packaging, is tagged `sigmetrics26-camera-ready`.
