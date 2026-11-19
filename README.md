# ECCentric

[![Paper](https://img.shields.io/badge/SIGMETRICS'26-10.1145%2F3805635-blue)](https://doi.org/10.1145/3805635)
[![Tests](https://github.com/TUM-DSE/eccentric/actions/workflows/tests.yml/badge.svg)](https://github.com/TUM-DSE/eccentric/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Website](https://img.shields.io/badge/results-eccentric--benchmark.github.io-orange)](https://eccentric-benchmark.github.io)

**ECCentric is a benchmark for quantum error correction codes under realistic hardware conditions.** It
compiles a code's syndrome-extraction circuit onto a device (topology, connectivity, routing, gate
translation), applies a hardware noise model, and decodes it, so codes, devices, noise models and decoders
can be compared end to end. It comes with a large dataset of results across all of them.

<div align="center">

![codes][stat-codes] ![devices][stat-devices] ![noise models][stat-noise-models] ![decoders][stat-decoders]<br>
![results][stat-results] ![shots decoded][stat-shots-decoded] ![CPU hours][stat-cpu-hours]

</div>

> 📄 **ECCentric: An Empirical Analysis of Quantum Error Correction Codes**, ACM SIGMETRICS 2026.
> Aleksandra Świerkowska, Jannik Pflieger, Emmanouil Giortamis, Pramod Bhatotia.
> [doi:10.1145/3805635](https://dl.acm.org/doi/abs/10.1145/3805635)
>
> The version evaluated in the paper is tagged [`v1.0.0-sigmetrics26`](https://github.com/TUM-DSE/eccentric/tree/v1.0.0-sigmetrics26).
> `main` is the actively developed benchmark: it adds codes, decoders, devices and noise models, and refines
> the noise modelling and circuit construction, so its numbers can differ from the paper's.
> See [CHANGELOG.md](CHANGELOG.md).

## Results at a glance

The dataset ([`experiment_results/Sweep/results.csv`](experiment_results/Sweep/results.csv)) covers every
code on IBM, Google, Quantinuum, Infleqtion and neutral-atom device models, under their own noise and scaled
versions of it, with every applicable decoder, in the Z and X basis. All results can be browsed on the
**[website](https://eccentric-benchmark.github.io)**; its columns are documented in [docs/dataset.md](docs/dataset.md). Some findings, with the full analyses in [`analysis/`](analysis/README.md):

- **Layout decides whether qLDPC codes work on IBM devices.** Under the same noise (a tenth of today's
  Flamingo errors, coherence times tripled), bivariate bicycle codes on a native layout reach logical error
  rates of 10⁻³ to 10⁻⁶. Routed onto Flamingo or Loon, they stay at 10–100%: 40× to 200,000× worse.
  ([INSIGHTS](analysis/INSIGHTS.md))
- **The surface code is below threshold on most of today's devices.** It is below threshold at today's
  noise on Willow, Infleqtion, Harvard/MIT neutral atoms, IBM's bicycle module and Loon. On Heron it needs
  40% of today's noise. ([NOISE_SCALING](analysis/NOISE_SCALING.md))
- **Z-basis memory experiments are optimistic.** For 791 of 1,492 configurations below threshold, the X basis is
  significantly worse; on Infleqtion, qLDPC codes are 7.6× worse in X (median). ([BASIS](analysis/BASIS.md))
- **There is no single best decoder.** The most accurate decoder and the fastest one that is good enough differ by orders
  of magnitude in cost, and which decoder wins depends on the code, the device and the noise.
  ([DECODING](analysis/DECODING.md), [DECODER_SETTINGS](analysis/DECODER_SETTINGS.md), [NEW_DECODERS](analysis/NEW_DECODERS.md))
- **The device models match experiments.** With matching decoders, the Willow model reproduces Google's
  surface-code memory within 1.15 to 1.25× at every distance, and its error suppression factor Λ.
  ([WILLOW](analysis/WILLOW.md))

Built on the dataset: [QEC Matchmaker](https://github.com/manosgior/qec-matchmaker) predicts the logical
error rate and decoding time of a circuit and recommends a decoder, without decoding it.

## Features

- **Codes** ([full list](docs/codes.md)): surface, heavy-hex, color, Bacon-Shor, concatenated Steane;
  bivariate/trivariate/coprime/generalized bicycle, balanced product, hypergraph and lifted product,
  lift-connected surface, quantum Tanner, many-hypercube, Mitten, Golay, Reed-Muller; Floquet codes
  (honeycomb, hyperbolic, semi-hyperbolic); XZZX, toric, repetition. Several syndrome-extraction schedules each.
- **Operations:** memory in the Z and X basis, lattice surgery on surface-code patches (tqec), and
  fault-tolerant measurements, joint measurements and transversal CNOT on CSS codes.
- **Devices** ([full list](docs/devices.md)): IBM Heron, Flamingo, Loon/Nighthawk and the bicycle-architecture
  module, Google Willow, Quantinuum Apollo, Infleqtion, Harvard/MIT neutral atoms, multi-QPU devices with remote
  gates, synthetic topologies, and code-native layouts.
- **Noise models** ([full list](docs/noise-models.md)): artificial circuit-level models (SD6, SI1000, ...),
  hardware-inspired models with gate, readout, idle, leakage and atom-loss errors, erasure, and a knob that scales any
  hardware model.
- **Decoders** ([full list](docs/decoders.md)): MWPM and correlated matching, BP-OSD, BP-LSD, BeliefFind,
  union-find, belief matching, Relay-BP, Tesseract, MWPF, Chromobius, GARI, level-by-level BP-OSD for
  concatenated codes, and GPU versions of the BP-based decoders.
- **Compilation:** Qiskit layout and routing, gate-set translation with Qiskit, BQSKit or TKET.

## Installation (Linux)

Python 3.11 is recommended.

```bash
git clone --recurse-submodules https://github.com/TUM-DSE/eccentric.git
cd eccentric

python3.11 -m venv venv
source venv/bin/activate
pip install -e .

cd external/qiskit_qec
python setup.py build_ext --inplace
cd ../..
```

With CMake 4, build qiskit-qec with `CMAKE_POLICY_VERSION_MINIMUM=3.5 python setup.py build_ext --inplace`
(its CMake files predate CMake 4).

`pip install -e .` installs the dependencies from `requirements.txt` and adds an `eccentric` command,
equivalent to `python -m eccentric`. If you cloned without `--recurse-submodules`, fetch the submodules with
`git submodule update --init --recursive`.

Optional:

- **GARI decoders** (`gari`, `gari_pp`) compile their C decoder on first use and need a C compiler with OpenMP.
- **GPU decoders** (`*_gpu`) need an NVIDIA GPU and `pip install --no-deps -r requirements-gpu.txt`.

Run all commands from the repository root: results are written to `experiment_results/` and figures to
`figures/` relative to the current directory.

## Quick start

```bash
python -m eccentric configs/example.yaml
```

This runs the small example in [`configs/example.yaml`](configs/example.yaml) and writes to `experiment_results/Example/`:

- `results.csv`: one row per configuration, for example the `logical_error_rate` over all `rounds` of syndrome extraction and the `logical_error_rate_per_round` derived from it as (1 − (1 − 2P)^(1/rounds))/2
- `experiment.log`: completed, skipped and failed configurations, with error messages
- `metadata_<timestamp>.json`: a copy of the experiment configuration

Results are appended to `results.csv`, so delete the directory or rename the experiment before re-running it.

Command-line options:

| Option | Description |
| --- | --- |
| `python -m eccentric a.yaml b.yaml` | Run several configuration files in sequence |
| `-j N`, `--workers N` | Number of parallel worker processes (default: all CPUs). Lower it if you run out of memory |
| `-e NAME`, `--experiment NAME` | Only run the experiment called `NAME` (repeatable) |
| `eccentric plugins` | List the installed plugins ([docs/plugins.md](docs/plugins.md)) |
| `eccentric <command> ...` | Run a command added by a plugin or tool |

## Configuration

Experiments are described in a YAML file containing a list under `experiments`. Every list-valued field is a
sweep: the runner executes the Cartesian product of all lists, in parallel.

```yaml
experiments:
  - name: "Example"
    type: logical_error
    num_samples: 1000
    backends: ["custom_grid"]
    backends_sizes: [300]
    codes: ["surface", "color"]
    error_types: ["modsi1000"]
    error_probabilities: [0.002]
    decoders: ["bposd"]
```

Reference: [all configuration fields](docs/configuration.md), [codes](docs/codes.md),
[devices](docs/devices.md), [noise models](docs/noise-models.md), [decoders](docs/decoders.md),
[the dataset](docs/dataset.md), [plugins](docs/plugins.md).

## Reproducing the paper

To reproduce the paper exactly, check out the tag `v1.0.0-sigmetrics26`. Every experiment of the paper has
a configuration in [`configs/`](configs):

| Configuration | Paper experiment |
| --- | --- |
| `configs/size.yaml` | Code size |
| `configs/connectivity.yaml` | Connectivity and real topologies |
| `configs/variance.yaml` | Variance of decoherence and readout errors |
| `configs/technologies.yaml` | Hardware technologies |
| `configs/dqc.yaml` | Distributed quantum computing |
| `configs/routing.yaml` | Routing overhead |
| `configs/translation.yaml` | Gate-set translation overhead |
| `configs/accumulated.yaml` | Accumulated error and thresholds |
| `configs/decoder.yaml` | Decoders |
| `configs/chromobius.yaml` | Chromobius decoder (appendix) |
| `configs/program_stats.yaml` | Time and memory of the benchmark |

```bash
python -m eccentric configs/size.yaml   # one experiment
bash run_experiments.sh                 # all experiments, then all plots
python -m eccentric.plots               # only regenerate the figures from experiment_results/
```

## The sweep

The dataset is produced by `sweep/`, which generates one job per circuit or decode, runs them on a cluster
(`python -m sweep.run_job <run_dir> <job_id>`, each until 100 logical errors or its time budget) and collects
them (`python -m sweep.collect <run dirs> <out>`; `python -m sweep.stats` then updates the numbers above).
Each row records the commit that produced it. `validation/` holds the cluster checks of circuits, code and
circuit distances, and decoders.

## Running with Docker

1. Fetch the submodules and build the image:

    ```bash
    git submodule update --init --recursive
    docker build -t eccentric .
    ```

2. Run all experiments and plots (this executes `run_experiments.sh`):

    ```bash
    docker run --name eccentric_run eccentric
    ```

3. Copy the results and plots out of the container, then remove it:

    ```bash
    docker cp eccentric_run:/app/experiment_results ./experiment_results_docker
    docker cp eccentric_run:/app/figures ./figures_docker
    docker rm eccentric_run
    ```

To run a single experiment instead, override the command, for example `docker run --rm eccentric python -m eccentric configs/example.yaml`.

## Comparison with real IBM hardware

`eccentric.hardware.ibm_submit` submits random circuits (2–7 qubits, depth 100, 10 circuits per width, 1000 shots each) to an IBM Quantum backend. `eccentric.hardware.ibm_demo` then compares the measured fidelities with noiseless simulation and the noise models in this repository.

```bash
python -m eccentric.hardware.ibm_submit submit            # submit the job (asks for confirmation; uses IBM credits)
python -m eccentric.hardware.ibm_submit results <job_id>  # download and store the results
python -m eccentric.hardware.ibm_demo [<job_id>]          # analyse and plot (defaults to the job used in the paper)
```

Job metadata and results are stored in `experiment_results/ibm_runs/`. These scripts need an IBM Quantum API token. Follow the [IBM Quantum setup guide](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup) to save it in your environment.

## Extending ECCentric

- **As a plugin** (no changes to ECCentric): a package registers codes, decoders, devices, noise models or
  command-line tools through entry points, and they work in every configuration under their names. Built-in
  names always take precedence. See [docs/plugins.md](docs/plugins.md); `eccentric plugins` lists what is installed.
- **Tools on the dataset** read the sweep table through `eccentric.dataset`, a versioned schema
  ([docs/dataset.md](docs/dataset.md)). Tools maintained with ECCentric live in [`tools/`](tools).
- **In ECCentric itself:** add a code, device, noise model or decoder to its package and its dispatcher
  (`get_code` in `eccentric/codes/utils.py` or the catalog in `eccentric/codes/catalog.py`, `get_backend` in
  `eccentric/backends/utils.py`, `get_noise_model` in `eccentric/noise/utils.py`, `decode` in
  `eccentric/decoders/decode.py`). A new experiment type is a module in `eccentric/experiments/` that defines
  `REQUIRED` (the required config fields) and `run(task)`, which returns a list of result rows, registered in
  `EXPERIMENTS` in `eccentric/experiments/__init__.py`; `eccentric/experiments/base.py` provides the shared steps.

Run the tests with `pytest`.

## Repository layout

```
eccentric/            the framework (Python package): runner, experiments, codes, backends, noise,
                      decoders, transpilers, metrics, plots, hardware comparison
configs/              configurations of the paper's experiments, plus example.yaml
experiment_results/   results of the paper's experiments and the sweep dataset (Sweep/)
analysis/             analyses of the dataset (index: analysis/README.md)
tools/                tools built on ECCentric, each its own package
sweep/                the sweep: job generation, cluster runner, collection, website export
validation/           cluster checks of circuits, distances and decoders
docs/                 reference documentation
tests/                unit and end-to-end tests (pytest)
external/             submodules: qiskit-qec fork, more-bacon-less-threshold, honeycomb_threshold, honeycomb-boundaries
```

## Citation

```bibtex
@inproceedings{swierkowska2026eccentric,
  title     = {ECCentric: An Empirical Analysis of Quantum Error Correction Codes},
  author    = {{\'S}wierkowska, Aleksandra and Pflieger, Jannik and Giortamis, Emmanouil and Bhatotia, Pramod},
  booktitle = {Proceedings of the ACM SIGMETRICS Conference (SIGMETRICS'26)},
  year      = {2026},
  doi       = {10.1145/3805635}
}
```

## License

ECCentric is released under the [Apache License 2.0](LICENSE). Vendored and submodule code keeps its own
license (e.g. the GARI decoder in `eccentric/decoders/gari/`).

<!-- Badges: numbers read from experiment_results/Sweep/stats.json on main (python -m sweep.stats) -->
[stat-codes]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FTUM-DSE%2Feccentric%2Fmain%2Fexperiment_results%2FSweep%2Fstats.json&query=%24.codes&label=codes&style=for-the-badge&color=2b6cb0
[stat-devices]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FTUM-DSE%2Feccentric%2Fmain%2Fexperiment_results%2FSweep%2Fstats.json&query=%24.devices&label=devices&style=for-the-badge&color=2b6cb0
[stat-noise-models]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FTUM-DSE%2Feccentric%2Fmain%2Fexperiment_results%2FSweep%2Fstats.json&query=%24.noise_models&label=noise%20models&style=for-the-badge&color=2b6cb0
[stat-decoders]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FTUM-DSE%2Feccentric%2Fmain%2Fexperiment_results%2FSweep%2Fstats.json&query=%24.decoders&label=decoders&style=for-the-badge&color=2b6cb0
[stat-results]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FTUM-DSE%2Feccentric%2Fmain%2Fexperiment_results%2FSweep%2Fstats.json&query=%24.display.results&label=results&style=for-the-badge&color=2f855a
[stat-shots-decoded]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FTUM-DSE%2Feccentric%2Fmain%2Fexperiment_results%2FSweep%2Fstats.json&query=%24.display.shots&label=shots%20decoded&style=for-the-badge&color=2f855a
[stat-cpu-hours]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FTUM-DSE%2Feccentric%2Fmain%2Fexperiment_results%2FSweep%2Fstats.json&query=%24.display.cpu_hours&label=CPU%20hours&style=for-the-badge&color=2f855a
