# ECCentric Bench

ECCentric is a benchmarking framework for evaluating Quantum Error Correction (QEC) codes under realistic hardware conditions: device topologies, connectivity, noise models, routing, gate translation, and decoders.

> 📄 **Paper:** <ins>**ECCentric: An Empirical Analysis of Quantum Error Correction Codes**</ins>, accepted at <ins>**ACM SIGMETRICS 2026 (SIGMETRICS'26)**</ins>.
> Aleksandra Świerkowska, Jannik Pflieger, Emmanouil Giortamis, Pramod Bhatotia.
> [arXiv:2511.01062 (PDF)](https://arxiv.org/pdf/2511.01062)

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | Main benchmark runner: computes logical error rates for every configuration in a YAML file |
| `get_*.py` | Runners for the specialised experiments in the paper (DQC, decoders, routing, translation, program stats, accumulated error) |
| `experiments_conf/` | YAML configurations used for the paper |
| `backends/` | Fake hardware backends (IBM Flamingo/Loon, Google Willow, Quantinuum Apollo, Infleqtion) and custom topologies |
| `codes/` | QEC code circuit generators (surface, heavy-hex, gross, color, Bacon-Shor, concatenated Steane) |
| `noise/` | Artificial and hardware-inspired noise models |
| `decoders/` | MWPM, BP-OSD and Chromobius decoders |
| `transpilers/` | Routing, layout and gate-set translation (Qiskit, BQSKit, TKET) |
| `plots/` | Plot generation for the paper figures (written to `data/`) |
| `experiment_results/` | Raw results used for the paper |
| `ibm_submit.py`, `ibm_demo.py` | Comparison of the noise models against real IBM hardware |

## Installation (Linux)

Python 3.11 is recommended.

```bash
git clone --recurse-submodules https://github.com/TUM-DSE/eccentric.git
cd eccentric

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd external/qiskit_qec
python setup.py build_ext --inplace
cd ../..
```

If you cloned without `--recurse-submodules`, fetch the submodules first:

```bash
git submodule update --init --recursive
```

All commands below must be run from the repository root.

## Quick start

```bash
python3 main.py experiments.yaml
```

This runs the small example in [`experiments.yaml`](experiments.yaml). Results are written to `experiment_results/Example/`:

- `results.csv`: one row per configuration, including the `logical_error_rate`
- `experiment.log`: progress and errors
- `metadata_<timestamp>.json`: a copy of the experiment configuration

Results are appended to `results.csv`, so delete the directory or rename the experiment before re-running it.

## Configuration

Experiments are described in a YAML file containing a list under `experiments`. Every list-valued field is a sweep: the runner executes the Cartesian product of all lists, in parallel.

```yaml
experiments:
  - name: "Example"
    num_samples: 1000
    backends: ["custom_grid"]
    backends_sizes: [300]
    codes: ["surface", "color"]
    error_types: ["modsi1000"]
    error_probabilities: [0.002]
    decoders: ["bposd"]
```

### Fields

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Experiment name; results go to `experiment_results/<name>/` |
| `num_samples` | yes | Number of shots sampled for the decoder |
| `backends` | yes | List of backends (see below) |
| `codes` | yes | List of codes: `surface`, `hh` (heavy-hex), `gross`, `color`, `bacon` (Bacon-Shor), `steane` (concatenated Steane) |
| `decoders` | yes | List of decoders: `bposd`, `bposd_chk`, `bposd_batch`, `mwpm`, `chromobius` (color code only) |
| `error_types` | yes | List of noise models (see below) |
| `error_probabilities` | for artificial noise | Physical error rates `p` used by the artificial noise models |
| `backends_sizes` | for `custom_*` backends | Number of qubits of the custom topology. The code distance is then chosen as the largest distance that fits on the backend |
| `distances` | no | Fixed code distances. The backend size is set to the minimum needed for the code. **Cannot be combined with `backends_sizes`** |
| `cycles` | no | Number of syndrome-extraction rounds (single integer, must be ≥ 2). Defaults to the code distance |
| `layout_methods` | no | Qiskit layout methods, e.g. `trivial`, `dense`, `sabre` |
| `routing_methods` | no | Qiskit routing methods, e.g. `basic`, `sabre`, `stochastic`, `lookahead` |
| `translating_methods` | no | Gate translation before transpilation: `qiskit`, `bqskit`, `tket` |
| `gate_sets` | no | Target gate sets, used only by `get_translate_stats.py`: `h2`, `ibm_heron` |

If neither `backends_sizes` nor `distances` is given, the backend size comes from the backend itself and the largest code distance that fits is used.

### Backends

| Value | Description |
| --- | --- |
| `custom_line`, `custom_grid`, `custom_cube`, `custom_full`, `custom_heavyhex` | Synthetic topologies; require `backends_sizes` (or `distances`) |
| `real_flamingo`, `real_loon` | IBM Flamingo and IBM Loon/Nighthawk (multi-chip, with remote gates) |
| `real_flamingo_1_qpu`, `real_loon_1_qpu` | The same devices restricted to a single QPU |
| `real_torino` | IBM Torino (Qiskit fake backend) |
| `real_willow` | Google Willow |
| `real_apollo`, `real_nsapollo` | Quantinuum Apollo with and without shuttling |
| `real_infleqtion`, `real_nsinfleqtion` | Infleqtion with and without shuttling |
| `variance_<level>_<scale>` | Flamingo-like device with T1/T2 variance `<level>` ∈ {`none`, `low`, `mid`, `high`} and coherence times scaled by `<scale>`, e.g. `variance_mid_10` |
| `ibm_<device>` | A real IBM device through `QiskitRuntimeService` (requires an IBM token) |

### Noise models (`error_types`)

| Value | Description |
| --- | --- |
| `sd6`, `pm3`, `em3_1`, `em3_2`, `si1000`, `modsi1000`, `constant` | Artificial noise models parameterised by `error_probabilities` |
| `real_willow`, `real_apollo`, `real_infleqtion`, `real_heron` | Hardware-inspired noise models |
| `real_flamingo`, `real_flamingo_downscaled` | IBM Flamingo noise, and the same noise scaled down 10× |
| `variance` | Decoherence driven by the T1/T2 values of a `variance_*` backend |
| `variance_low`, `variance_mid`, `variance_high` | Readout error drawn from a log-normal distribution with increasing variance |

Hardware noise models must be paired with a matching backend, for example `real_infleqtion` with `real_infleqtion`.

## Reproducing the paper

Each figure has its own runner and configuration:

| Experiment | Command |
| --- | --- |
| Code size | `python3 main.py experiments_conf/size.yaml` |
| Connectivity and real topologies | `python3 main.py experiments_conf/connectivity.yaml` |
| Variance | `python3 main.py experiments_conf/variance.yaml` |
| Technologies | `python3 main.py experiments_conf/technologies.yaml` |
| Distributed QC | `python3 get_dqc.py experiments_conf/dqc.yaml` |
| Routing | `python3 get_routing_stats.py routing_experiments.yaml` |
| Gate translation | `python3 get_translate_stats.py translate_experiments.yaml` |
| Accumulated error / thresholds | `python3 get_accumulated_err.py experiments_conf/accumulated.yaml` |
| Decoders | `python3 get_decoder.py experiments_conf/decoder.yaml` |
| Chromobius decoder | `python3 get_chromobius.py experiments_conf/decoder_chromobius.yaml` |
| Program statistics | `python3 get_program_stats.py experiments_conf/program_stats.yaml` |
| Plots | `python3 plots/plots.py` (reads `experiment_results/`, writes PDFs to `data/`) |

To run everything in sequence:

```bash
bash run_experiments.sh
```

The full suite takes a long time. The results used in the paper are already in `experiment_results/`, so `python3 plots/plots.py` regenerates the figures without re-running anything.

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
    docker cp eccentric_run:/app/data ./data_docker
    docker rm eccentric_run
    ```

To run a single experiment instead, override the command, for example `docker run --rm eccentric python3 main.py experiments.yaml`.

## Comparison with real IBM hardware

`ibm_submit.py` submits random circuits (2–7 qubits, depth 100, 10 circuits per width, 1000 shots each) to an IBM Quantum backend. `ibm_demo.py` then compares the measured fidelities with noiseless simulation and the noise models in this repository.

```bash
python3 ibm_submit.py submit            # submit the job (asks for confirmation; uses IBM credits)
python3 ibm_submit.py results <job_id>  # download and store the results
python3 ibm_demo.py [<job_id>]          # analyse and plot (defaults to the job used in the paper)
```

Job metadata and results are stored in `experiment_results/ibm_runs/`. These scripts need an IBM Quantum API token. Follow the [IBM Quantum setup guide](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup) to save it in your environment.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'qiskit_qec'` or missing compiled extensions**: the submodules are missing or `qiskit_qec` was not built. Run `git submodule update --init --recursive`, then `python setup.py build_ext --inplace` in `external/qiskit_qec`.
- **`ModuleNotFoundError` for `backends`, `codes`, …**: run the scripts from the repository root.
- **A configuration produces no row in `results.csv`**: check `experiment_results/<name>/experiment.log`. Configurations where the code does not fit on the backend (distance < 3) are skipped and logged.

## Citation

```bibtex
@inproceedings{swierkowska2026eccentric,
  title     = {ECCentric: An Empirical Analysis of Quantum Error Correction Codes},
  author    = {{\'S}wierkowska, Aleksandra and Pflieger, Jannik and Giortamis, Emmanouil and Bhatotia, Pramod},
  booktitle = {Proceedings of the ACM SIGMETRICS Conference (SIGMETRICS'26)},
  year      = {2026},
  note      = {arXiv:2511.01062}
}
```
