# Reproducing the SIGMETRICS'26 paper

To reproduce the paper exactly, check out the tag `v1.0.0-sigmetrics26`. Every experiment of the paper has
a configuration in [`configs/`](../configs):

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

## With Docker

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
