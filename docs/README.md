# Documentation

| | |
| --- | --- |
| **Running experiments** | [Configuration and command line](configuration.md) |
| **Reference** | [Codes](codes.md) · [Devices](devices.md) · [Noise models](noise-models.md) · [Decoders](decoders.md) |
| **The dataset** | [Columns and statuses](dataset.md) · [How it is produced](sweep.md) · [Analyses](../analysis/README.md) · [Figures](../site/README.md) |
| **Tools** | [QEC Matchmaker](../tools/matchmaker/README.md) · [How tools fit in](../tools/README.md) |
| **Extending** | [Plugins](plugins.md) · [Extending ECCentric](extending.md) |
| **More** | [Reproducing the paper](reproducing-the-paper.md) (incl. Docker) · [Comparison with real IBM hardware](ibm-hardware.md) · [Changes since v1.0.0](DETAILED_CHANGES.md) |

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
site/                 what the website shows: figures (python -m eccentric.figures) with their data
tests/                unit and end-to-end tests (pytest)
external/             submodules: qiskit-qec fork, more-bacon-less-threshold, honeycomb_threshold, honeycomb-boundaries
```
