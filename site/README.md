# site/

What the website ([eccentric-benchmark.github.io](https://eccentric-benchmark.github.io)) shows, built from this
repository's dataset. The website copies this directory; nothing here is edited by hand.

- `figures/`: `python -m eccentric.figures` builds every figure as SVG and PNG, the plotted values as CSV, and
  `manifest.json` with each figure's title, finding, caption, and the dataset and commit it was built from.
  Rebuild after every new sweep table (`python -m sweep.collect`, then `python -m sweep.stats`).
