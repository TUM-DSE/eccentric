# Extending ECCentric

- **As a plugin** (no changes to ECCentric): a package registers codes, decoders, devices, noise models or
  command-line tools through entry points, and they work in every configuration under their names. Built-in
  names always take precedence. See [plugins.md](plugins.md); `eccentric plugins` lists what is installed.
- **Tools on the dataset** read the sweep table through `eccentric.dataset`, a versioned schema
  ([dataset.md](dataset.md)). Tools maintained with ECCentric live in [`tools/`](../tools).
- **In ECCentric itself:** add a code, device, noise model or decoder to its package and its dispatcher
  (`get_code` in `eccentric/codes/utils.py` or the catalog in `eccentric/codes/catalog.py`, `get_backend` in
  `eccentric/backends/utils.py`, `get_noise_model` in `eccentric/noise/utils.py`, `decode` in
  `eccentric/decoders/decode.py`). A new experiment type is a module in `eccentric/experiments/` that defines
  `REQUIRED` (the required config fields) and `run(task)`, which returns a list of result rows, registered in
  `EXPERIMENTS` in `eccentric/experiments/__init__.py`; `eccentric/experiments/base.py` provides the shared steps.

Run the tests with `pytest`.
