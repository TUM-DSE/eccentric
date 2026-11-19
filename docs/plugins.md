# Plugins

Other packages can add codes, decoders, devices, noise models and command-line tools to ECCentric without changing
it. A plugin is an [entry point](https://packaging.python.org/en/latest/specifications/entry-points/) in the
package's `pyproject.toml`: its name is the name configurations use, its object is loaded the first time that
name is asked for. Install the package next to ECCentric and the name works in every configuration.

```toml
[project]
name = "my-qec-plugins"
dependencies = ["eccentric"]

[project.entry-points."eccentric.codes"]
my_code = "my_qec_plugins.codes:my_code"

[project.entry-points."eccentric.decoders"]
my_decoder = "my_qec_plugins.decoders:decode"

[project.entry-points."eccentric.backends"]
my_device = "my_qec_plugins.devices:make"

[project.entry-points."eccentric.noise_models"]
my_noise = "my_qec_plugins.noise:make"

[project.entry-points."eccentric.commands"]
my-tool = "my_qec_plugins.cli:main"
```

`eccentric plugins` lists what is installed.

## What each entry point must be

| Group | Object | Used by |
| --- | --- | --- |
| `eccentric.codes` | An `eccentric.codes.catalog.CodeFamily` (parametrized by distance) or `FixedCode` (one [[n, k, d]] instance) | `codes:` |
| `eccentric.decoders` | `decode(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool)`, returning the logical error rate, or `None` when it cannot decode the circuit | `decoders:` |
| `eccentric.backends` | `make(backend_size: int)`, returning a Qiskit `BackendV2` | `backends:` |
| `eccentric.noise_models` | `make(qt, p, backend)`, returning an `eccentric.noise.noise.NoiseModel` | `error_types:`; `<name>_x<factor>` scales it like any hardware model |
| `eccentric.commands` | `main(argv: list[str])`, returning an exit code or `None` | `eccentric <name> ...` |

A code's `make` builds a noiseless Stim memory circuit; the circuit then goes through the same compilation, noise
and decoding as the built-in codes. See `eccentric/codes/catalog.py` for examples:

```python
import stim
from eccentric.codes.catalog import CodeFamily

def _make(d, rounds, schedule, basis):
    return stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", distance=d, rounds=rounds)

my_code = CodeFamily(_make, num_qubits=lambda d: 2 * d * d - 1)
```

A decoder plugin gets the noisy circuit and samples its own shots; `eccentric/decoders/` has examples.

## Rules

- **Built-in names win.** A plugin with the name of a built-in code, decoder, device or noise model is never
  used, so installing a plugin cannot change the results of an existing configuration.
- **Results name the plugin** the same way they name built-ins: the `code`, `decoder`, `backend` and
  `error_type` columns hold the plugin's name. Record the plugin's version alongside your results.
- Plugins load in every worker process, so they work with parallel runs (`-j`).
