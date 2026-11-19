# Configuration and command line

## Running

```bash
python -m eccentric configs/example.yaml
```

This runs the small example in [`configs/example.yaml`](../configs/example.yaml) and writes to `experiment_results/Example/`:

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
| `eccentric plugins` | List the installed plugins ([docs/plugins.md](plugins.md)) |
| `eccentric <command> ...` | Run a command added by a plugin or tool |

## Configuration fields


| Field | Description |
| --- | --- |
| `name` | Experiment name; results go to `experiment_results/<name>/` |
| `type` | Experiment type: `logical_error` (default), `decoder`, `program_stats`, `accumulated_error`, `chromobius`, `routing`, `translation` |
| `num_samples` | Number of shots (or, for `routing` and `translation`, number of repetitions) |
| `backends` | List of backends  |
| `codes` | List of codes: `surface`, `hh` (heavy-hex), `gross`, `color`, `bacon` (Bacon-Shor), `steane` (concatenated Steane), and the codes listed in [codes.md](codes.md) |
| `decoders` | List of decoders: `bposd`, `bposd_chk`, `bposd_batch`, `mwpm`, `chromobius` (color code only), `bplsd`, `union_find`, `belief_find`, `belief_matching`, `relay_bp`, `tesseract`, `mwpf`, `bposd_tuned`, `gari`, `gari_pp`, and GPU versions with a `_gpu` suffix (see [Decoders](#decoders)) |
| `error_types` | List of noise models  |
| `error_probabilities` | Physical error rates `p` for the artificial noise models |
| `backends_sizes` | Number of qubits of a `custom_*` backend. The largest code distance that fits is used |
| `distances` | Fixed code distances. The backend size is set to the minimum needed by the code. **Cannot be combined with `backends_sizes`** |
| `cycles` | Number of syndrome-extraction rounds (a single integer ≥ 2). Defaults to the code distance |
| `layout_methods` | Qiskit layout methods, e.g. `trivial`, `dense`, `sabre` |
| `routing_methods` | Qiskit routing methods, e.g. `basic`, `sabre`, `stochastic`, `lookahead` |
| `translating_methods` | Gate translation: `qiskit`, `bqskit`, `tket` |
| `gate_sets` | Target gate sets for `translation`: `h2`, `ibm_heron` |
| `optimization_levels` | Translation optimization levels for `translation` (default `[0, 2, 3]`) |
| `bases` | Memory bases, `Z` (default) and/or `X`. Result rows report the `basis`. The concatenated Steane, repetition and Floquet circuits exist only in the Z basis |
| `operations` | Fault-tolerant operations instead of the memory (see [codes.md](codes.md#fault-tolerant-operations-on-css-codes)); needs `distances` |
| `schedules` | Syndrome-extraction schedules (see [codes.md](codes.md)). Each code uses its default when omitted; configurations whose code does not support a schedule are skipped |

If neither `backends_sizes` nor `distances` is given, the backend's own size is used and the largest code distance that fits is chosen. Configurations whose code does not fit (distance < 3) are skipped and logged.
