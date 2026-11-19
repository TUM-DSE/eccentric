# Tools

Tools built on ECCentric live here, each in its own directory with its own `pyproject.toml`, version, tests and
changelog. A tool:

- depends on the `eccentric` package, and only through its public interfaces: the experiment runner,
  `eccentric.dataset` for the sweep table ([docs/dataset.md](../docs/dataset.md)), and the dispatchers
  (`get_code`, `get_backend`, `get_noise_model`, `decode`);
- adds itself to the command line through an `eccentric.commands` entry point, so it runs as
  `eccentric <tool> ...` ([docs/plugins.md](../docs/plugins.md));
- records the dataset `schema_version` (and the table's commit) that its models or outputs were built from.

ECCentric itself never imports a tool.

```
tools/
└── <tool>/
    ├── pyproject.toml     depends on eccentric; [project.entry-points."eccentric.commands"]
    ├── README.md
    ├── CHANGELOG.md
    ├── <package>/
    └── tests/
```

Install a tool next to ECCentric with `pip install -e tools/<tool>`.
