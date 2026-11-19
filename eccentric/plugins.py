"""Plugins: codes, decoders, devices, noise models and commands that other packages add to ECCentric.

A package registers a plugin through an entry point in its pyproject.toml. The entry point's name is the
name configurations and the command line use, its object is loaded on first use:

    [project.entry-points."eccentric.decoders"]
    my_decoder = "my_package.decoder:decode"

| Group                    | Object                                                                     |
| ------------------------ | -------------------------------------------------------------------------- |
| `eccentric.codes`        | an `eccentric.codes.catalog.CodeFamily` or `FixedCode`                       |
| `eccentric.decoders`     | `decode(circuit: stim.Circuit, num_shots, approximate_disjoint_errors)`      |
|                          | returning the logical error rate, or None if the decoder cannot decode it   |
| `eccentric.backends`     | `make(backend_size)` returning a Qiskit backend                              |
| `eccentric.noise_models` | `make(qt, p, backend)` returning an `eccentric.noise.noise.NoiseModel`       |
| `eccentric.commands`     | `main(argv: list[str]) -> int or None`, run as `eccentric <name> ...`        |

Built-in names always take precedence: a plugin with the name of a built-in is never used.
See docs/plugins.md.
"""
from functools import lru_cache
from importlib.metadata import entry_points

GROUPS = {
    "codes": "eccentric.codes",
    "decoders": "eccentric.decoders",
    "backends": "eccentric.backends",
    "noise_models": "eccentric.noise_models",
    "commands": "eccentric.commands",
}


@lru_cache(maxsize=None)
def _entry_points(kind: str) -> dict:
    """The entry points of one group by name, not yet loaded"""
    return {ep.name: ep for ep in entry_points(group=GROUPS[kind])}


@lru_cache(maxsize=None)
def get(kind: str, name: str):
    """The plugin object registered under `name` in group `kind`, or None"""
    ep = _entry_points(kind).get(name)
    if ep is None:
        return None
    obj = ep.load()
    if kind == "codes":
        from .codes.catalog import CodeFamily, FixedCode
        if not isinstance(obj, (CodeFamily, FixedCode)):
            raise TypeError(f"code plugin {name} ({ep.value}) must be a CodeFamily or FixedCode, "
                            f"not {type(obj).__name__}")
    return obj


def names(kind: str) -> list:
    """The names registered in group `kind`"""
    return sorted(_entry_points(kind))


def describe() -> dict:
    """Every registered plugin: {kind: {name: "module:object (distribution)"}}"""
    out = {}
    for kind in GROUPS:
        out[kind] = {}
        for name, ep in sorted(_entry_points(kind).items()):
            dist = getattr(ep, "dist", None)
            out[kind][name] = f"{ep.value}" + (f" ({dist.name} {dist.version})" if dist else "")
    return out


def reset():
    """Forget loaded entry points (tests, or after installing a plugin in a running interpreter)"""
    _entry_points.cache_clear()
    get.cache_clear()
