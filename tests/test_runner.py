import glob

import pytest

from eccentric.experiments import EXPERIMENTS, Skip, Task, logical_error
from eccentric.experiments.base import resolve_cycles
from eccentric.runner import expand, load_config


def experiment(**overrides):
    config = {
        "name": "Test",
        "num_samples": 10,
        "backends": ["custom_grid"],
        "backends_sizes": [100, 200],
        "codes": ["surface", "color"],
        "decoders": ["bposd"],
        "error_types": ["sd6", "pm3"],
        "error_probabilities": [0.001],
    }
    config.update(overrides)
    return config


def test_expand_is_cartesian_product():
    module, tasks = expand(experiment())
    assert module is logical_error
    assert len(tasks) == 2 * 2 * 2
    assert {t.backend_size for t in tasks} == {100, 200}
    assert all(t.distance is None for t in tasks)


def test_distances_set_backend_size():
    config = experiment(distances=[3])
    del config["backends_sizes"]
    _, tasks = expand(config)
    sizes = {t.code: t.backend_size for t in tasks}
    assert sizes["surface"] == 2 * 9 + 3 * 3 - 1


def test_distances_and_sizes_are_exclusive():
    with pytest.raises(ValueError):
        expand(experiment(distances=[3]))


def test_missing_required_field():
    config = experiment()
    del config["decoders"]
    with pytest.raises(ValueError, match="decoders"):
        expand(config)


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown experiment type"):
        expand(experiment(type="nope"))


def test_translation_default_levels():
    _, tasks = expand({
        "name": "T",
        "type": "translation",
        "num_samples": 1,
        "backends": ["custom_full"],
        "backends_sizes": [300],
        "codes": ["surface"],
        "translating_methods": ["qiskit"],
    })
    assert sorted(t.optimization_level for t in tasks) == [0, 2, 3]


def test_cycles_must_exceed_one():
    task = Task(experiment="T", num_samples=1, backend="custom_full", code="surface", cycles=1)
    with pytest.raises(Skip):
        resolve_cycles(task, 3)


@pytest.mark.parametrize("path", sorted(glob.glob("configs/*.yaml")))
def test_shipped_configs_are_valid(path):
    for config in load_config(path):
        assert config.get("type", "logical_error") in EXPERIMENTS
        _, tasks = expand(config)
        assert tasks


def test_logical_error_end_to_end():
    task = Task(
        experiment="T",
        num_samples=200,
        backend="custom_full",
        backend_size=17,
        code="surface",
        distance=3,
        decoder="mwpm",
        error_type="sd6",
        error_probability=0.001,
    )
    (row,) = logical_error.run(task)
    assert row["distance"] == 3 and row["cycles"] == 3
    assert 0.0 <= float(row["logical_error_rate"]) <= 1.0
