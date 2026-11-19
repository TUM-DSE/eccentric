"""Plugins (eccentric.plugins): registered through entry points, used under their names, never over a built-in"""
from importlib.metadata import EntryPoint

import pytest

from eccentric import plugins
from eccentric.__main__ import main
from eccentric.codes.catalog import CODES
from eccentric.codes.utils import get_code, get_min_n
from eccentric.experiments import Task, logical_error
import plugin_fixtures

ENTRY_POINTS = {
    "eccentric.codes": {"plugin_surface": "plugin_surface", "surface": "surface"},
    "eccentric.decoders": {"plugin_decoder": "plugin_decoder", "bposd": "plugin_decoder"},
    "eccentric.backends": {"plugin_ring": "plugin_ring"},
    "eccentric.noise_models": {"plugin_noise": "plugin_noise"},
    "eccentric.commands": {"plugin_command": "plugin_command"},
}


@pytest.fixture(autouse=True)
def fake_entry_points(monkeypatch):
    def entry_points(group):
        return [EntryPoint(name, f"plugin_fixtures:{obj}", group) for name, obj in ENTRY_POINTS.get(group, {}).items()]

    monkeypatch.setattr(plugins, "entry_points", entry_points)
    plugins.reset()
    yield
    plugins.reset()
    CODES.pop("plugin_surface", None)


def test_listing():
    assert plugins.names("decoders") == ["bposd", "plugin_decoder"]
    assert "plugin_fixtures:plugin_ring" in plugins.describe()["backends"]["plugin_ring"]


def test_plugin_code_is_in_the_catalog():
    assert "plugin_surface" in CODES
    assert get_min_n("plugin_surface", 3) == 17
    assert get_code("plugin_surface", 3, 3).circuit is not None


def test_plugin_cannot_replace_a_builtin_code():
    assert "surface" not in CODES
    assert get_min_n("surface", 3) != 1


def test_unknown_names_still_fail():
    assert "no_such_code" not in CODES
    with pytest.raises(NotImplementedError, match="unknown decoder"):
        from eccentric.decoders.decode import decode
        decode("surface", None, 1, "no_such_decoder", "custom_full", "sd6")


def test_pipeline_with_plugin_code_device_noise_and_decoder():
    plugin_fixtures.decoded.clear()
    task = Task(experiment="T", num_samples=200, backend="plugin_ring", backend_size=40, code="plugin_surface",
                distance=3, cycles=3, decoder="plugin_decoder", error_type="plugin_noise", error_probability=0.001)
    (row,) = logical_error.run(task)
    assert plugin_fixtures.decoded == [200]
    assert 0.0 <= float(row["logical_error_rate"]) < 0.5


def test_builtin_decoder_wins():
    plugin_fixtures.decoded.clear()
    task = Task(experiment="T", num_samples=50, backend="custom_full", backend_size=get_min_n("surface", 3), code="surface",
                distance=3, cycles=3, decoder="bposd", error_type="sd6", error_probability=0.001)
    logical_error.run(task)
    assert plugin_fixtures.decoded == []


def test_scaled_plugin_noise_model():
    from eccentric.noise.noise import ScaledNoise
    from eccentric.noise.utils import get_noise_model
    assert isinstance(get_noise_model("plugin_noise_x0.5", None, 0.001, None), ScaledNoise)


def test_commands(capsys):
    assert main(["plugin_command", "a", "b"]) == 0
    assert "plugin command a b" in capsys.readouterr().out
    main(["plugins"])
    assert "plugin_ring" in capsys.readouterr().out
