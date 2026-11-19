from sweep.jobs import config_id


def test_z_basis_configurations_keep_their_ids():
    # Configurations from before the memory basis was a setting have no basis key; their ids must not change
    config = dict(code="surface", schedule=None, distance=3, cycles=3, backend="real_willow", error_type="real_willow",
                  error_probability=None)
    assert config_id("willow_surface", config) == \
        "willow_surface.code=surface.distance=3.cycles=3.backend=real_willow.error_type=real_willow"
    assert config_id("basis_x", dict(config, basis="X")) == \
        "basis_x.code=surface.distance=3.cycles=3.backend=real_willow.error_type=real_willow.basis=X"
