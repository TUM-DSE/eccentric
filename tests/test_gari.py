import numpy as np
import pytest
import stim

pytest.importorskip("ldpc.ckt_noise.dem_matrices")

from eccentric.decoders.gari import gari, gari_encoding  # noqa: E402
from eccentric.decoders.mwpm import mwpm  # noqa: E402


def surface(basis="z", d=3, p=0.003):
    return stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=d, rounds=d,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p, before_round_data_depolarization=p)


@pytest.mark.parametrize("basis", ["z", "x"])
def test_gari_matrix_has_the_effective_error_structure(basis):
    enc = gari_encoding(surface(basis))
    # rows: the detectors, then one check per single-type error; columns: effective and original single-type
    # errors, then the Y-type ones
    assert enc.hrows.shape[0] == len(enc.order) + enc.nx + enc.nz
    assert enc.hcols.shape[0] >= 2 * (enc.nx + enc.nz)
    # Detectors are labelled in the data qubits' preparation frame, so an X-basis memory (prepared with RX) has its
    # observable on the same side as a Z-basis one
    assert not enc.flip_sides
    assert enc.nx > 0 and enc.nz > 0 and len(enc.i_y) > 0
    assert (enc.llr[:enc.nx + enc.nz] == 0).all()


@pytest.mark.parametrize("basis", ["z", "x"])
def test_gari_decodes_the_surface_code_below_threshold(basis):
    circuit = surface(basis, d=5, p=0.002)
    ler = gari(circuit, 4000, False)
    # GARI alone does not correct the shots it does not converge on, so it is not as good as matching, but it
    # suppresses errors far below the undecoded rate
    assert ler < 0.1
    assert ler < 10 * max(mwpm(circuit, 4000, False), 1e-3)


def test_gari_refuses_non_css_circuits():
    from eccentric.codes import get_code
    xzzx = get_code("xzzx", 3, 3).stim_circuit
    noisy = stim.Circuit()
    for inst in xzzx.flattened():
        noisy.append(inst)
        if inst.name in ("CX", "CZ", "XCX"):
            noisy.append("DEPOLARIZE2", [t.value for t in inst.targets_copy()], 0.001)
    with pytest.raises(ValueError):
        gari_encoding(noisy)
