import os

import numpy as np
import stim
from quits.gf2_util import cyclic, direct_product, symmetric
from quits.qldpc_code import BpcCode, HgpCode, LcsCode, MittenCode, QlpCode
from quits.qldpc_code.mitten import sample_mitten_base_matrices

# QLDPC codes built with QUITS (https://github.com/mkangquantum/quits), with QUITS' cardinal
# syndrome-extraction schedule unless noted otherwise

# Balanced product cyclic codes (Tiew & Breuckmann, arXiv:2411.03302). p2 follows QUITS' transpose
# convention.
BPC_CODES = {
    36: dict(p1=[0, 1, 2], p2=[0, 4, 5], lift_size=6),
    54: dict(p1=[0, 1, 2], p2=[0, 7, 8], lift_size=9),
    72: dict(p1=[0, 1, 5], p2=[0, 4, 11], lift_size=12),
    90: dict(p1=[0, 1, 5], p2=[0, 8, 13], lift_size=15),
    108: dict(p1=[0, 1, 5], p2=[0, 16, 17], lift_size=18),
    126: dict(p1=[0, 1, 5], p2=[0, 14, 19], lift_size=21),
    144: dict(p1=[0, 1, 5], p2=[0, 13, 23], lift_size=24),
    162: dict(p1=[0, 1, 5], p2=[0, 16, 26], lift_size=27),
    180: dict(p1=[0, 1, 5], p2=[0, 13, 29], lift_size=30),
}

# Hypergraph products of classical (3,4)-regular LDPC codes shipped with QUITS
HGP_MATRICES = {
    225: "n=12_dv=3_dc=4_dist=6.txt",
    625: "n=20_dv=3_dc=4_dist=8_arXiv2311_03307.txt",
    900: "n=24_dv=3_dc=4_dist=10_arXiv2311_03307.txt",
}
HGP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "quits_hgp")

# Lift-connected surface codes (Old et al., arXiv:2401.02911) as (lift size L, base matrix width l+1)
LCS_CODES = {
    15: (3, 2),
    52: (4, 3),
    65: (5, 3),
    150: (6, 4),
    175: (7, 4),
}

# Quasi-cyclic lifted product code [[544,80,<=12]] (Xu et al., arXiv:2308.08648)
QLP_BASE = np.array([[0, 0, 0, 0, 0], [0, 2, 4, 7, 11], [0, 3, 10, 14, 15]])


def _circuit(code, rounds: int, strategy: str = "cardinal") -> stim.Circuit:
    kwargs = {"seed": 1} if strategy == "cardinal" else {}
    circuit = code.build_circuit(strategy=strategy, num_rounds=rounds, basis="Z", **kwargs)
    return circuit.without_noise()


def make_bpc_circuit(n: int, rounds: int) -> stim.Circuit:
    return _circuit(BpcCode(**BPC_CODES[n], factor=3), rounds)


def make_hgp_circuit(n: int, rounds: int) -> stim.Circuit:
    h = np.loadtxt(os.path.join(HGP_DIR, HGP_MATRICES[n]), dtype=int)
    return _circuit(HgpCode(h, h), rounds)


def make_lcs_circuit(n: int, rounds: int) -> stim.Circuit:
    # zxcoloration keeps more circuit-level distance than cardinal for these codes
    return _circuit(LcsCode(*LCS_CODES[n]), rounds, strategy="zxcoloration")


def make_qlp_circuit(rounds: int) -> stim.Circuit:
    return _circuit(QlpCode(QLP_BASE, QLP_BASE, 16), rounds)


def make_mitten_circuit(rounds: int) -> stim.Circuit:
    # The [[150,30]] instance over C5 x S3 used in QUITS' Mitten demo, with the Mitten-specific
    # depth-12 schedule. These random base matrices give distance <= 8, below the [[150,30,10]]
    # instance of arXiv:2607.28795.
    group = direct_product(cyclic(5), symmetric(3))
    A, B = sample_mitten_base_matrices(group, np.random.default_rng(2))
    return _circuit(MittenCode(A, B, group), rounds, strategy="custom")
