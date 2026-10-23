"""What the validation expects of each code: check matrices, logical qubits and test settings."""
import numpy as np

# Codes of the paper, validated alongside the catalog codes in the decoder comparison
PAPER_CODES = {"surface": 5, "color": 5, "hh": 5, "bacon": 5, "steane": 9, "gross": 12}

HYPERBOLIC_K = {
    "hyperbolic_floquet_60": 5, "hyperbolic_floquet_120": 8, "hyperbolic_floquet_160": 10,
    "hyperbolic_floquet_240": 8, "hyperbolic_floquet_256": 34, "hyperbolic_floquet_320": 18,
}

# Circuits that measure fewer logical operators than the code has
TRACKED_OBSERVABLES = {"repetition": 1, "xzzx": 1, "honeycomb": 1, "planar_honeycomb": 1}

DECODERS = ["mwpm", "bposd", "bplsd", "belief_find", "union_find", "belief_matching", "relay_bp", "tesseract"]
ERROR_RATES = [0.001, 0.003]


def gf2_rank(m) -> int:
    from ldpc import mod2
    return int(mod2.rank(np.asarray(m, dtype=np.uint8)))


def css_matrices(name: str):
    """(hx, hz) of the code, or None for codes that are only defined by their circuits"""
    from eccentric.codes import gross_code, qldpc_codes, quits_codes

    if name.startswith("bb") and name[2:].isdigit():
        n = int(name[2:])
        (k, d), = [(k, d) for (n_, k, d) in gross_code.BBCODE_DICT if n_ == n]
        l, m, (ax, *ay), (by, *bx) = gross_code.BBCODE_DICT[(n, k, d)]
        code, _, _ = gross_code.create_bivariate_bicycle_codes(l, m, [ax], ay, bx, [by], code_distance=d)
        return np.asarray(code.hx), np.asarray(code.hz)

    qldpc = {
        "toric": lambda: qldpc_codes.toric_code(test_distance("toric")),
        "hgp_hamming": qldpc_codes.hgp_hamming_code,
        "rm15": qldpc_codes.reed_muller_15_code,
        "lp": qldpc_codes.lifted_product_code,
        "qt36": lambda: qldpc_codes.quantum_tanner_code(1),
        "qt72": lambda: qldpc_codes.quantum_tanner_code(0),
    }
    if name.startswith("gb"):
        qldpc[name] = lambda: qldpc_codes.generalized_bicycle_code(int(name[2:]))
    if name in qldpc:
        code = qldpc[name]()
        return np.asarray(code.matrix_x, dtype=np.uint8), np.asarray(code.matrix_z, dtype=np.uint8)

    from quits.qldpc_code import BpcCode, HgpCode, LcsCode, QlpCode
    if name.startswith("bpc"):
        code = BpcCode(**quits_codes.BPC_CODES[int(name[3:])], factor=3)
    elif name.startswith("hgp") and name[3:].isdigit():
        h = np.loadtxt(f"{quits_codes.HGP_DIR}/{quits_codes.HGP_MATRICES[int(name[3:])]}", dtype=int)
        code = HgpCode(h, h)
    elif name.startswith("lcs"):
        code = LcsCode(*quits_codes.LCS_CODES[int(name[3:])])
    elif name == "qlp544":
        code = QlpCode(quits_codes.QLP_BASE, quits_codes.QLP_BASE, 16)
    elif name == "mitten150":
        from quits.gf2_util import cyclic, direct_product, symmetric
        from quits.qldpc_code import MittenCode
        from quits.qldpc_code.mitten import sample_mitten_base_matrices
        group = direct_product(cyclic(5), symmetric(3))
        A, B = sample_mitten_base_matrices(group, np.random.default_rng(2))
        code = MittenCode(A, B, group)
    else:
        return None
    return np.asarray(code.hx, dtype=np.uint8), np.asarray(code.hz, dtype=np.uint8)


def expected_observables(name: str):
    if name in TRACKED_OBSERVABLES:
        return TRACKED_OBSERVABLES[name]
    if name in HYPERBOLIC_K:
        return HYPERBOLIC_K[name]
    if name.startswith("semi_hyperbolic_floquet"):
        return 4
    matrices = css_matrices(name)
    if matrices is None:
        return None
    hx, hz = matrices
    return hx.shape[1] - gf2_rank(hx) - gf2_rank(hz)


def test_distance(name: str) -> int:
    """Distance used for the decoder comparison: the smallest but one for families"""
    from eccentric.codes.catalog import CODES, FixedCode

    if name in PAPER_CODES:
        return PAPER_CODES[name]
    code = CODES[name]
    return code.distance if isinstance(code, FixedCode) else code.min_d + code.step
