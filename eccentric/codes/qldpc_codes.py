import itertools

import numpy as np
import stim
from qldpc import abstract, circuits, codes
from qldpc.objects import Pauli

# Noiseless memory experiments built with qLDPC (https://github.com/qLDPCOrg/qLDPC): each syndrome
# round measures all checks with one ancilla per check, scheduled by edge coloring of the Tanner
# graph, either of all checks together or of the X checks before the Z checks
SCHEDULES = {
    "edge_coloring": circuits.EdgeColoring,
    "edge_coloring_xz": circuits.EdgeColoringXZ,
}


def make_memory_circuit(code: codes.CSSCode, rounds: int, schedule: str = "edge_coloring", basis: str = "Z") -> stim.Circuit:
    return circuits.get_memory_experiment(
        code, basis=Pauli[basis], num_rounds=rounds, syndrome_measurement_strategy=SCHEDULES[schedule]()
    )


def toric_code(d: int) -> codes.CSSCode:
    # Unrotated toric code, [[2d^2, 2, d]]
    return codes.ToricCode(d, rotated=False)


def hgp_hamming_code() -> codes.CSSCode:
    # Hypergraph product of two [7,4,3] Hamming codes, [[58,16,3]]
    return codes.HGPCode(codes.HammingCode(3))


def reed_muller_15_code() -> codes.CSSCode:
    # Punctured quantum Reed-Muller code [[15,1,3]]: X checks are the 4 weight-8 rows of the
    # [15,4,8] simplex code, Z checks are those rows and their 6 pairwise products
    columns = [v for v in itertools.product([0, 1], repeat=4) if any(v)]
    hx = np.array(columns, dtype=int).T
    hz = np.vstack([hx, [hx[i] & hx[j] for i, j in itertools.combinations(range(4), 2)]])
    return codes.CSSCode(hx, hz)


def lifted_product_code() -> codes.CSSCode:
    # Lift-connected surface code (arXiv:2401.02911): three distance-3 surface codes stitched
    # together with lifted connections, [[39,3,3]]
    group = abstract.CyclicGroup(3)
    ring = abstract.GroupRing(group)
    x = ring.generators[0]
    rep = abstract.RingArray.build(codes.RepetitionCode(3).matrix, ring)
    links = abstract.RingArray.build(np.zeros((2, 3), dtype=int), ring)
    for row in range(2):
        links[row, row + 1] = x
    return codes.LPCode(rep + links)


def quantum_tanner_code(seed: int) -> codes.CSSCode:
    # Quantum Tanner code on the dihedral group D4 with repetition-code subcodes: seed 1 gives
    # [[36,8,3]], seed 0 gives [[72,14,4]]
    return codes.QTCode.random(abstract.DihedralGroup(4), codes.RepetitionCode(3), seed=seed)


# Generalized bicycle codes from Panteleev & Kalachev, arXiv:1904.02703, Appendix B:
# H_X = [A | B], H_Z = [B^T | A^T] with A, B circulant, given as (size, powers of A, powers of B)
GB_CODES = {
    46: (23, [0, 5, 8, 12], [0, 1, 5, 7]),
    48: (24, [0, 2, 8, 15], [0, 2, 12, 17]),
    126: (63, [0, 1, 14, 16, 22], [0, 3, 13, 20, 42]),
}


def generalized_bicycle_code(n: int) -> codes.CSSCode:
    from .gross_code import create_circulant_matrix

    size, a, b = GB_CODES[n]
    A, B = create_circulant_matrix(size, a), create_circulant_matrix(size, b)
    return codes.CSSCode(np.hstack((A, B)), np.hstack((B.T, A.T)))


def bicycle_code(l: int, m: int, a_terms, b_terms) -> codes.CSSCode:
    """Two-block code over Z_l x Z_m: H_X = [A | B], H_Z = [B^T | A^T], where A and B are sums of
    monomials x^i y^j, given as (i, j), with x = S_l (x) I_m and y = I_l (x) S_m"""
    def shift(size, power):
        return np.roll(np.eye(size, dtype=int), power % size, axis=1)

    def poly(terms):
        return sum(np.kron(shift(l, i), shift(m, j)) for i, j in terms) % 2

    A, B = poly(a_terms), poly(b_terms)
    return codes.CSSCode(np.hstack((A, B)), np.hstack((B.T, A.T)))


# Coprime bivariate bicycle codes (Wang & Mueller, arXiv:2408.10001, Table 2): l and m coprime, A and
# B polynomials in pi = xy, which has order lm, given as (l, m, powers of pi in A, powers in B)
COPRIME_BB_CODES = {
    30: (3, 5, [0, 1, 2], [0, 2, 7]),
    42: (3, 7, [0, 2, 3], [0, 2, 10]),
    70: (5, 7, [0, 1, 5], [0, 1, 12]),
    108: (2, 27, [0, 3, 42], [0, 6, 39]),
    126: (7, 9, [0, 1, 58], [0, 13, 41]),
    154: (7, 11, [0, 1, 31], [0, 19, 53]),
}


def coprime_bicycle_code(n: int) -> codes.CSSCode:
    l, m, a, b = COPRIME_BB_CODES[n]
    return bicycle_code(l, m, [(p, p) for p in a], [(p, p) for p in b])


# Trivariate bicycle codes (Voss et al., arXiv:2406.19151, Appendix A): bivariate bicycle codes whose
# A and B may also use z = xy, given as (l, m, A, B) with x^i = (i, 0), y^j = (0, j), z^k = (k, k).
# Check weights 5, 6, 5 and 4
TRIVARIATE_BICYCLE_CODES = {
    30: (3, 5, [(1, 0), (4, 4)], [(1, 0), (0, 2), (2, 2)]),
    48: (4, 6, [(2, 0), (0, 4)], [(3, 0), (3, 3), (0, 2), (0, 1)]),
    72: (4, 9, [(1, 0), (0, 3)], [(2, 0), (0, 1), (0, 2)]),
    144: (8, 9, [(3, 0), (0, 7)], [(1, 0), (0, 5)]),
}


def trivariate_bicycle_code(n: int) -> codes.CSSCode:
    return bicycle_code(*TRIVARIATE_BICYCLE_CODES[n])


def many_hypercube_code(level: int) -> codes.CSSCode:
    # Concatenated [[6,4,2]] codes (Goto, arXiv:2403.16054): [[6^L, 4^L, 2^L]] at level L
    return codes.ManyHypercubeCode(level)


def quantum_golay_code() -> codes.CSSCode:
    # [[23,1,7]] CSS code from the classical [23,12,7] Golay code
    return codes.QuantumGolayCode()
