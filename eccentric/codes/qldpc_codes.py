import itertools

import numpy as np
import stim
from qldpc import abstract, circuits, codes
from qldpc.objects import Pauli

# Noiseless memory experiments built with qLDPC (https://github.com/qLDPCOrg/qLDPC):
# each syndrome round measures all checks with ancillas scheduled by edge coloring


def make_memory_circuit(code: codes.CSSCode, rounds: int, basis: str = "Z") -> stim.Circuit:
    return circuits.get_memory_experiment(code, basis=Pauli[basis], num_rounds=rounds)


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
