import stim
from quits.qldpc_code import BpcCode

# Balanced product cyclic codes (Tiew & Breuckmann, arXiv:2411.03302) built with QUITS
# (https://github.com/mkangquantum/quits). p2 follows QUITS' transpose convention.
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


def make_bpc_circuit(n: int, rounds: int) -> stim.Circuit:
    code = BpcCode(**BPC_CODES[n], factor=3)
    circuit = code.build_circuit(strategy="cardinal", num_rounds=rounds, basis="Z", seed=1)
    return circuit.without_noise()
