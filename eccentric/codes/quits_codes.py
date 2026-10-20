import stim
from quits.qldpc_code import BpcCode

# Balanced product cyclic codes (Tiew & Breuckmann, arXiv:2411.03302) built with QUITS
# (https://github.com/mkangquantum/quits). p2 follows QUITS' transpose convention.
BPC_CODES = {
    36: dict(p1=[0, 1, 2], p2=[0, 4, 5], lift_size=6),
    72: dict(p1=[0, 1, 5], p2=[0, 4, 11], lift_size=12),
    90: dict(p1=[0, 1, 5], p2=[0, 8, 13], lift_size=15),
}


def make_bpc_circuit(n: int, rounds: int) -> stim.Circuit:
    code = BpcCode(**BPC_CODES[n], factor=3)
    circuit = code.build_circuit(strategy="cardinal", num_rounds=rounds, basis="Z", seed=1)
    return circuit.without_noise()
