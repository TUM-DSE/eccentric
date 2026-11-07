import stim
from ..utils.math import *

from qiskit_qec.circuits import StimCodeCircuit

def get_surface_code(d, T = None, basis="Z"):
    if T == None:
        T = d
    stim_circuit = stim.Circuit.generated(
    f"surface_code:rotated_memory_{basis.lower()}",
        rounds=T,
        distance=d
    )

    print("This works right?")
    return StimCodeCircuit(stim_circuit = stim_circuit)
