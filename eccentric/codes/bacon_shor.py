from baconshor import _bacon_shor
from baconshor._bacon_shor import make_bacon_shor_constructions
from gen import _gen_util

from qiskit_qec.circuits import StimCodeCircuit

# Based on https://github.com/Strilanc/more-bacon-less-threshold
def get_bacon_shor_code(d=5, cycles=1):
    construction = {**make_bacon_shor_constructions()}
    circuit = _gen_util._generate_single_circuit(
                constructions=construction,
                params=_gen_util.CircuitBuildParams(style='bacon_shor', rounds=cycles, diameter=d, custom={'b': 'Z'}),
                noise=None,
                debug_out_dir=None,
                convert_to_cz='0'
            )
    return StimCodeCircuit(stim_circuit=circuit)


if __name__ == "__main__":
    get_bacon_shor_code()
