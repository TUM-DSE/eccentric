import logging
from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit
from bqskit import MachineModel, compile
from bqskit.ir.gates import *
from pytket import OpType
from pytket.passes import AutoRebase, DecomposeBoxes, RemoveBarriers, GreedyPauliSimp, RemoveRedundancies, FullPeepholeOptimise, SequencePass, KAKDecomposition, CliffordSimp, SynthesiseTket
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import transpile


qiskit_stim = [
    "x",
    "y",
    "z",
    "cx",
    "cz",
    "cy",
    "h",
    "s",
    "s_dag",
    "swap",
    "reset",
    "measure",
    "barrier",
]

bqskit_stim = [
    XGate(),
    YGate(),
    ZGate(),
    CXGate(),
    CZGate(),
    CYGate(),
    HGate(),
    SGate(),
    SdgGate(),
    SwapGate(),
    Reset(),
]

tket_stim = [
    OpType.X,
    OpType.Y,
    OpType.Z,
    OpType.CX,
    OpType.CZ,
    OpType.CY,
    OpType.H,
    OpType.S,
    OpType.Sdg,
    OpType.SWAP,
    OpType.Reset,
    OpType.Measure,
    OpType.Barrier,
]

qiskit_ibm_heron = [
    "x",
    "sx",
    "rzz",
    "rz",
    "rx",
    'cz',
    'measure',
    'reset',
    'barrier',
]

bqskit_ibm_heron = [
    XGate(),
    SXGate(),
    RZZGate(),
    RZGate(),
    RXGate(),
    CZGate(),
    Reset(),

]

tket_ibm_heron = [
    OpType.X,
    OpType.SX,
    OpType.ZZPhase,
    OpType.Rz,
    OpType.Rx,
    OpType.CZ,
    OpType.Measure,
    OpType.Reset,
    OpType.Barrier,
]

qiskit_h2 = [
    "rx",
    "ry",
    "rz",
    "rzz",
    'rxx',
    "ryy"
    "reset",
    "measure",
    "barrier",
]

bqskit_h2 = [
    RXGate(),
    RYGate(),
    RZGate(),
    ZZGate(),
    RZZGate(),
    RXXGate(),
    RYYGate(),
    Reset(),
]


tket_h2 =[
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
    OpType.ZZPhase,
    OpType.ZZMax,
    OpType.TK2,
    OpType.Measure,
    OpType.Reset,
    OpType.Barrier,
]


def translate(circuit, translating_method, gate_set=None, level=0):
    if gate_set:
        full_gate_set_name = f"{translating_method}_{gate_set}"
        try:
            gate_set_obj = globals()[full_gate_set_name]
        except KeyError:
            logging.error(f"Gate set '{full_gate_set_name}' is not defined.")
            raise ValueError(f"Unknown gate set: {full_gate_set_name}")
    else:
        gate_set_obj = {
            "qiskit": qiskit_stim,
            "bqskit": bqskit_stim,
            "tket": tket_stim,
        }.get(translating_method)

        if gate_set_obj is None:
            logging.error(f"Default gate set not defined for: {translating_method}")
            raise ValueError(f"Missing default gate set for: {translating_method}")

    if translating_method == "qiskit":
        qiskit_circuit = transpile(
            circuit,
            basis_gates=gate_set_obj,
            optimization_level=level,
        )
        return qiskit_circuit

    elif translating_method == "bqskit":
        bqskit_circuit = qiskit_to_bqskit(circuit)
        model = MachineModel(bqskit_circuit.num_qudits, gate_set=gate_set_obj)
        bqskit_circuit = compile(bqskit_circuit, model=model, optimization_level=1)
        return bqskit_circuit

    elif translating_method == "tket":
        tket_circuit = qiskit_to_tk(circuit)
        if level == 0:
            rebase_pass = AutoRebase(set(gate_set_obj))
            rebase_pass.apply(tket_circuit)
        elif level == 2:
            level2_pipeline = SequencePass([
                DecomposeBoxes(),
                FullPeepholeOptimise(),
                KAKDecomposition(allow_swaps=False),
                CliffordSimp(allow_swaps=False),
                SynthesiseTket(),
                AutoRebase(set(gate_set_obj)),
                RemoveRedundancies()
            ])
            level2_pipeline.apply(tket_circuit)
        elif level == 3:
            level3_pipeline = SequencePass([
                DecomposeBoxes(),
                RemoveBarriers(),
                AutoRebase(set(gate_set_obj)),
                GreedyPauliSimp(),
                AutoRebase(set(gate_set_obj)),
                SynthesiseTket(),
                AutoRebase(set(gate_set_obj)),
                RemoveRedundancies()
            ])
            level3_pipeline.apply(tket_circuit)
        return tket_circuit

    else:
        logging.error(f"Unknown translate method: {translating_method}")
        raise NotImplementedError
