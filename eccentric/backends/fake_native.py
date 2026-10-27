from qiskit.transpiler import Target, CouplingMap
from qiskit.providers import BackendV2, Options, QubitProperties
import numpy as np


class FakeNativeBackend(BackendV2):
    """A device built for one code: a qubit per code qubit and a coupler for every pair of qubits the
    syndrome circuit entangles, so the code is placed one to one and needs no SWAPs. For bivariate
    bicycle codes this is IBM's planned layout (arXiv:2308.07915), whose long-range c-couplers are
    assumed to be as good as nearest-neighbour couplers."""

    # Place circuit qubit i on device qubit i
    layout_method = "trivial"

    def __init__(self, num_qubits: int, pairs):
        super().__init__(name="FakeNativeBackend", backend_version="2")
        edges = sorted({(a, b) for a, b in pairs} | {(b, a) for a, b in pairs})
        self._coupling_map = CouplingMap(edges)
        for q in range(num_qubits):
            if q not in self._coupling_map.physical_qubits:
                self._coupling_map.add_physical_qubit(q)
        self._num_qubits = num_qubits
        self._remote_gates = []
        self._target = Target("Fake native device", num_qubits=num_qubits)
        self.addStateOfTheArtQubits()
        self.gate_set = ["id", "sx", "x", "rz", "rzz", "cz", "rx"]

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def coupling_map(self):
        return self._coupling_map

    @property
    def get_remote_gates(self):
        return self._remote_gates

    @property
    def num_qubits(self):
        return self._num_qubits

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")

    def addStateOfTheArtQubits(self):
        # Same coherence times as the Loon and Flamingo models
        qubit_props = []
        for i in range(self._num_qubits):
            t1 = np.clip(np.random.normal(190, 120, 1), 50, 500) * 1e-6
            t2 = np.clip(np.random.normal(130, 120, 1), 50, 650) * 1e-6
            qubit_props.append(QubitProperties(t1=t1, t2=t2, frequency=5.0e9))
        self.target.qubit_properties = qubit_props
