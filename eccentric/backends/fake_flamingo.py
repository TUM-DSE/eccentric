import numpy as np
from qiskit.transpiler import Target, CouplingMap
from qiskit.providers import BackendV2, Options
from qiskit.visualization import plot_coupling_map
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers import QubitProperties
from qiskit_ibm_runtime.fake_provider import FakeKyiv

class FakeIBMFlamingo(BackendV2):

    def __init__(self, num_qpus: int = 3):
        super().__init__(name="FakeIBMFlamingo", backend_version=2)
        self.num_qpus = num_qpus
        self._remote_gates, self._coupling_map = self.get_endpoints()
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake IBM Flamingo", num_qubits=self._num_qubits)
        self.addStateOfTheArtQubits()
        self.gate_set = ["id", "sx", "x", "rz", "rzz", "cz", "rx"]

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def get_remote_gates(self):
        return self._remote_gates

    @property
    def coupling_map(self):
        return self._coupling_map

    @property
    def qubit_positions(self):
        return self._qubit_positions

    @property
    def num_qubits(self):
        return self._num_qubits

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)


    def heavySquareHeronCouplingMap(self):
        backend = FakeKyiv()

        base_coupling_map = backend.coupling_map

        base_coupling_map.add_edge(13, 127)
        base_coupling_map.add_edge(113, 128)
        base_coupling_map.add_edge(117, 129)
        base_coupling_map.add_edge(121, 130)
        base_coupling_map.add_edge(124, 131)
        base_coupling_map.add_edge(128, 132)

        return base_coupling_map


    def DQCCouplingMap(self, coupling_map1: CouplingMap, coupling_map2: CouplingMap, endpoints: list):
        edges_1 = coupling_map1.get_edges()
        last_qubit = coupling_map1.size()
        converted_edges_1 = [[i, j] for i,j in edges_1]
        new_endpoints = [[i, last_qubit + j] for i,j in endpoints]
        new_coupling_map2 = [[last_qubit + i, last_qubit + j] for i,j in coupling_map2]

        new_coupling_map = CouplingMap(converted_edges_1 + new_endpoints + new_coupling_map2)
        new_endpoints_tuple = [(i, j) for i,j in new_endpoints]

        return new_endpoints_tuple, new_coupling_map

    def get_endpoints(self):
        # A chain of QPUs, each linked to the next by four l-couplers
        qpu = self.heavySquareHeronCouplingMap()
        all_endpoints, coupling_map = [], qpu
        for i in range(1, self.num_qpus):
            offset = (i - 1) * qpu.size()
            endpoints, coupling_map = self.DQCCouplingMap(coupling_map, qpu, [[offset + 32, 18], [offset + 51, 37], [offset + 70, 56], [offset + 89, 75]])
            all_endpoints += endpoints
        return all_endpoints, coupling_map

    def addStateOfTheArtQubits(self):
        qubit_props = []

        for i in range(self._num_qubits):
            t1 = np.random.normal(190, 120, 1)
            t1 = np.clip(t1, 50, 500)
            t1 = t1 * 1e-6

            t2 = np.random.normal(130, 120, 1)
            t2 = np.clip(t2, 50, 650)
            t2 = t2 * 1e-6

            qubit_props.append(QubitProperties(t1=t1, t2=t2, frequency=5.0e9))

        self.target.qubit_properties = qubit_props


    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")


if __name__ == "__main__":
    backend = FakeIBMFlamingo()
    print(backend.get_remote_gates)
