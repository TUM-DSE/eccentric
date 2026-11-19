from qiskit.transpiler import Target, CouplingMap
from qiskit.providers import BackendV2, Options
from qiskit.providers import QubitProperties
from qiskit.visualization import plot_coupling_map

# Harvard/MIT reconfigurable neutral-atom array with up to 448 atoms, from
# Bluvstein et al., Nature 649, 39-46 (2025), https://arxiv.org/abs/2506.20661


class FakeHarvardBackend(BackendV2):
    def __init__(self):
        super().__init__(name="FakeHarvard", backend_version=2)
        # 448 atoms laid out on a grid, nearest neighbours interact in place and any other pair
        # after moving an atom
        self.rows = 16
        self.columns = 28
        self._coupling_map = CouplingMap.from_grid(self.rows, self.columns)
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake Harvard NA", num_qubits=self._num_qubits)
        self.addStateOfTheArtQubits()
        self._remote_gates = self.add_shuttling_connections()

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

    def addStateOfTheArtQubits(self):
        # T2 > 1 s for the hyperfine clock qubit (Methods). T1 is not reported, so it is
        # conservatively set to T2.
        t1 = 1.0
        t2 = 1.0
        self.target.qubit_properties = [QubitProperties(t1=t1, t2=t2) for _ in range(self.num_qubits)]

    def add_shuttling_connections(self):
        existing_edges = set(self._coupling_map.get_edges())
        all_pairs = {(q1, q2) for q1 in range(self.num_qubits) for q2 in range(q1 + 1, self.num_qubits)}
        new_edges = all_pairs - existing_edges
        for edge in new_edges:
            self._coupling_map.add_edge(edge[0], edge[1])
        # A set, since the noise model looks up every two-qubit gate in it
        return new_edges

    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")


if __name__ == "__main__":
    backend = FakeHarvardBackend()
    plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="harvard.png")
