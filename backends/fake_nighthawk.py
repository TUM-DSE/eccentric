from qiskit.transpiler import Target, CouplingMap
from qiskit.providers import BackendV2, Options, QubitProperties
from qiskit.visualization import plot_coupling_map
import numpy as np


class FakeIBMNighthawk(BackendV2):
    """Fake IBM Nighthawk Backend."""

    def __init__(self):
        super().__init__(name="FakeIBMNighthawk", backend_version="2")

        self._remote_gates, self._coupling_map = self.remote_rectangular_lattice_CouplingMap()

        self._num_qubits = self._coupling_map.size()

        # TODO: hardware limitations
        self._target = Target("Fake IBM Nighthawk", num_qubits=self._num_qubits)
        
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
    def qubit_positions(self):
        return self._qubit_positions
    
    @property
    def num_qubits(self):
        return self._num_qubits
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
    
    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")

    def remote_rectangular_lattice_CouplingMap(self, add_c_couplers: bool = False) -> tuple[list, CouplingMap]:
        """ Construct coupling map of a interconected square lattice for the Nighthawk processor 
        
        The coupling map consists of a simple rectangular lattice connectivity, with additional special c_couplers
        connecting distant nodes.
        Additionally, remote gates are constructed for 

        Information taken from:
        - https://www.ibm.com/quantum/blog/large-scale-ftqc
        - https://www.ibm.com/quantum/blog/nature-qldpc-error-correction

        TODO: Add variable amount of c_couplers. Currently only the first row can have distant connections.
        """
        
        GRID_SIZE = 17

        # Vertical and horizontal distance of c_couplers
        c_coupler_distance = 6

        def get_subgrid(row):
            """Returns subgrid index (0, 1, 2) based on row."""
            if row <= 5:
                return 0
            elif row <= 11:
                return 1
            else:
                return 2

        def to_node_id(node):
            row, col = node
            return row * GRID_SIZE + col

        grid = {}
        cross_subgrid_connections = []
        connections = []

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                node = (row, col)
                subgrid = get_subgrid(row)

                # Normal neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = (row + dr) % GRID_SIZE
                    nc = (col + dc) % GRID_SIZE
                    neighbor = (nr, nc)
                    neighbor_subgrid = get_subgrid(nr)
                    
                    connections.append((to_node_id(node), to_node_id(neighbor)))

                    if neighbor_subgrid != subgrid:
                        cross_subgrid_connections.append((to_node_id(node), to_node_id(neighbor)))

                # Longer range c-coupler connections
                for dr, dc in [(int(c_coupler_distance), int(c_coupler_distance/2)), (int(c_coupler_distance/2), c_coupler_distance)]:
                    nr = (row + dr) % GRID_SIZE
                    nc = (col + dc) % GRID_SIZE
                    neighbor = (nr, nc)
                    neighbor_subgrid = get_subgrid(nr)
                    connections.append((to_node_id(node), to_node_id(neighbor)))
                    if neighbor_subgrid != subgrid:
                        cross_subgrid_connections.append((to_node_id(node), to_node_id(neighbor)))

                grid[node] = connections

            #return grid, cross_subgrid_connections
        print(len(cross_subgrid_connections))
        return cross_subgrid_connections, CouplingMap(connections)

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
    

if __name__ == "__main__":
    backend = FakeIBMNighthawk()

    plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="nighthawk.png", planar=False)
