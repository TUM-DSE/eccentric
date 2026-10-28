from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeTorino
from .custom_topologies import get_custom_topology
from .fake_apollo import FakeQuantinuumApolloBackend
from .fake_apollo_no_shuttling import FakeQuantinuumApolloNoShuttlingBackend
from .fake_flamingo import FakeIBMFlamingo
from .fake_loon import FakeIBMLoon
from .fake_native import FakeNativeBackend
from .fake_bicycle import FakeIBMBicycle
from .fake_willow import FakeGoogleWillowBackend
from .fake_infleqtion import FakeInfleqtionBackend
from .fake_infleqtion_no_shuttling import FakeInfleqtionNoShuttlingBackend
from .fake_harvard import FakeHarvardBackend
from .variance_backend import VarianceBackend

def get_backend(backend: str, backend_size: int):
    backend_type = backend.split("_")[0]
    if backend_type == "ibm":
        service = QiskitRuntimeService(instance="open-instance")
        backend = service.backend(backend)
    elif backend_type == "custom" and backend_size:
        shape = backend.split("_")[1]
        backend = get_custom_topology(shape, backend_size)
    elif backend_type == "real":
        parts = backend.split("_")
        name = parts[1]
        # real_loon_1_qpu is a single QPU, real_loon_5_qpus a projected chain of five
        num_qpus = {"num_qpus": int(parts[-2])} if len(parts) >= 4 and parts[-1] in ("qpu", "qpus") else {}
        if name == "flamingo":
            backend = FakeIBMFlamingo(**num_qpus)
        elif name == "loon":
            backend = FakeIBMLoon(**num_qpus)
        elif name == "bicycle":
            # real_bicycle_lr: c-couplers as noisy as remote ones
            backend = FakeIBMBicycle(noisy_c_couplers=parts[2:3] == ["lr"], **num_qpus)
        elif name == "torino":
            backend = FakeTorino()
        elif name == "willow":
            backend = FakeGoogleWillowBackend(extended=True)
        elif name == "apollo":
            backend = FakeQuantinuumApolloBackend()
        elif name == "nsapollo":
            backend = FakeQuantinuumApolloNoShuttlingBackend()
        elif name == "infleqtion":
            backend = FakeInfleqtionBackend(extended=True)
        elif name == "nsinfleqtion":
            backend = FakeInfleqtionNoShuttlingBackend(extended=True)
        elif name == "harvard":
            backend = FakeHarvardBackend()
        else:
            raise NotImplementedError
    elif backend_type == "variance":
        name = backend.split("_")[1]
        scale = backend.split("_")[2]
        backend = VarianceBackend(name, scale)
    else:
        raise NotImplementedError
    return backend
