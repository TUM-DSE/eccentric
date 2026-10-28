import numpy as np
from .fake_native import FakeNativeBackend


class FakeIBMBicycle(FakeNativeBackend):
    """IBM's bicycle architecture module (arXiv:2308.07915, arXiv:2506.03094), the device Loon tests the
    couplers of: the 288 qubits of the gross code in the toric layout, each with four couplers to its
    neighbours on the torus and two long-range c-couplers. The c-couplers are as good as the others or,
    with noisy_c_couplers, as noisy as remote ones. Modules are chained by remote links. Unlike the
    native devices, it is the same device for every code, so other codes are placed and routed onto it."""

    # Let the transpiler place the circuit
    layout_method = None
    # Remote links between neighbouring modules, at these qubits of each
    LINKS = (72, 108, 144, 180)

    def __init__(self, num_qpus: int = 1, noisy_c_couplers: bool = False):
        from ..codes.gross_code import create_bivariate_bicycle_codes

        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1, 2], [1, 2], [3])
        half = code.N // 2
        # X checks, data qubits, Z checks
        def edges(h, offset):
            return {(offset + i, half + j) for i, j in zip(*np.nonzero(h))}
        a1, b1 = A_list[0].toarray(), B_list[0].toarray()
        couplers = edges(code.hx, 0) | edges(code.hz, half + code.N)
        c_couplers = edges(np.hstack([a1, b1]), 0) | edges(np.hstack([b1.T, a1.T]), half + code.N)
        size = code.N + 2 * half

        pairs, remote = [], []
        for i in range(num_qpus):
            pairs += [(a + i * size, b + i * size) for a, b in couplers]
            if noisy_c_couplers:
                remote += [(a + i * size, b + i * size) for a, b in c_couplers]
            if i > 0:
                links = [(q + (i - 1) * size, q + i * size) for q in self.LINKS]
                pairs += links
                remote += links
        super().__init__(num_qpus * size, pairs, remote)
        self.qpu_size = size
