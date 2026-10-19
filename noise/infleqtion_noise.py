from .noise import *
from backends import QubitTracking, FakeInfleqtionBackend

class InfleqtionNoise(NoiseModel):

    @staticmethod
    def get_noise(qt: QubitTracking, backend: FakeInfleqtionBackend) -> 'NoiseModel':
        return NoiseModel(
            # https://arxiv.org/pdf/2408.08288
            sq=0.00098,
            tq=0.0065,
            crosstalk=0,
            leakage=0,
            measure=0.004,
            shuttle=0.001999,
            gate_times={
                "SQ": 2.5e-7,
                "TQ": 4.16e-7,
                "M": 6.0e-3,
                "R": 1.22e-2,
                "REMOTE": 2.182e-4,
            },
            qt=qt,
            backend=backend
        )
