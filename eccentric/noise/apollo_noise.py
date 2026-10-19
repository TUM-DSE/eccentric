from .noise import *
from ..backends import QubitTracking


class ApolloNoise(NoiseModel):
    # H2 rescaled according to roadmap: https://www.quantinuum.com/press-releases/quantinuum-unveils-accelerated-roadmap-to-achieve-universal-fault-tolerant-quantum-computing-by-2030
    @staticmethod
    def get_noise(qt: QubitTracking) -> 'NoiseModel':
        return NoiseModel(
            sq=8.0e-5 / 10,
            tq=1.4e-3 / 10,
            idle=5.3e-4 / 10,
            crosstalk=6.3e-6 / 10,
            measure=1.33e-3 / 10,
            leakage=4.3e-4 / 10,
            shuttle=6.3e-06 / 10,
            gate_times={
                # As suggested by Quantinuum FAQ: https://arxiv.org/pdf/2003.01293
                "SQ": 5 * 1e-6,
                "TQ": 25 * 1e-6,
                "M": 60 * 1e-6,
                "R": 10 * 1e-6,
                "REMOTE": 0,
            },
            qt=qt
        )
