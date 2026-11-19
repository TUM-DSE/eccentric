from .noise import *
from ..backends import FakeIBMFlamingo, QubitTracking


class FlamingoNoise(NoiseModel):

    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend: FakeIBMFlamingo,
        measure_time: float = 1.56e-6
    ) -> 'NoiseModel':
        return NoiseModel(
            sq=0.00025,
            tq=0.002,
            measure=0.01,
            remote=0.03,
            gate_times={
                "SQ": 50 * 1e-9,
                "TQ": 70 * 1e-9,
                # Readout as long as on IBM Heron today (1.56 us in its calibration data)
                "M": measure_time,
                "REMOTE": round((300 * 1e-9) / (2.2222222222222221e-10 * 1e9)) * (2.2222222222222221e-10 * 1e9),
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )
