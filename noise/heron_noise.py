from .noise import *
from backends import QubitTracking


class HeronNoise(NoiseModel):
    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend,
        m_error_multiplier = 1,
        m_time_multiplier = 1,
        idle_multiplier: float = 1.0
    ) -> 'NoiseModel':
        m_error_multiplier = float(m_error_multiplier)
        m_time_multiplier = float(m_time_multiplier)
        return NoiseModel(
            sq=0.00025,
            tq=0.002,
            measure=0.001 * m_error_multiplier,
            #sq=0.00025 / 10,
            #tq=0.002 / 10,
            #measure=0.01 * m_error_multiplier / 5,
            gate_times={
                "SQ": 50 * 1e-9,
                "TQ": 70 * 1e-9,
                "M": 2000 * 1e-9 * m_time_multiplier, 
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend,
            idle_multiplier=idle_multiplier
        )
