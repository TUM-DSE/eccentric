from .noise import *
from backends import QubitTracking


class SycamoreNoise(NoiseModel):

    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend,
        m_error_multiplier = 1,
        m_time_multiplier = 1,
        decoding_time = 0
    ) -> 'NoiseModel':
        m_error_multiplier = float(m_error_multiplier)
        m_time_multiplier = float(m_time_multiplier)
        decoding_time = float(decoding_time)
        return NoiseModel(
            sq=0.0016,
            tq=0.0062,
            measure=0.038 * m_error_multiplier,
            gate_times={
                "SQ": 35 * 1e-9,
                "TQ": 42 * 1e-9,
                "M": 660 * 1e-9 * m_time_multiplier + decoding_time * 1e-6,
               # "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )
