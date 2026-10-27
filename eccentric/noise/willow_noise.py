from .noise import *

class WillowNoise(NoiseModel):

    @staticmethod
    def get_noise(qt) -> 'NoiseModel':
        # From https://arxiv.org/pdf/2408.13687; its idle error is the error of the data qubits while
        # the measure qubits are measured and reset, once per cycle
        return NoiseModel(
            sq=6.2e-4,
            tq=2.8e-3,
            idle=0.9e-2,
            crosstalk=5.5e-4,
            leakage=2.5e-4,
            leakage_propagation=2.0e-4,
            reset=1.5e-3,
            measure=0.8e-2,
            idle_window="measurement",
            qt=qt
        )

    @staticmethod
    def get_spec_sheet_noise(qt) -> 'NoiseModel':
        # Gate and measurement errors of the QEC chip (Chip 1) from the Willow spec sheet (Dec 2024):
        # https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf
        # The spec sheet does not report idle, crosstalk, leakage or reset errors, so these are the
        # values of get_noise
        return NoiseModel(
            sq=3.5e-4,
            tq=3.3e-3,
            idle=0.9e-2,
            crosstalk=5.5e-4,
            leakage=2.5e-4,
            leakage_propagation=2.0e-4,
            reset=1.5e-3,
            measure=7.7e-3,
            idle_window="measurement",
            qt=qt
        )
