from .noise import *
from ..backends import QubitTracking
from ..backends.fake_harvard import FakeHarvardBackend


class HarvardNoise(NoiseModel):
    # Bluvstein et al., Nature 649, 39-46 (2025), https://arxiv.org/abs/2506.20661
    @staticmethod
    def get_noise(qt: QubitTracking, backend: FakeHarvardBackend, loss_detection: str = "measurement") -> 'NoiseModel':
        return NoiseModel(
            # Local single-qubit gates are ~99.9% fidelity, CZ gates ~99.5% (Methods)
            sq=1e-3,
            tq=5e-3,
            crosstalk=0,
            leakage=0,
            # Bit-flip error of the non-destructive readout (Fig. 1b)
            measure=4.6e-3,
            # Gates between distant atoms need a move, with the same CZ error
            shuttle=5e-3,
            # Atom loss: ~1% per ancilla over the ~4 moves of a surface-code round, and 0.24% per
            # readout (Fig. 1b). The array detects lost atoms when it images them.
            loss_remote=2.5e-3,
            loss_measure=2.4e-3,
            loss_detection=loss_detection,
            gate_times={
                "SQ": 5e-6,        # global Raman rotations
                "TQ": 270e-9,      # Rydberg CZ
                "M": 10e-3,        # imaging
                "R": 1.1e-3,       # re-initialization
                "REMOTE": 655e-6,  # average transversal CNOT, including the move
            },
            qt=qt,
            backend=backend
        )
