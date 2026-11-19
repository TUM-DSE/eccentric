"""GARI: normalized min-sum decoding on the GARI-transformed correlated detector error model (Maan,
Garcia-Herrero, Paler, Savin, "Decoding correlated errors in quantum LDPC codes", arXiv:2510.14060), with the
authors' C decoder (https://github.com/astra-decoders/gari-nms, Apache License 2.0, vendored unchanged in
hbp_decoder_v2.c).

The detector error model of a CSS memory circuit is split into its X-type detectors (which detect Z errors) and
Z-type detectors (which detect X errors). Errors seen by only one type are single-type errors; an error seen by
both is a Y-type error, whose two parts are single-type errors. GARI adds a variable per single-type error for
its effective value (its own value plus the Y-type errors that contain it) and parity checks tying them together,
which removes the 4-cycles that Y-type errors cause, and decodes that matrix with serial normalized min-sum.

Here, as in the authors' code: detectors are classified by the errors on the data qubits (the qubits measured
once) that flip them (by stim's detecting regions, read in each data qubit's frame, which compiled circuits change
with Hadamards and move with SWAPs; the authors inject errors after reset and measurement layers); the observables are predicted from the effective errors on the side
that flips them; and a shot whose decoding does not converge gets no correction. Unlike the authors' code, the
encoding is built with sparse matrices and hashed columns (the same matrix), and a circuit without this
structure (a detector or an observable on both sides, or a Y-type error whose part is not a single-type error)
is refused with ValueError: GARI does not apply to it. Any labelling that passes these checks gives a valid
GARI matrix; the labels only decide which errors count as Y-type.

gari_pp: GARI++ (arXiv:2611.12833), a BETA REIMPLEMENTATION (the authors have not released it): GARI for at most
7 iterations with the paper's normalization factor per code family; on shots it does not converge on, the
iteration with the lightest residual syndrome is kept and its residual is decoded by ldpc's union-find guided by
GARI's posterior LLRs of that iteration. Differences from the paper: union-find (matrix-solve mode on
hypergraphs) is the second stage for every code (the paper uses Chromobius for color codes and a hypergraph
union-find of its own for BB codes), and the iteration is chosen by rerunning the seeded decoder.
"""
from .gari import gari, gari_encoding, gari_pp

__all__ = ["gari", "gari_encoding", "gari_pp"]
