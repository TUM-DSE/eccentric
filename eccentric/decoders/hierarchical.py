"""Level-by-level decoding of concatenated codes (hier_bposd, hier_bposd_hard).

Concatenated codes (many-hypercube mhc36 and mhc216, concatenated Steane at d = 9 and 27) have checks at
several levels: a level-1 check acts within one inner block, a level-l check within one block of the
level-l code. Poulin (Phys. Rev. A 74, 052333, 2006) decodes such codes level by level, passing soft
information (probabilities) instead of hard decisions from each level to the next, which is exact on the
code's tree-shaped factor graph and raises the threshold of the concatenated five-qubit code from 0.1376
to at least 0.1885 under depolarizing noise. Goto's many-hypercube codes (Science Advances, 2024)
rely on a level-by-level soft decoder in the same spirit.

Here the same schedule runs on the circuit's detector error model, where the variables are the circuit's
error mechanisms and each detector inherits the level of the check it compares across rounds:

1. BP on the level-1 detectors only, from the channel probabilities;
2. BP on the level-2 detectors only, from the posteriors of step 1 (the soft information), and so on;
3. BP-OSD on all detectors from the last posteriors, so that the correction explains the whole syndrome.

hier_bposd_hard passes hard decisions instead (an error mechanism is taken as having happened, with
probability 1 - eps, or not, with eps), to separate the value of soft information from the schedule.
This is a heuristic for circuit-level noise: on a detector error model the factor graph is not a tree,
and the posteriors of single mechanisms stand in for the inner blocks' logical-class probabilities.
"""
import numpy as np
import stim
from ldpc import BpDecoder, BpOsdDecoder

from .bp_osd import dem_to_check_matrices

STAGE_BP = dict(bp_method="product_sum", max_iter=50)
FINAL = dict(bp_method="minimum_sum", ms_scaling_factor=1.0, max_iter=100, osd_method="osd_cs", osd_order=7)
EPS = 1e-6
_last = {}


def _level(support, base):
    """The level of a check: the smallest l whose blocks of base**l consecutive qubits hold its support"""
    support = sorted(support)
    level = 1
    while support[0] // base ** level != support[-1] // base ** level:
        level += 1
    return level


def check_levels(code_name: str, distance=None):
    """The level of each check of one basis, in the order the memory circuits measure them, and the
    number of levels; None for a code without levels"""
    if code_name in ("mhc36", "mhc216"):
        from eccentric.codes import qldpc_codes
        code = qldpc_codes.many_hypercube_code(2 if code_name == "mhc36" else 3)
        hz = np.asarray(code.matrix_z if hasattr(code, "matrix_z") else code.get_matrix("Z"))
        hx = np.asarray(code.matrix_x if hasattr(code, "matrix_x") else code.get_matrix("X"))
        levels = [_level(np.flatnonzero(row), 6) for row in hz]
        # The X and Z checks differ, but check i has the same level in both, so the basis does not matter
        if levels != [_level(np.flatnonzero(row), 6) for row in hx]:
            raise ValueError("many-hypercube X and Z check levels differ; levels would depend on the basis")
    elif code_name == "steane" and distance in (9, 27):
        from eccentric.codes import concat_steane
        orderings = (concat_steane.ordering_concat_steane_code() if distance == 9
                     else concat_steane.ordering_extended_steane_code())
        levels = [_level(ordering, 7) for ordering in orderings]
    else:
        return None
    return np.array(levels), max(levels)


def detector_levels(circuit: stim.Circuit, code_name: str, distance=None):
    """The level of each detector of a memory circuit of the code"""
    found = check_levels(code_name, distance)
    if found is None:
        raise ValueError(f"{code_name} (d = {distance}) has no levels for hierarchical decoding")
    levels, _ = found
    n = len(levels)
    if code_name.startswith("mhc"):
        # qLDPC memory circuits: detector coordinates (round, 0, index of the check among the basis's checks)
        coords = circuit.get_detector_coordinates()
        if len(coords) != circuit.num_detectors or any(len(c) < 3 or not 0 <= int(c[2]) < n for c in coords.values()):
            raise ValueError("detectors without the check index in their coordinates (not a memory circuit?)")
        return np.array([levels[int(coords[d][2])] for d in range(circuit.num_detectors)])
    # Concatenated Steane: every round compares the X checks then the Z checks, in the order of the check
    # list; the first round's X checks compare with the initial MPPs; the final readout gives the Z checks
    if circuit.num_detectors % n or (circuit.num_detectors // n) % 2 != 1:
        raise ValueError(f"{circuit.num_detectors} detectors do not fit rounds of {2 * n} and a readout of {n}")
    return np.tile(levels, circuit.num_detectors // n)


class Hierarchical:
    def __init__(self, dem: stim.DetectorErrorModel, levels: np.ndarray, soft: bool = True, stages: bool = True,
                 final_iter: int = FINAL["max_iter"]):
        """stages False: the final BP-OSD alone (the same settings without the level-by-level schedule)"""
        self.chk, self.obs, self.priors = dem_to_check_matrices(dem)
        self.chk = self.chk.tocsr()
        self.soft = soft
        self.rows = [np.flatnonzero(levels == level) for level in range(1, int(levels.max()) + 1)] if stages else []
        self.stages = [BpDecoder(self.chk[rows].tocsc(), error_channel=list(self.priors), **STAGE_BP)
                       for rows in self.rows if len(rows)]
        self.rows = [rows for rows in self.rows if len(rows)]
        self.final = BpOsdDecoder(self.chk.tocsc(), error_channel=list(self.priors), **{**FINAL, "max_iter": final_iter})

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        probs = self.priors
        for rows, stage in zip(self.rows, self.stages):
            stage.update_channel_probs(probs)
            stage.decode(syndrome[rows])
            posterior = 1.0 / (1.0 + np.exp(np.clip(stage.log_prob_ratios, -700, 700)))
            probs = (np.clip(posterior, EPS, 1 - EPS) if self.soft
                     else np.where(posterior > 0.5, 1 - EPS, EPS))
        self.final.update_channel_probs(probs)
        return self.final.decode(syndrome)


def hierarchical(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool, code_name: str,
                 distance=None, soft: bool = True):
    key = (approximate_disjoint_errors, code_name, distance, soft)
    if _last.get("circuit") is not circuit or _last.get("key") != key:
        levels = detector_levels(circuit, code_name, distance)
        dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
        _last.clear()
        _last.update(circuit=circuit, key=key, value=Hierarchical(dem, levels, soft))
    decoder = _last["value"]
    dets, obs = circuit.compile_detector_sampler().sample(num_shots, separate_observables=True)
    errors = 0
    for shot in range(num_shots):
        e_hat = decoder.decode(dets[shot].astype(np.uint8))
        errors += bool(np.any((decoder.obs @ e_hat + obs[shot]) % 2))
    return errors / num_shots
