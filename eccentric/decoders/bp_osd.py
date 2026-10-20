import stim
import numpy as np
from ldpc import BpOsdDecoder
from scipy.sparse import csc_matrix
from typing import Dict, List, FrozenSet

# Functions adapted from https://github.com/gongaa/SlidingWindowDecoder


def dict_to_csc_matrix(elements_dict, shape):
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


def dem_to_check_matrices(dem: stim.DetectorErrorModel, return_col_dict=False, xor_priors=False):
    DL_ids: Dict[str, int] = {}
    L_map: Dict[int, FrozenSet[int]] = {}
    priors_dict: Dict[int, float] = {}

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        dets = frozenset(detectors)
        obs = frozenset(observables)
        key = " ".join([f"D{s}" for s in sorted(dets)] + [f"L{s}" for s in sorted(obs)])

        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0

        hid = DL_ids[key]
        L_map[hid] = obs
        if xor_priors:
            priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        else:
            priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[int] = []
            frames: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    check_matrix = dict_to_csc_matrix(
        {
            v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")]
            for k, v in DL_ids.items()
        },
        shape=(dem.num_detectors, len(DL_ids)),
    )
    observables_matrix = dict_to_csc_matrix(
        L_map, shape=(dem.num_observables, len(DL_ids))
    )
    priors = np.zeros(len(DL_ids))
    for i, p in priors_dict.items():
        priors[i] = p

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors


def bposd_chk(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool):
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)

    bpd = BpOsdDecoder(
        chk,
        error_channel=list(priors),
        max_iter=10000,
        bp_method="minimum_sum",
        ms_scaling_factor=1.0,
        osd_method="osd_cs",
        osd_order=7,
    )

    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    det_data, obs_data, err_data = dem_sampler.sample(
        shots=num_shots, return_errors=False, bit_packed=False
    )

    num_err = 0
    num_flag_err = 0
    for i in range(num_shots):
        e_hat = bpd.decode(det_data[i])
        num_flag_err += ((chk @ e_hat + det_data[i]) % 2).any()
        ans = (obs @ e_hat + obs_data[i]) % 2
        num_err += ans.any()
    p_l = num_err / num_shots
    return p_l


def bposd_batch(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool):
    # Same decoder setup as stimbposd.BPOSD, which is incompatible with ldpc v2
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    chk, obs, priors = dem_to_check_matrices(dem, xor_priors=True)
    bpd = BpOsdDecoder(
        chk,
        error_channel=list(priors),
        max_iter=10000,
        bp_method="minimum_sum",
        ms_scaling_factor=1.0,
        osd_method="osd_cs",
        osd_order=min(7, chk.shape[1] - chk.shape[0]),
    )
    num_errors = 0
    for shot in range(num_shots):
        predicted_for_shot = (obs @ bpd.decode(detection_events[shot])) % 2
        if not np.array_equal(observable_flips[shot], predicted_for_shot):
            num_errors += 1
    return num_errors / num_shots
