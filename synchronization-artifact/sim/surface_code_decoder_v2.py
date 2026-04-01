# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import networkx as nx
import stim
import pymatching
import math
from typing import Callable, List, Iterable, Dict
from itertools import product
from more_itertools import powerset

### The following three functions below have been copied from the built-in method of stim: stim.Circuit.generated("surface_code:rotated_memory_z",...)
def iter_flatten_model(model: stim.DetectorErrorModel,
                    handle_error: Callable[[float, List[int], List[int]], None],
                    handle_detector_coords: Callable[[int, np.ndarray], None]):
    det_offset = 0
    coords_offset = np.zeros(100, dtype=np.float64)

    def _helper(m: stim.DetectorErrorModel, reps: int):
        nonlocal det_offset
        nonlocal coords_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _helper(instruction.body_copy(), instruction.repeat_count)
                elif isinstance(instruction, stim.DemInstruction):
                    if instruction.type == "error":
                        dets: List[int] = []
                        frames: List[int] = []
                        t: stim.DemTarget
                        p = instruction.args_copy()[0]
                        for t in instruction.targets_copy():
                            if t.is_relative_detector_id():
                                dets.append(t.val + det_offset)
                            elif t.is_logical_observable_id():
                                frames.append(t.val)
                            elif t.is_separator():
                                # Treat each component of a decomposed error as an independent error.
                                # (Ideally we could configure some sort of correlated analysis; oh well.)
                                handle_error(p, dets, frames)
                                frames = []
                                dets = []
                        # Handle last component.
                        handle_error(p, dets, frames)
                    elif instruction.type == "shift_detectors":
                        det_offset += instruction.targets_copy()[0]
                        a = np.array(instruction.args_copy())
                        coords_offset[:len(a)] += a
                    elif instruction.type == "detector":
                        a = np.array(instruction.args_copy())
                        for t in instruction.targets_copy():
                            handle_detector_coords(t.val + det_offset, a + coords_offset[:len(a)])
                    elif instruction.type == "logical_observable":
                        pass
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
    _helper(model, 1)


def detector_error_model_to_nx_graph(model: stim.DetectorErrorModel, relevant_nodes = []) -> 'nx.Graph':
    """Convert a stim error model into a NetworkX graph."""

    g = nx.Graph()
    num_obs = model.num_observables
    num_det = model.num_detectors

    log_nodes = []
    triv_nodes = []
    for i in range(num_obs):
        log_nodes.append(num_det+i*2)
        triv_nodes.append(num_det+i*2+1)
        g.add_node(log_nodes[-1], is_boundary=False, coords=[-1, -1, -1])
        g.add_node(triv_nodes[-1], is_boundary=False, coords=[-1, -1, -1])
    irrel_node = num_det+num_obs*2
    g.add_node(irrel_node, is_boundary=False, coords=[-1, -1, -1])


    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return
        if len(dets) == 0:
            # No symptoms for this error.
            # Code probably has distance 1.
            # Accept it and keep going, though of course decoding will probably perform terribly.
            return
        if len(dets) == 1:
            if set(frame_changes) == set():
                for i in range(num_obs):
                    if dets[0] in relevant_nodes[i]:
                        dets = [dets[0], triv_nodes[i]]
                if len(dets)==1:
                    # none of the above was true
                    dets = [dets[0], irrel_node]
            else:
                for frame_ind in frame_changes:
                    handle_error(p, [dets[0], log_nodes[frame_ind]], [frame_ind])
                return
        if len(dets) > 2:
            print("Warning: hyperedge with detectors "+str(dets)+" is ignored, logical change: "+str(bool(frame_changes)))
            return
            # raise NotImplementedError(
            #     f"Error with more than 2 symptoms can't become an edge or boundary edge: {dets!r}.")
        if g.has_edge(*dets):
            edge_data = g.get_edge_data(*dets)
            old_p = edge_data["error_probability"]
            old_frame_changes = edge_data["fault_ids"]
            # If frame changes differ, the code has distance 2; just keep whichever was first.
            if set(old_frame_changes) == set(frame_changes):
                p = p * (1 - old_p) + old_p * (1 - p)
                g.remove_edge(*dets)
        if p > 0.5:
            p = 1 - p
        if p > 0:
            g.add_edge(*dets, weight=math.log((1 - p) / p), fault_ids=frame_changes, error_probability=p)

    def handle_detector_coords(detector: int, coords: np.ndarray):
        g.add_node(detector, coords=coords)

    iter_flatten_model(model, handle_error=handle_error, handle_detector_coords=handle_detector_coords)

    return g, log_nodes, triv_nodes, irrel_node

def reweight_graph_with_data(g, weights_data):
    freqs,detector_samples = weights_data
    for edge in g.edges():
        dets = edge[:2]
        edge_data = g.get_edge_data(*dets)
        frame_changes = edge_data["fault_ids"]
        node_flips = detector_samples[:,tuple(dets)]
        ones = sum([freq for freq,(a,b) in zip(freqs,node_flips) if (a,b)==(1,1)])
        zeros = sum([freq for freq,(a,b) in zip(freqs,node_flips) if (a,b)==(0,0)])
        p = ones/(zeros+ones)
        g.remove_edge(*dets)
        g.add_edge(*dets, weight=math.log((1 - p) / p), fault_ids=frame_changes, error_probability=p)
    return g


def detector_error_model_to_pymatching_graph(model: stim.DetectorErrorModel, weights_data = [], relevant_nodes = []) -> 'pymatching.Matching':
    """Convert a stim error model into a pymatching graph."""

    g, log_nodes, triv_nodes, irrel_node = detector_error_model_to_nx_graph(model, relevant_nodes)

    if weights_data!=[]:
        g = reweight_graph_with_data(g,weights_data=weights_data)


    num_detectors = model.num_detectors
    num_observables = model.num_observables

    # Ensure invincible detectors and logicals are seen by explicitly adding a node for each detector.
    for k in range(num_detectors + 2*num_observables + 1):
        g.add_node(k)

    return pymatching.Matching(g), log_nodes, triv_nodes, irrel_node

class DEM_to_matching:
    def __init__(self, model: stim.DetectorErrorModel, weights_data = [], relevant_nodes = []):
        self.num_obs = model.num_observables
        self.matching,log_nodes,triv_nodes,irrel_node = detector_error_model_to_pymatching_graph(model = model, weights_data = weights_data, relevant_nodes = relevant_nodes)
        self.matching.set_boundary_nodes(set.union(set(log_nodes),set(triv_nodes),set([irrel_node])))
    
    def decode(self,syndrome):
        """Returns an array of the corrected observables (not the flips!)"""
        actual_observables = syndrome[-self.num_obs:]
        ##we don't give the logicals to matching since they are boundary nodes anyway
        predicted_flip = self.matching.decode(syndrome[:-self.num_obs])
        return (predicted_flip+actual_observables)%2

    def decode_batch(self,syndrome_batch):
        """Returns an array of the corrected observables (not the flips!)"""
        actual_obs_batch = syndrome_batch[:,-self.num_obs:]
        ##we don't give the logicals to matching since they are boundary nodes anyway
        predicted_flip_batch = self.matching.decode_batch(syndrome_batch[:,:-self.num_obs])

        correction_batch = (predicted_flip_batch != actual_obs_batch)

        return correction_batch

class DEM_to_postsel_matching:
    def __init__(self, model: stim.DetectorErrorModel, weights_data = [], relevant_nodes = []):
        init_matching,log_nodes,triv_nodes,irrel_node = detector_error_model_to_pymatching_graph(model = model, weights_data = weights_data, relevant_nodes = relevant_nodes)
        
        self.num_obs = model.num_observables

        log_subsets = [[tup for tup in powerset([log,triv]) if tup!=()] for log,triv in zip(log_nodes,triv_nodes)]
        boundary_sets = [set.union(*[set(log_subset[inds[log_ind]]+(irrel_node,))
                                        for log_ind, log_subset in enumerate(log_subsets)])
                            for inds in list(product([0,1,2], repeat=self.num_obs))]

        self.matchings = []
        for boundary in boundary_sets:
            matching= pymatching.Matching()
            self.logflip_edges = [[] for _ in range(self.num_obs)]
            for new_fault_id,edge in enumerate(init_matching.edges()):
                dets = edge[:2]
                edge_data = edge[2]
                if edge_data['fault_ids']!=set():
                    for logical in edge_data['fault_ids']:
                        self.logflip_edges[logical].append(new_fault_id)
                matching.add_edge(*dets, weight=edge_data["weight"], fault_ids={new_fault_id}, error_probability=edge_data["error_probability"],merge_strategy="replace")
            matching.set_boundary_nodes(boundary)
            self.matchings.append(matching)

            self.logflip_edges_mat = np.array([sum(np.identity(init_matching.num_edges)[logflip_edge_inds]) for logflip_edge_inds in self.logflip_edges]).transpose()
    
    def decode(self,syndrome):
        """Returns an array of the corrected observables (not the flips!)"""
        syndrome = (1*syndrome).tolist()
        log_obs = syndrome[-self.num_obs:]
        log_flips = []
        len_match = []
        for matching in self.matchings:
            ##we do give the logicals to matching since they are sometimes ordinary nodes
            min_match = matching.decode(syndrome[:-self.num_obs] + [0]*(2*self.num_obs+1))
            len_match.append(sum(min_match))
            log_flips.append([bool(sum([in_matching for edge_ind,in_matching in enumerate(min_match)
                                        if edge_ind in logflip_edge]
                                        )%2)
                                for logflip_edge in self.logflip_edges])
        min_len_logflips = [flip_list for len,flip_list in zip(len_match,log_flips) if len == min(len_match)]
        if min_len_logflips == len(min_len_logflips)*[min_len_logflips[0]]:
            return np.array([pred!=meas for meas,pred in zip(min_len_logflips[0],log_obs)])
        else:
            return 'ambig'


    def decode_batch(self,syndrome_batch):
        """Returns a batch of arrays with the corrected observables (not the flips!)"""
        log_obs_batch = syndrome_batch[:,-self.num_obs:]
        syndrome_batch = [(1*syndrome[:-self.num_obs]).tolist() + [0]*(2*self.num_obs+1) for syndrome in syndrome_batch]
        correction_batch = []
        len_match_batch = []
        for matching in self.matchings:
            min_match_batch = matching.decode_batch(syndrome_batch)
            len_match_batch.append(min_match_batch@np.array([1]*len(min_match_batch[0])))
            correction_batch.append(((min_match_batch@self.logflip_edges_mat)%2 != log_obs_batch).transpose())
        len_match_batch = np.array(len_match_batch).transpose()
        correction_batch = np.array(correction_batch).transpose()
        unambig_correction_batch = []
        for corr_list,len_list in zip(correction_batch,len_match_batch):
            corr_min_len = [corr.tolist() for corr,len in zip(corr_list.transpose(),len_list) if len==min(len_list)]
            if corr_min_len.count(corr_min_len[0])==len(corr_min_len):
                unambig_correction_batch.append(corr_min_len[0])
        return np.array(unambig_correction_batch)