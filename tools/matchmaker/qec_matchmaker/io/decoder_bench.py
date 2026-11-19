"""Reading decoder-bench traces (https://github.com/satvikmaurya/decoder-bench).

A trace is one HDF5 file as decoder_bench/dataset_gen.py writes it, with no attributes and the datasets
`syndromes` (shots x detectors, bool), `observables` (shots x observables, bool), `check_matrix`
(detectors x mechanisms, dense uint8), `obs_matrix` (observables x mechanisms, dense uint8), `priors`
(per mechanism, float) and `circuit` (the noisy Stim circuit as a UTF-8 string).

By default decoder-bench keeps only the first occurrence of each syndrome, so common syndromes (the
empty one above all) are underrepresented: a logical error rate over a trace's syndromes is not the
circuit's, and nothing here computes one.

Its matrices come from the circuit's DEM without `approximate_disjoint_errors`, with mechanisms of equal
symptoms merged by adding their probabilities, so features from them can differ from the circuit's;
`features_from_trace` uses the circuit when the trace has it.
"""
import numpy as np
import stim

from ..features import features_from_circuit, features_from_matrices


def _modules():
    try:
        import h5py
        from scipy import sparse
    except ImportError as e:
        raise ImportError("reading decoder-bench traces needs h5py and scipy: "
                          "pip install 'qec-matchmaker[decoder-bench]'") from e
    return h5py, sparse


def read_trace(path, max_shots=None) -> dict:
    """Circuit (None if absent), check and observables matrices (scipy csc), priors, and the first
    `max_shots` (all if None) syndromes and observable flips of a trace; `stored_shots` is how many it has"""
    h5py, sparse = _modules()
    with h5py.File(path, "r") as f:
        circuit = None
        if "circuit" in f:
            text = f["circuit"][()]
            circuit = stim.Circuit(text.decode("utf-8") if isinstance(text, bytes) else str(text))
        check_matrix = sparse.csc_matrix(f["check_matrix"][()].astype(np.uint8))
        observables_matrix = sparse.csc_matrix(f["obs_matrix"][()].astype(np.uint8)) if "obs_matrix" in f else None
        priors = np.asarray(f["priors"][()], float).ravel()
        shots = slice(max_shots)
        return {"circuit": circuit, "check_matrix": check_matrix, "priors": priors,
                "observables_matrix": observables_matrix, "syndromes": f["syndromes"][shots].astype(bool),
                "observables": f["observables"][shots].astype(bool), "stored_shots": len(f["syndromes"])}


def features_from_trace(path, check_decomposition: bool = True) -> dict:
    """Features of a trace: from its circuit if it has one, else from its matrices and priors"""
    trace = read_trace(path, max_shots=0)
    if trace["circuit"] is not None:
        return features_from_circuit(trace["circuit"], check_decomposition)
    return features_from_matrices(trace["check_matrix"], trace["priors"], trace["observables_matrix"])
