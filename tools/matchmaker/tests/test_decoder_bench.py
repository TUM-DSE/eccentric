import math
import sys

import numpy as np
import pytest
import stim

h5py = pytest.importorskip("h5py")
sparse = pytest.importorskip("scipy.sparse")

from qec_matchmaker.features import features_from_circuit, features_from_matrices  # noqa: E402
from qec_matchmaker.io.decoder_bench import features_from_trace, read_trace  # noqa: E402


def surface(p=0.003, d=3):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p)


def bench_matrices(dem):
    """decoder-bench's dem_to_check_matrices: mechanisms of equal symptoms merged, priors added"""
    columns, priors = {}, []
    for inst in dem.flattened():
        if inst.type == "error":
            targets = inst.targets_copy()
            key = (frozenset(t.val for t in targets if t.is_relative_detector_id()),
                   frozenset(t.val for t in targets if t.is_logical_observable_id()))
            if key not in columns:
                columns[key] = len(priors)
                priors.append(0.0)
            priors[columns[key]] += inst.args_copy()[0]
    H = np.zeros((dem.num_detectors, len(priors)), np.uint8)
    L = np.zeros((dem.num_observables, len(priors)), np.uint8)
    for (dets, obs), j in columns.items():
        H[list(dets), j] = 1
        L[list(obs), j] = 1
    return H, L, np.array(priors)


def write_trace(path, circuit, shots=2000, batches=2, with_circuit=True):
    """A trace laid out as decoder_bench/dataset_gen.py writes one with store_unique=True: the first
    batch creates resizable datasets, later batches append their new unique syndromes"""
    H, L, priors = bench_matrices(circuit.detector_error_model())
    m = H.shape[0]
    seen, kept_dets, kept_obs = set(), [], []
    for batch in range(batches):
        dets, obs = circuit.compile_detector_sampler(seed=batch + 42).sample(shots, separate_observables=True)
        syndromes, observables = [], []
        for det, ob in zip(dets, obs):
            key = det.tobytes()
            if key not in seen:
                seen.add(key)
                syndromes.append(det)
                observables.append(ob)
        if batch == 0:
            with h5py.File(path, "w") as file:
                file.create_dataset("syndromes", data=syndromes, dtype=np.bool_, compression="gzip",
                                    shape=(len(syndromes), m), maxshape=(None, m))
                file.create_dataset("observables", data=observables, dtype=np.bool_, compression="gzip",
                                    shape=(len(observables), len(observables[0])), maxshape=(None, len(observables[0])))
                file.create_dataset("check_matrix", data=H.astype(np.uint8), dtype=np.uint8, compression="gzip")
                file.create_dataset("obs_matrix", data=L.astype(np.uint8), dtype=np.uint8, compression="gzip")
                file.create_dataset("priors", data=priors, dtype=float, compression="gzip")
                if with_circuit:
                    file.create_dataset("circuit", data=str(circuit), dtype=h5py.string_dtype(encoding="utf-8"))
        else:
            with h5py.File(path, "a") as file:
                file["syndromes"].resize((len(file["syndromes"]) + len(syndromes), m))
                file["syndromes"][-len(syndromes):] = syndromes
                file["observables"].resize((len(file["observables"]) + len(observables), len(observables[0])))
                file["observables"][-len(observables):] = observables
        kept_dets += syndromes
        kept_obs += observables
    return H, L, priors, np.array(kept_dets), np.array(kept_obs)


def same(a, b):
    assert set(a) == set(b)
    for name in a:
        if isinstance(a[name], float) and math.isnan(a[name]):
            assert isinstance(b[name], float) and math.isnan(b[name]), name
        else:
            assert a[name] == pytest.approx(b[name]), name


def test_read_trace_round_trips(tmp_path):
    circuit = surface()
    path = tmp_path / "surface_d3.h5"
    H, L, priors, dets, obs = write_trace(path, circuit)
    trace = read_trace(path)
    assert trace["circuit"] == circuit
    assert sparse.issparse(trace["check_matrix"]) and trace["check_matrix"].format == "csc"
    assert np.array_equal(trace["check_matrix"].toarray(), H)
    assert np.array_equal(trace["observables_matrix"].toarray(), L)
    assert np.array_equal(trace["priors"], priors)
    assert trace["syndromes"].dtype == bool and np.array_equal(trace["syndromes"], dets)
    assert trace["observables"].dtype == bool and np.array_equal(trace["observables"], obs)
    assert trace["stored_shots"] == len(dets) and len(np.unique(dets, axis=0)) == len(dets)

    head = read_trace(path, max_shots=10)
    assert np.array_equal(head["syndromes"], dets[:10]) and np.array_equal(head["observables"], obs[:10])
    none = read_trace(path, max_shots=0)
    assert none["syndromes"].shape == (0, circuit.num_detectors) and none["observables"].shape == (0, 1)


def test_features_from_trace_with_circuit(tmp_path):
    circuit = surface()
    path = tmp_path / "with_circuit.h5"
    write_trace(path, circuit, shots=200, batches=1)
    same(features_from_trace(path), features_from_circuit(circuit))


def test_features_from_trace_without_circuit(tmp_path):
    circuit = surface()
    path = tmp_path / "without_circuit.h5"
    H, L, priors, _, _ = write_trace(path, circuit, shots=200, batches=1, with_circuit=False)
    assert read_trace(path)["circuit"] is None
    f = features_from_trace(path)
    same(f, features_from_matrices(H, priors, L))
    # Stim already merges equal mechanisms here, so the matrices give the circuit's error-model features
    full = features_from_circuit(circuit)
    for name, value in f.items():
        if not (isinstance(value, float) and math.isnan(value)):
            assert value == pytest.approx(full[name]), name


def test_missing_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match="decoder-bench"):
        read_trace(tmp_path / "absent.h5")
