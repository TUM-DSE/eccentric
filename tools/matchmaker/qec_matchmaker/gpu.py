"""Decoding on a GPU: ECCentric's GPU decoders (eccentric/decoders/gpu) and a model of their time per shot.

The six decoders below have GPU versions that give the CPU decoder's corrections, exactly or within statistical
error (ECCentric's analysis/GPU.md), so Matchmaker's accuracy predictions hold for both and only the time
differs. GPU_ONLY decoders have no CPU version: their accuracy is trained on their own slice decodes. The time
model is trained on GPU pilots: up to 10^5 shots of 306 circuits of ECCentric's sweep per
decoder on one NVIDIA H200 NVL (experiments/gpu), stratified by code family. Its times are steady-state ms per
shot of that GPU, CPU post-processing included; building the decoder takes a further `build_seconds`.
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .models import TimeModel, _cqr

GPU_VERSIONS = {"bposd": "bposd_gpu", "bposd_tuned": "bposd_tuned_gpu", "bplsd": "bplsd_gpu",
                "belief_find": "belief_find_gpu", "relay_bp": "relay_bp_gpu", "gari": "gari_gpu"}
# Decoders that exist on the GPU only: their accuracy is learned from their own decodes of the GPU slice
# (experiments/gpu_tcbp: same settings and shots as the CPU decoders' jobs), joined to the training table; they
# are predicted only with gpu=True
GPU_ONLY = {"tcbp_jobs_sc_gpu": "TCBP-JOBS (arXiv:2611.21321, preprint) with JOBS on every shot: CUDA-Q QEC "
                                "nv-qldpc's tcbp-jobs-sc preset, the code family's TCBP model"}
GPU_ONLY_RESULTS = {"tcbp_jobs_sc_gpu": "results_tcbp_jobs_sc_gpu"}
Z95 = 1.959963984540054
DEVICE = "NVIDIA H200 NVL"
COVERAGE = 0.8  # of the [q10, q90] interval, after calibration


def load_pilots(results_dir: str) -> pd.DataFrame:
    """The measured pilots of experiments/gpu/results/*.jsonl: one row per circuit and decoder"""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "pilot" and r.get("status") == "measured" and r.get("gpu_shots_per_second"):
                    rows.append({"circuit_key": r["key"], "decoder": r["decoder"],
                                 "gpu_ms_per_shot": 1000.0 / r["gpu_shots_per_second"],
                                 "build_seconds": r["build_seconds"], "shots": r["shots"],
                                 "cpu_ms_per_shot": r["cpu_ms_per_shot"]})
    return pd.DataFrame(rows)


def _wilson(errors, shots):
    """ECCentric's 95% Wilson interval (sweep/collect.py)"""
    p = errors / shots
    denom = 1 + Z95 ** 2 / shots
    centre = (p + Z95 ** 2 / (2 * shots)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / shots + Z95 ** 2 / (4 * shots ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _gpu_only_records(gpu_only_dir: str):
    for decoder, sub in GPU_ONLY_RESULTS.items():
        for path in sorted(glob.glob(os.path.join(gpu_only_dir, sub, "*.json"))):
            with open(path) as f:
                yield decoder, json.load(f)


def load_gpu_only_rows(gpu_only_dir: str, table: pd.DataFrame) -> pd.DataFrame:
    """Training rows of the GPU-only decoders (experiments/gpu_tcbp/results_<decoder>/<i>.json), one per circuit,
    with the circuit's columns from `table` and ECCentric's status rules (sweep/collect.py): measured at the target
    errors, low statistics below, an upper bound without errors. ms_per_shot stays empty: their time is the GPU
    time model's (load_pilots)"""
    from .dataset import KEYS, LABELS
    circuits = table.drop_duplicates("circuit_key").set_index("circuit_key")
    rows = []
    for decoder, r in _gpu_only_records(gpu_only_dir):
        ler = r.get("ler") or {}
        if r["key"] not in circuits.index or "errors" not in ler or not ler.get("shots"):
            continue
        shots, errors = int(ler["shots"]), int(ler["errors"])
        low, high = _wilson(errors, shots)
        status = ("upper bound" if errors == 0 else
                  "measured" if errors >= r["settings"]["target_errors"] else "low statistics")
        row = circuits.loc[r["key"]].to_dict()
        row.update(circuit_key=r["key"], decoder=decoder, status=status, shots=shots, errors=errors,
                   ler=errors / shots if errors else math.nan, ler_low=low if errors else math.nan, ler_high=high,
                   ms_per_shot=math.nan, timeout_s=math.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(table.columns) + [c for c in KEYS + LABELS if c not in table.columns])


def load_gpu_only_pilots(gpu_only_dir: str) -> pd.DataFrame:
    """GPU times of the GPU-only decoders, measured as the pilots (experiments/gpu_tcbp, the `timing` part)"""
    rows = []
    for decoder, r in _gpu_only_records(gpu_only_dir):
        t = r.get("timing") or {}
        if t.get("gpu_shots_per_second"):
            rows.append({"circuit_key": r["key"], "decoder": decoder, "gpu_ms_per_shot": 1000.0 / t["gpu_shots_per_second"],
                         "build_seconds": t["build_seconds"], "shots": t["shots"], "cpu_ms_per_shot": math.nan})
    return pd.DataFrame(rows)


def _frame(pilots: pd.DataFrame, store_features: dict, features: list) -> pd.DataFrame:
    """The pilots joined with their circuits' features (pilots without features are dropped)"""
    known = pilots[pilots.circuit_key.isin(store_features)]
    feats = pd.DataFrame([store_features[k] for k in known.circuit_key]).reindex(columns=features)
    return pd.concat([known.reset_index(drop=True), feats.apply(pd.to_numeric, errors="coerce")], axis=1)


def _fit(frame, features, quantiles, parametric, random_state=0):
    y = np.log10(frame.gpu_ms_per_shot.to_numpy(float))
    return TimeModel(quantiles, random_state, parametric).fit(frame, frame.decoder, features, y,
                                                              sorted(GPU_VERSIONS) + sorted(GPU_ONLY))


def evaluate(frame: pd.DataFrame, features: list, quantiles=(0.1, 0.5, 0.9), folds: int = 10,
             groups: str = "code") -> dict:
    """Out-of-fold predictions over held-out groups (codes): accuracy of the median, interval coverage before and
    after calibration, for the time law + trees and for trees alone; and the widening that calibrates them"""
    q = list(quantiles)
    out, oof = {}, {}
    y = np.log10(frame.gpu_ms_per_shot.to_numpy(float))
    for parametric in (True, False):
        pred = np.full((len(frame), len(q)), np.nan)
        for train, test in GroupKFold(min(folds, frame[groups].nunique())).split(frame, groups=frame[groups]):
            model = _fit(frame.iloc[train], features, q, parametric)
            pred[test] = model.predict(frame.iloc[test], frame.decoder.iloc[test])
        name = "law+trees" if parametric else "trees"
        oof[name] = pred
        err = np.abs(pred[:, q.index(0.5)] - y)
        lo, hi = pred[:, 0], pred[:, -1]
        out[name] = {"median_abs_log10": float(np.median(err)), "p90_abs_log10": float(np.quantile(err, 0.9)),
                     "within_3x": float(np.mean(err <= math.log10(3))),
                     "coverage": float(np.mean((lo <= y) & (y <= hi)))}
    best = min(out, key=lambda n: out[n]["median_abs_log10"])
    widen = _cqr(oof[best][:, 0], oof[best][:, -1], y, COVERAGE)
    per_decoder = {}
    for d in sorted(frame.decoder.unique()):
        mine = (frame.decoder == d).to_numpy()
        err = np.abs(oof[best][mine, q.index(0.5)] - y[mine])
        per_decoder[d] = {"pilots": int(mine.sum()), "within_3x": float(np.mean(err <= math.log10(3))),
                          "median_abs_log10": float(np.median(err))}
    lo, hi = oof[best][:, 0] - widen, oof[best][:, -1] + widen
    # Reference: the circuit's measured CPU time divided by its decoder's median speedup (needs a CPU decode, so
    # not for the GPU-only decoders)
    speedup = np.log10(frame.cpu_ms_per_shot.to_numpy(float)) - y
    med = frame.assign(s=speedup).groupby("decoder").s.transform("median").to_numpy()
    ref = np.abs(np.log10(frame.cpu_ms_per_shot.to_numpy(float)) - med - y)
    ref = ref[np.isfinite(ref)]
    return {"held_out": groups, "folds": min(folds, frame[groups].nunique()), "pilots": len(frame),
            "circuits": int(frame.circuit_key.nunique()), "variants": out, "chosen": best, "widen_log10": widen,
            "coverage_calibrated": float(np.mean((lo <= y) & (y <= hi))), "per_decoder": per_decoder,
            "reference_measured_cpu_over_median_speedup": {
                "median_abs_log10": float(np.median(ref)), "within_3x": float(np.mean(ref <= math.log10(3)))}}


def add_gpu_model(bundle, pilots: pd.DataFrame, store_features: dict, codes: dict, folds: int = 10) -> dict:
    """Evaluate on held-out codes, then fit on all pilots and add the model to `bundle` (models["gpu_time"],
    meta["gpu"]). codes: {circuit_key: code} for the held-out grouping. Returns the evaluation"""
    frame = _frame(pilots, store_features, bundle.features)
    frame["code"] = frame.circuit_key.map(codes)
    frame = frame.dropna(subset=["code"]).reset_index(drop=True)
    report = evaluate(frame, bundle.features, bundle.quantiles, folds)
    model = _fit(frame, bundle.features, bundle.quantiles, report["chosen"] == "law+trees")
    bundle.models["gpu_time"] = model
    bundle.meta["gpu"] = {
        "device": DEVICE, "versions": GPU_VERSIONS, "gpu_only": sorted(frame.decoder[frame.decoder.isin(GPU_ONLY)].unique()),
        "widen_log10": report["widen_log10"],
        "build_seconds": frame.groupby("decoder").build_seconds.median().round(3).to_dict(),
        "evaluation": report,
    }
    return report


def gpu_ms(bundle, frame: pd.DataFrame, decoders) -> np.ndarray:
    """rows x quantiles of GPU ms per shot (NaN for decoders without a GPU version or a GPU model)"""
    decoders = np.asarray(decoders, dtype=object)
    out = np.full((len(frame), len(bundle.quantiles)), np.nan)
    model = bundle.models.get("gpu_time")
    has = np.isin(decoders, list(GPU_VERSIONS) + list(GPU_ONLY))
    if model is None or not has.any():
        return out
    pred = model.predict(frame[has].reset_index(drop=True), decoders[has])
    widen = bundle.meta["gpu"].get("widen_log10", 0.0)
    pred[:, 0] -= widen
    pred[:, -1] += widen
    out[has] = 10 ** pred
    return out
