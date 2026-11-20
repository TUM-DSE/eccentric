"""Decoding on a GPU: ECCentric's GPU decoders (eccentric/decoders/gpu) and a model of their time per shot.

The six decoders below have GPU versions that give the CPU decoder's corrections, exactly or within statistical
error (ECCentric's analysis/GPU.md), so Matchmaker's accuracy predictions hold for both and only the time
differs. The time model is trained on GPU pilots: up to 10^5 shots of 306 circuits of ECCentric's sweep per
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


def _frame(pilots: pd.DataFrame, store_features: dict, features: list) -> pd.DataFrame:
    """The pilots joined with their circuits' features (pilots without features are dropped)"""
    known = pilots[pilots.circuit_key.isin(store_features)]
    feats = pd.DataFrame([store_features[k] for k in known.circuit_key]).reindex(columns=features)
    return pd.concat([known.reset_index(drop=True), feats.apply(pd.to_numeric, errors="coerce")], axis=1)


def _fit(frame, features, quantiles, parametric, random_state=0):
    y = np.log10(frame.gpu_ms_per_shot.to_numpy(float))
    return TimeModel(quantiles, random_state, parametric).fit(frame, frame.decoder, features, y,
                                                              sorted(GPU_VERSIONS))


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
    # Reference: the circuit's measured CPU time divided by its decoder's median speedup (needs a CPU decode)
    speedup = np.log10(frame.cpu_ms_per_shot.to_numpy(float)) - y
    med = frame.assign(s=speedup).groupby("decoder").s.transform("median").to_numpy()
    ref = np.abs(np.log10(frame.cpu_ms_per_shot.to_numpy(float)) - med - y)
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
        "device": DEVICE, "versions": GPU_VERSIONS, "widen_log10": report["widen_log10"],
        "build_seconds": frame.groupby("decoder").build_seconds.median().round(3).to_dict(),
        "evaluation": report,
    }
    return report


def gpu_ms(bundle, frame: pd.DataFrame, decoders) -> np.ndarray:
    """rows x quantiles of GPU ms per shot (NaN for decoders without a GPU version or a GPU model)"""
    decoders = np.asarray(decoders, dtype=object)
    out = np.full((len(frame), len(bundle.quantiles)), np.nan)
    model = bundle.models.get("gpu_time")
    has = np.isin(decoders, list(GPU_VERSIONS))
    if model is None or not has.any():
        return out
    pred = model.predict(frame[has].reset_index(drop=True), decoders[has])
    widen = bundle.meta["gpu"].get("widen_log10", 0.0)
    pred[:, 0] -= widen
    pred[:, -1] += widen
    out[has] = 10 ** pred
    return out
