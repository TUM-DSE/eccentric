"""Training, saving, loading and predicting with Matchmaker's models.

Gradient-boosted trees (scikit-learn HistGradientBoosting) on the circuit features of FEATURE_NAMES, never
on the code's name, so that unseen codes are predicted from their circuits. Features the training table
does not fill are left out, and missing values elsewhere stay NaN, which the trees handle. The
per-decoder models also take the decoder, as a categorical input.

- regime: per circuit, P(best LER > 0.5)
- ler: per circuit, quantiles of log10 best LER, on circuits whose best decode is measured
- regret: per decoder, median of log10(LER / best LER), on circuits with at least two measured decoders
- time: per decoder, quantiles of log10 ms per shot. A timed-out decode is right-censored: its time per
  shot exceeds timeout_s over a nominal first batch of 100 shots. After a first fit on the decodes with a
  time, each censored decode joins a second fit at max(its predicted median, that bound).

Scaling laws under the trees (parametric=True): trees cannot extrapolate, so beyond the training data they
repeat the value at its edge. The LER and time quantile trees therefore model what is left after a law
fitted first, and predictions beyond the training range follow the law:
- LER: log10 LER = a + b log10 p + c d log10 p + e d + f log10 detectors, fitted on circuits clearly below
  threshold; below threshold the LER falls as (p / p_th)^(d / 2) with the code distance d, here p is the
  error mass per detector, and it grows with the circuit's size. The code distance is an input the user may
  give (`code_distance`), not a circuit feature; without it, the same law without the d terms. Above threshold the law does not hold, so it is used where the trees alone
  predict an LER below 0.2 (fully below 0.02).
- time, per decoder: log10 ms per shot = a + b log10 mechanisms + c log10 (detector firing probability),
  the slopes non-negative: size and noise apart, since a large circuit at low noise is not a small one at
  high noise (BP converges in a few iterations below threshold, runs to its limit above).

With calibrate=True, the outer quantiles are conformalized (conformalized quantile regression): models
trained without a group of codes predict that group, and the interval [q_lo, q_hi] is widened, in
log10, by the amount that makes it hold the nominal share of those held-out truths; for the time, per
decoder. Quantile regression alone is too confident on codes it has not seen. The training table also
records each feature's range, and predictions name the features of a circuit outside it.
"""
import datetime
import json
import math
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import lsq_linear
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .decoders import DECODERS, applicable
from .features import FEATURE_NAMES, FEATURE_VERSION

RATED = ("measured", "low statistics", "upper bound")
ABOVE = 0.5
LER_FLOOR = 1e-8
FIRST_BATCH = 100
PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
              early_stopping=False)
# The LER and time quantiles gain from more capacity on held-out codes (experiments/RESULTS.md: LER p90
# error 0.57 -> 0.51 decades, time within 3x 83% -> 85%); the regime and regret models lose from it
QUANTILE_PARAMS = {**PARAMS, "learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 400}
# Bump when the saved models change form: bundles of another version cannot be loaded
MODEL_VERSION = 3
# Quantiles of a decoder's regret: the median, and pessimistic ones for choosing a cheaper decoder with some
# confidence that it is within the tolerance (recommend(..., confidence=))
REGRET_QUANTILES = (0.5, 0.7, 0.8, 0.9)
# The LER law is fitted on circuits whose best LER is below this, clearly below threshold
LER_LAW_BELOW = 0.05
# Inputs of the laws that are not circuit features, so the trees never see them: the code's distance, which
# the user may give (ECCentric's tables always do); without it, the LER law without distance applies
LAW_INPUTS = ("code_distance",)


class Constant:
    """Stands in for a model whose training labels leave nothing to learn (no rows, or one class)"""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row of features per circuit of the training table, indexed by circuit key, with the laws' inputs
    (LAW_INPUTS; the code distance from ECCentric's `distance` where there is no `code_distance` column)"""
    if "code_distance" not in df.columns and "distance" in df.columns:
        df = df.assign(code_distance=df.distance)
    return df.groupby("circuit_key", sort=False)[[c for c in [*FEATURE_NAMES, *LAW_INPUTS] if c in df.columns]].first()


def circuit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Per circuit with a rate: its best LER (an upper bound counts as its bound), the status of the decode
    that gave it, and whether it is above threshold; indexed by circuit key"""
    rated = df[df.status.isin(RATED)]
    rated = rated.assign(rate=np.where(rated.errors > 0, rated.ler, rated.ler_high)).dropna(subset=["rate"])
    best = rated.loc[rated.groupby("circuit_key").rate.idxmin()]
    return pd.DataFrame({"best_ler": best.rate.to_numpy(float), "best_status": best.status.to_numpy(),
                         "above": (best.rate > ABOVE).to_numpy()}, index=pd.Index(best.circuit_key, name="circuit_key"))


def regret_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Measured decodes of circuits with at least two measured decoders, with log10(LER / best measured LER)"""
    m = df[(df.status == "measured") & (df.ler > 0)]
    m = m[m.groupby("circuit_key").decoder.transform("size") >= 2]
    return m.assign(regret=np.log10(m.ler / m.groupby("circuit_key").ler.transform("min")))


def timeout_bound_ms(timeout_s):
    """Lower bound on the ms per shot of a decode that timed out before its first batch finished"""
    return np.asarray(timeout_s, float) * 1000 / FIRST_BATCH


def _matrix(frame, features):
    return frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _with_decoder(X, decoders, vocab):
    """X with the decoder as a last, categorical column; decoders outside vocab are NaN"""
    index = {d: i for i, d in enumerate(vocab)}
    return np.column_stack([X, np.array([index.get(d, np.nan) for d in decoders], dtype=float)])


def _qname(prefix, q):
    return f"{prefix}_q{round(100 * q):02d}"


def _regressor(quantile, categorical, random_state, params=PARAMS):
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, random_state=random_state,
                                         categorical_features=categorical, **params)


def _fit_quantiles(X, y, quantiles, categorical, random_state, params=PARAMS):
    if len(y) == 0:
        return {q: Constant(math.nan) for q in quantiles}
    return {q: _regressor(q, categorical, random_state, params).fit(X, y) for q in quantiles}


def _predict_quantiles(models, X, quantiles):
    """Quantile predictions (rows x quantiles), sorted along each row so that they never cross"""
    return np.sort(np.column_stack([models[q].predict(X) for q in quantiles]), axis=1)


def _p_above(model, X):
    return model.predict(X) if isinstance(model, Constant) else model.predict_proba(X)[:, 1]


def _column(frame, name):
    if name not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def _log10(values):
    values = np.asarray(values, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(values > 0, np.log10(np.where(values > 0, values, 1.0)), np.nan)


def _ler_terms(frame):
    p, d = _log10(_column(frame, "error_mass_per_detector")), _column(frame, "code_distance")
    return np.column_stack([p, d * p, d, _log10(_column(frame, "detectors"))])


def _ler_terms_without_distance(frame):
    return np.column_stack([_log10(_column(frame, "error_mass_per_detector")), _log10(_column(frame, "detectors"))])


def _time_terms(frame):
    return np.column_stack([_log10(_column(frame, "mechanisms")), _log10(_column(frame, "det_fire_mean"))])


# name: (terms, lower bound of each term's coefficient)
LAWS = {"ler": (_ler_terms, [0.0, 0.0, -np.inf, 0.0]), "ler_without_distance": (_ler_terms_without_distance, [0.0, 0.0]),
        "time": (_time_terms, [0.0, 0.0])}


class Law:
    """y = intercept + terms(frame) @ coef, the coefficients bounded below, fitted by least squares on the rows
    that have every term; predicts NaN for rows missing a term, and for every row if too few rows had them"""

    def __init__(self, name: str):
        self.name, self.coef = name, None

    def fit(self, frame, y):
        terms, lower = LAWS[self.name]
        T, y = terms(frame), np.asarray(y, float)
        ok = np.isfinite(T).all(axis=1) & np.isfinite(y)
        if ok.sum() >= 10 * (T.shape[1] + 1):
            bounds = (np.r_[-np.inf, lower], np.full(T.shape[1] + 1, np.inf))
            self.coef = lsq_linear(np.column_stack([np.ones(ok.sum()), T[ok]]), y[ok], bounds=bounds).x
        return self

    def predict(self, frame):
        if self.coef is None:
            return np.full(len(frame), np.nan)
        return self.coef[0] + LAWS[self.name][0](frame) @ self.coef[1:]


class LerModel:
    """Quantiles of log10 best LER per circuit. Trees alone, and the LER law (with the circuit distance where it
    is known, else without) with trees on what it leaves; the law applies below threshold only, so the two are
    blended by the trees' own median: the law's where that is below LAW_FULL, the trees' above LAW_NONE, and
    in between linearly in log10. Without parametric, the trees alone"""

    # Tuned on the extrapolation splits (experiments/EXTRAPOLATION.md)
    LAW_NONE, LAW_FULL = 0.2, 0.02

    def __init__(self, quantiles, random_state=0, parametric=True, params=QUANTILE_PARAMS):
        self.quantiles, self.random_state, self.parametric, self.params = list(quantiles), random_state, parametric, params

    def fit(self, frame, features, y):
        y = np.asarray(y, float)
        X = _matrix(frame, features)
        self.features, self.laws, self.fallback, self.residual = features, [], 0.0, None
        self.trees = _fit_quantiles(X, y, self.quantiles, None, self.random_state, self.params)
        if self.parametric and len(y):
            below = y < math.log10(LER_LAW_BELOW)
            self.laws = [Law("ler").fit(frame[below], y[below]), Law("ler_without_distance").fit(frame[below], y[below])]
            self.fallback = float(np.median(y))
            self.residual = _fit_quantiles(X, y - self.baseline(frame), self.quantiles, None, self.random_state, self.params)
        return self

    def baseline(self, frame):
        """The law's log10 LER per circuit (the law without distance where the distance is unknown)"""
        out = np.full(len(frame), self.fallback)
        for law in reversed(self.laws):
            p = law.predict(frame)
            out = np.where(np.isfinite(p), p, out)
        return np.clip(out, math.log10(LER_FLOOR), 0.0)

    def law_weight(self, trees_median):
        """How much of the law's prediction to take, from the trees' median log10 LER"""
        none, full = math.log10(self.LAW_NONE), math.log10(self.LAW_FULL)
        return np.clip((none - trees_median) / (none - full), 0.0, 1.0)

    def predict(self, frame):
        """rows x quantiles, log10"""
        X = _matrix(frame, self.features)
        trees = _predict_quantiles(self.trees, X, self.quantiles)
        if self.residual is None:
            return trees
        law = np.clip(_predict_quantiles(self.residual, X, self.quantiles) + self.baseline(frame)[:, None],
                      math.log10(LER_FLOOR), 0.0)
        w = self.law_weight(trees[:, self.quantiles.index(0.5)] if 0.5 in self.quantiles else trees.mean(axis=1))[:, None]
        return np.sort(w * law + (1 - w) * trees, axis=1)


class TimeModel:
    """Quantiles of log10 ms per shot per decode: a time law per decoder, and quantile trees, which also take
    the decoder, on what it leaves; without parametric, the trees alone"""

    def __init__(self, quantiles, random_state=0, parametric=True, params=QUANTILE_PARAMS):
        self.quantiles, self.random_state, self.parametric, self.params = list(quantiles), random_state, parametric, params

    def fit(self, frame, decoders, features, y, vocab):
        y, decoders = np.asarray(y, float), np.asarray(decoders, dtype=object)
        self.features, self.vocab, self.laws, self.fallback = features, list(vocab), {}, {}
        if self.parametric:
            for d in self.vocab:
                mine = decoders == d
                self.laws[d] = Law("time").fit(frame[mine], y[mine])
                self.fallback[d] = float(np.median(y[mine])) if mine.any() else math.nan
        X = _with_decoder(_matrix(frame, features), decoders, self.vocab)
        self.trees = _fit_quantiles(X, y - self.baseline(frame, decoders), self.quantiles,
                                    [False] * len(features) + [True], self.random_state, self.params)
        return self

    def baseline(self, frame, decoders):
        decoders = np.asarray(decoders, dtype=object)
        out = np.zeros(len(frame))
        for d, law in self.laws.items():
            mine = decoders == d
            if mine.any():
                p = law.predict(frame[mine])
                out[mine] = np.where(np.isfinite(p), p, self.fallback[d])
        return out

    def predict(self, frame, decoders):
        """rows x quantiles, log10"""
        X = _with_decoder(_matrix(frame, self.features), decoders, self.vocab)
        return _predict_quantiles(self.trees, X, self.quantiles) + self.baseline(frame, decoders)[:, None]


def _fit_ler(df, circuits, features, quantiles, random_state, parametric):
    """The LER model on the circuits of df whose best decode is measured"""
    labels = circuit_labels(df)
    labels = labels[labels.best_status == "measured"]
    y = np.log10(np.clip(labels.best_ler.to_numpy(float), LER_FLOOR, 1))
    return LerModel(quantiles, random_state, parametric).fit(circuits.loc[labels.index], features, y)


def _fit_time(df, features, quantiles, random_state, parametric):
    """The time model on df's timed decodes; then, with each timed-out decode at max(its predicted median,
    its bound), refitted on both. Also returns the counts of timed and censored decodes"""
    timed = df[df.ms_per_shot > 0]
    censored = df[(df.status == "timeout") & (df.timeout_s > 0) & ~(df.ms_per_shot > 0)]
    vocab = sorted(set(timed.decoder) | set(censored.decoder))
    yt = np.log10(timed.ms_per_shot.to_numpy(float))
    time = TimeModel(quantiles, random_state, parametric).fit(timed, timed.decoder, features, yt, vocab)
    if len(censored) and len(timed):
        median = time.predict(censored, censored.decoder)[:, time.quantiles.index(0.5)] if 0.5 in time.quantiles else \
            time.predict(censored, censored.decoder).mean(axis=1)
        yc = np.maximum(median, np.log10(timeout_bound_ms(censored.timeout_s)))
        both = pd.concat([timed, censored], ignore_index=True)
        time = TimeModel(quantiles, random_state, parametric).fit(both, both.decoder, features, np.concatenate([yt, yc]), vocab)
    return time, len(timed), len(censored)


def train(df: pd.DataFrame, quantiles=(0.1, 0.5, 0.9), random_state=0, manifest: dict | None = None,
          calibrate: bool = False, calibration_folds: int = 4, parametric: bool = True) -> "ModelBundle":
    """Train all models on a training table (dataset.py); manifest, if given, is kept in the bundle's meta;
    calibrate conformalizes the outer quantiles over `calibration_folds` groups of codes; parametric puts the
    scaling laws under the LER and time trees"""
    quantiles = sorted(float(q) for q in quantiles)
    if 0.5 not in quantiles:
        raise ValueError("quantiles must include 0.5")
    features = [c for c in FEATURE_NAMES if c in df.columns and df[c].notna().any()]
    circuits = circuit_features(df)
    labels = circuit_labels(df)

    X = _matrix(circuits.loc[labels.index], features)
    above = labels.above.to_numpy(int)
    if len(np.unique(above)) < 2:
        regime = Constant(float(above[0]) if len(above) else math.nan)
    else:
        regime = HistGradientBoostingClassifier(random_state=random_state, **PARAMS).fit(X, above)

    ler = _fit_ler(df, circuits, features, quantiles, random_state, parametric)

    reg = regret_rows(df)
    regret_vocab = sorted(reg.decoder.unique())
    categorical = [False] * len(features) + [True]
    regret = _fit_quantiles(_with_decoder(_matrix(reg, features), reg.decoder, regret_vocab), reg.regret.to_numpy(),
                            list(REGRET_QUANTILES), categorical, random_state)

    time, timed, censored = _fit_time(df, features, quantiles, random_state, parametric)

    meta = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "features": features,
        "quantiles": quantiles,
        "decoders": {"regret": regret_vocab, "time": time.vocab},
        "params": {"regime": PARAMS, "regret": PARAMS, "ler": QUANTILE_PARAMS, "time": QUANTILE_PARAMS},
        "parametric": parametric,
        "laws": {"ler": {law.name: None if law.coef is None else [float(c) for c in law.coef] for law in ler.laws},
                 "time": {d: None if law.coef is None else [float(c) for c in law.coef] for d, law in time.laws.items()}},
        "random_state": random_state,
        "training": {"rows": len(df), "circuits": len(circuits), "regime_circuits": len(labels),
                     "above_share": float(above.mean()) if len(above) else math.nan,
                     "ler_circuits": int((labels.best_status == "measured").sum()),
                     "regret_decodes": len(reg), "time_decodes": timed, "time_censored": censored},
        "manifest": manifest,
        "ranges": {f: [float(np.nanmin(circuits[f])), float(np.nanmax(circuits[f]))] for f in [*features, *LAW_INPUTS]
                   if f in circuits.columns and circuits[f].notna().any()},
        "conformal": _conformal(df, features, quantiles, calibration_folds, random_state, parametric) if calibrate else None,
    }
    return ModelBundle({"regime": regime, "ler": ler, "regret": regret, "time": time}, meta)


def _cqr(lo, hi, y, coverage):
    """The widening (in log10) that makes [lo, hi] hold `coverage` of y, with the finite-sample correction"""
    if len(y) == 0:
        return 0.0
    scores = np.maximum(lo - y, y - hi)
    level = min(1.0, coverage * (1 + 1 / len(y)))
    return float(max(np.quantile(scores, level), 0.0))


def _conformal(df, features, quantiles, folds, random_state, parametric=True):
    """Widenings of the outer LER and time quantiles from predictions of codes held out of training"""
    outer = [quantiles[0], quantiles[-1]]
    coverage = quantiles[-1] - quantiles[0]
    circuits = circuit_features(df)
    labels = circuit_labels(df)
    codes = df.groupby("circuit_key", sort=False).code.first()
    ler_lo, ler_hi, ler_y = [], [], []
    time_parts = []
    keys = circuits.index.to_numpy()
    groups = codes.loc[keys].to_numpy()
    for train_idx, test_idx in GroupKFold(n_splits=min(folds, len(set(groups)))).split(keys, groups=groups):
        train_keys, test_keys = set(keys[train_idx]), set(keys[test_idx])
        part, held = df[df.circuit_key.isin(train_keys)], df[df.circuit_key.isin(test_keys)]
        model = _fit_ler(part, circuits, features, outer, random_state, parametric)
        test_lab = labels[labels.index.isin(test_keys) & (labels.best_status == "measured")]
        if len(test_lab):
            pred = model.predict(circuits.loc[test_lab.index])
            ler_lo.append(pred[:, 0])
            ler_hi.append(pred[:, 1])
            ler_y.append(np.log10(np.clip(test_lab.best_ler.to_numpy(), LER_FLOOR, 1)))
        timed, held_timed = part[part.ms_per_shot > 0], held[held.ms_per_shot > 0]
        vocab = sorted(timed.decoder.unique())
        held_timed = held_timed[held_timed.decoder.isin(vocab)]
        if len(timed) and len(held_timed):
            model = TimeModel(outer, random_state, parametric).fit(timed, timed.decoder, features,
                                                                   np.log10(timed.ms_per_shot.to_numpy(float)), vocab)
            pred = model.predict(held_timed, held_timed.decoder)
            time_parts.append(pd.DataFrame({"decoder": held_timed.decoder.to_numpy(), "lo": pred[:, 0], "hi": pred[:, 1],
                                            "y": np.log10(held_timed.ms_per_shot.to_numpy(float))}))
    ler = _cqr(np.concatenate(ler_lo), np.concatenate(ler_hi), np.concatenate(ler_y), coverage) if ler_y else 0.0
    times = pd.concat(time_parts, ignore_index=True) if time_parts else pd.DataFrame(columns=["decoder", "lo", "hi", "y"])
    time = {d: _cqr(g.lo.to_numpy(), g.hi.to_numpy(), g.y.to_numpy(), coverage) for d, g in times.groupby("decoder")}
    return {"coverage": coverage, "folds": folds, "ler": ler, "time": time,
            "time_all": _cqr(times.lo.to_numpy(), times.hi.to_numpy(), times.y.to_numpy(), coverage) if len(times) else 0.0}


class ModelBundle:
    """The trained models, and in meta what they were trained on"""

    def __init__(self, models: dict, meta: dict):
        self.models, self.meta = models, meta

    @property
    def features(self) -> list:
        return self.meta["features"]

    @property
    def quantiles(self) -> list:
        return self.meta["quantiles"]

    def decoders(self) -> list:
        """The decoders seen in training, in the order of decoders.DECODERS"""
        seen = set(self.meta["decoders"]["regret"]) | set(self.meta["decoders"]["time"])
        return [d for d in DECODERS if d in seen] + sorted(seen - set(DECODERS))

    def predict(self, features, decoders=None) -> pd.DataFrame:
        """Predictions per circuit and decoder. features: a frame with one row per circuit (FEATURE_NAMES
        columns, optionally `circuit_key`, else the index is the key), or one circuit's features dict.
        decoders: the decoders to predict for (default: those seen in training); a model that did not see
        a decoder in training predicts NaN for it"""
        frame = pd.DataFrame([features]) if isinstance(features, dict) else pd.DataFrame(features)
        keys = frame["circuit_key"].to_numpy() if "circuit_key" in frame.columns else frame.index.to_numpy()
        decoders = list(self.decoders() if decoders is None else decoders)
        n, k = len(frame), len(decoders)
        X = _matrix(frame, self.features)
        q, models = self.quantiles, self.models

        ler = models["ler"].predict(frame)
        conformal = self.meta.get("conformal")
        if conformal:
            ler[:, 0] -= conformal["ler"]
            ler[:, -1] += conformal["ler"]
        ler = 10 ** ler
        Xd, dec = np.repeat(X, k, axis=0), np.tile(np.asarray(decoders, dtype=object), n)
        regret_vocab, time_vocab = self.meta["decoders"]["regret"], self.meta["decoders"]["time"]
        regret = 10 ** np.maximum(_predict_quantiles(models["regret"], _with_decoder(Xd, dec, regret_vocab),
                                                     REGRET_QUANTILES), 0)
        regret[~np.isin(dec, regret_vocab)] = np.nan
        ms = models["time"].predict(frame.iloc[np.repeat(np.arange(n), k)].reset_index(drop=True), dec)
        if conformal:
            widen = np.array([conformal["time"].get(d, conformal["time_all"]) for d in dec])
            ms[:, 0] -= widen
            ms[:, -1] += widen
        ms = 10 ** ms
        ms[~np.isin(dec, time_vocab)] = np.nan

        out = pd.DataFrame({"circuit_key": np.repeat(keys, k), "decoder": dec,
                            "p_above": np.repeat(_p_above(models["regime"], X), k)})
        for j, quantile in enumerate(q):
            out[_qname("ler", quantile)] = np.repeat(ler[:, j], k)
        for j, quantile in enumerate(REGRET_QUANTILES):
            out[_qname("regret", quantile)] = regret[:, j]
        out["ler_decoder_q50"] = out.ler_q50 * out.regret_q50
        for j, quantile in enumerate(q):
            out[_qname("ms", quantile)] = ms[:, j]
        everything = frame.reindex(columns=[*FEATURE_NAMES, *LAW_INPUTS]).apply(pd.to_numeric, errors="coerce").astype(float)
        fits = [applicable(f) for f in everything.to_dict("records")]
        out["applicable"] = [fit.get(d, True) for fit in fits for d in decoders]
        out["outside_training"] = np.repeat([", ".join(self.outside_training(f)) for f in everything.to_dict("records")], k)
        return out

    def outside_training(self, features: dict) -> list:
        """Features of a circuit beyond the range seen in training, where predictions extrapolate"""
        out = []
        for name, (lo, hi) in self.meta.get("ranges", {}).items():
            v = features.get(name)
            if v is not None and not (isinstance(v, float) and math.isnan(v)) and not lo <= v <= hi:
                out.append(f"{name} {v:.4g} (trained on {lo:.4g}-{hi:.4g})")
        return out

    def save(self, out_dir: str) -> None:
        """Write models.joblib (the estimators) and meta.json"""
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.models, os.path.join(out_dir, "models.joblib"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    @classmethod
    def load(cls, model_dir: str) -> "ModelBundle":
        """Read a saved bundle; warns when it was trained with another scikit-learn or feature version"""
        with open(os.path.join(model_dir, "meta.json")) as f:
            meta = json.load(f)
        if meta.get("model_version", 1) != MODEL_VERSION:
            raise ValueError(f"{model_dir} holds models of version {meta.get('model_version', 1)}, this Matchmaker "
                             f"reads version {MODEL_VERSION}: retrain (qec-matchmaker train)")
        if meta.get("sklearn_version") != sklearn.__version__:
            warnings.warn(f"models trained with scikit-learn {meta.get('sklearn_version')}, running "
                          f"{sklearn.__version__}: predictions may differ or fail", stacklevel=2)
        if meta.get("feature_version") != FEATURE_VERSION:
            warnings.warn(f"models trained on feature version {meta.get('feature_version')}, features are now "
                          f"version {FEATURE_VERSION}: retrain", stacklevel=2)
        return cls(joblib.load(os.path.join(model_dir, "models.joblib")), meta)
