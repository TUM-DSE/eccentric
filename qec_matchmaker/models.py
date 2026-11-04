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


class Constant:
    """Stands in for a model whose training labels leave nothing to learn (no rows, or one class)"""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row of features per circuit of the training table, indexed by circuit key"""
    return df.groupby("circuit_key", sort=False)[[c for c in FEATURE_NAMES if c in df.columns]].first()


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


def _regressor(quantile, categorical, random_state):
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, random_state=random_state,
                                         categorical_features=categorical, **PARAMS)


def _fit_quantiles(X, y, quantiles, categorical, random_state):
    if len(y) == 0:
        return {q: Constant(math.nan) for q in quantiles}
    return {q: _regressor(q, categorical, random_state).fit(X, y) for q in quantiles}


def _predict_quantiles(models, X, quantiles):
    """Quantile predictions (rows x quantiles), sorted along each row so that they never cross"""
    return np.sort(np.column_stack([models[q].predict(X) for q in quantiles]), axis=1)


def _p_above(model, X):
    return model.predict(X) if isinstance(model, Constant) else model.predict_proba(X)[:, 1]


def train(df: pd.DataFrame, quantiles=(0.1, 0.5, 0.9), random_state=0, manifest: dict | None = None,
          calibrate: bool = False, calibration_folds: int = 4) -> "ModelBundle":
    """Train all models on a training table (dataset.py); manifest, if given, is kept in the bundle's meta;
    calibrate conformalizes the outer quantiles over `calibration_folds` groups of codes"""
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

    measured = labels.best_status.to_numpy() == "measured"
    y = np.log10(np.clip(labels.best_ler.to_numpy()[measured], LER_FLOOR, 1))
    ler = _fit_quantiles(X[measured], y, quantiles, None, random_state)

    reg = regret_rows(df)
    regret_vocab = sorted(reg.decoder.unique())
    categorical = [False] * len(features) + [True]
    regret = _fit_quantiles(_with_decoder(_matrix(reg, features), reg.decoder, regret_vocab), reg.regret.to_numpy(),
                            [0.5], categorical, random_state)[0.5]

    timed = df[df.ms_per_shot > 0]
    censored = df[(df.status == "timeout") & (df.timeout_s > 0) & ~(df.ms_per_shot > 0)]
    time_vocab = sorted(set(timed.decoder) | set(censored.decoder))
    Xt = _with_decoder(_matrix(timed, features), timed.decoder, time_vocab)
    yt = np.log10(timed.ms_per_shot.to_numpy(float))
    time = _fit_quantiles(Xt, yt, quantiles, categorical, random_state)
    if len(censored) and len(timed):
        Xc = _with_decoder(_matrix(censored, features), censored.decoder, time_vocab)
        yc = np.maximum(time[0.5].predict(Xc), np.log10(timeout_bound_ms(censored.timeout_s)))
        time = _fit_quantiles(np.vstack([Xt, Xc]), np.concatenate([yt, yc]), quantiles, categorical, random_state)

    meta = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "feature_version": FEATURE_VERSION,
        "features": features,
        "quantiles": quantiles,
        "decoders": {"regret": regret_vocab, "time": time_vocab},
        "params": PARAMS,
        "random_state": random_state,
        "training": {"rows": len(df), "circuits": len(circuits), "regime_circuits": len(labels),
                     "above_share": float(above.mean()) if len(above) else math.nan, "ler_circuits": int(measured.sum()),
                     "regret_decodes": len(reg), "time_decodes": len(timed), "time_censored": len(censored)},
        "manifest": manifest,
        "ranges": {f: [float(np.nanmin(circuits[f])), float(np.nanmax(circuits[f]))] for f in features
                   if circuits[f].notna().any()},
        "conformal": _conformal(df, features, quantiles, calibration_folds, random_state) if calibrate else None,
    }
    return ModelBundle({"regime": regime, "ler": ler, "regret": regret, "time": time}, meta)


def _cqr(lo, hi, y, coverage):
    """The widening (in log10) that makes [lo, hi] hold `coverage` of y, with the finite-sample correction"""
    if len(y) == 0:
        return 0.0
    scores = np.maximum(lo - y, y - hi)
    level = min(1.0, coverage * (1 + 1 / len(y)))
    return float(max(np.quantile(scores, level), 0.0))


def _conformal(df, features, quantiles, folds, random_state):
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
        lab = circuit_labels(part)
        measured = lab.best_status.to_numpy() == "measured"
        Xl = _matrix(circuits.loc[lab.index[measured]], features)
        yl = np.log10(np.clip(lab.best_ler.to_numpy()[measured], LER_FLOOR, 1))
        models = _fit_quantiles(Xl, yl, outer, None, random_state)
        test_lab = labels[labels.index.isin(test_keys) & (labels.best_status == "measured")]
        if len(test_lab):
            Xh = _matrix(circuits.loc[test_lab.index], features)
            pred = _predict_quantiles(models, Xh, outer)
            ler_lo.append(pred[:, 0])
            ler_hi.append(pred[:, 1])
            ler_y.append(np.log10(np.clip(test_lab.best_ler.to_numpy(), LER_FLOOR, 1)))
        timed, held_timed = part[part.ms_per_shot > 0], held[held.ms_per_shot > 0]
        vocab = sorted(timed.decoder.unique())
        if len(timed) and len(held_timed):
            categorical = [False] * len(features) + [True]
            models = _fit_quantiles(_with_decoder(_matrix(timed, features), timed.decoder, vocab),
                                    np.log10(timed.ms_per_shot.to_numpy(float)), outer, categorical, random_state)
            held_timed = held_timed[held_timed.decoder.isin(vocab)]
            pred = _predict_quantiles(models, _with_decoder(_matrix(held_timed, features), held_timed.decoder, vocab), outer)
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

        ler = _predict_quantiles(models["ler"], X, q)
        conformal = self.meta.get("conformal")
        if conformal:
            ler[:, 0] -= conformal["ler"]
            ler[:, -1] += conformal["ler"]
        ler = 10 ** ler
        Xd, dec = np.repeat(X, k, axis=0), np.tile(np.asarray(decoders, dtype=object), n)
        regret_vocab, time_vocab = self.meta["decoders"]["regret"], self.meta["decoders"]["time"]
        regret = 10 ** np.maximum(models["regret"].predict(_with_decoder(Xd, dec, regret_vocab)), 0)
        regret[~np.isin(dec, regret_vocab)] = np.nan
        ms = _predict_quantiles(models["time"], _with_decoder(Xd, dec, time_vocab), q)
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
        out["regret_q50"] = regret
        out["ler_decoder_q50"] = out.ler_q50 * regret
        for j, quantile in enumerate(q):
            out[_qname("ms", quantile)] = ms[:, j]
        everything = frame.reindex(columns=FEATURE_NAMES).apply(pd.to_numeric, errors="coerce").astype(float)
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
        if meta.get("sklearn_version") != sklearn.__version__:
            warnings.warn(f"models trained with scikit-learn {meta.get('sklearn_version')}, running "
                          f"{sklearn.__version__}: predictions may differ or fail", stacklevel=2)
        if meta.get("feature_version") != FEATURE_VERSION:
            warnings.warn(f"models trained on feature version {meta.get('feature_version')}, features are now "
                          f"version {FEATURE_VERSION}: retrain", stacklevel=2)
        return cls(joblib.load(os.path.join(model_dir, "models.joblib")), meta)
