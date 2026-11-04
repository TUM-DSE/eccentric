"""Core of the model-quality experiments: a frozen, parameterised copy of qec_matchmaker's training,
prediction and held-out metrics (models.py / evaluate.py / recommend.py at ec4f622, i.e. evaluate() with
calibrate=False), so that concurrent edits to the package do not move these numbers.

What can be varied per experiment (Config):
- HistGradientBoosting parameters, globally and per model (regime, ler, regret, time)
- monotonic constraints per model ({feature: +1/-1})
- features dropped from every model
- the share of each fold's training codes used (learning curve)

Every unit-level held-out result (per circuit / per decode) keeps its code, so that differences between
two configurations can be bootstrapped over codes (paired).
"""
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from qec_matchmaker.dataset import load_dataset
from qec_matchmaker.features import FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# ---- constants and helpers copied from qec_matchmaker/models.py (ec4f622) ----
RATED = ("measured", "low statistics", "upper bound")
ABOVE = 0.5
LER_FLOOR = 1e-8
FIRST_BATCH = 100
QUANTILES = (0.1, 0.5, 0.9)
BASE_PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                   early_stopping=False)
MODELS = ("regime", "ler", "regret", "time")
DECODER_ORDER = ["mwpm", "bposd", "bplsd", "belief_find", "relay_bp", "tesseract", "chromobius"]


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def circuit_features(df):
    return df.groupby("circuit_key", sort=False)[[c for c in FEATURE_NAMES if c in df.columns]].first()


def circuit_labels(df):
    rated = df[df.status.isin(RATED)]
    rated = rated.assign(rate=np.where(rated.errors > 0, rated.ler, rated.ler_high)).dropna(subset=["rate"])
    best = rated.loc[rated.groupby("circuit_key").rate.idxmin()]
    return pd.DataFrame({"best_ler": best.rate.to_numpy(float), "best_status": best.status.to_numpy(),
                         "above": (best.rate > ABOVE).to_numpy()}, index=pd.Index(best.circuit_key, name="circuit_key"))


def regret_rows(df):
    m = df[(df.status == "measured") & (df.ler > 0)]
    m = m[m.groupby("circuit_key").decoder.transform("size") >= 2]
    return m.assign(regret=np.log10(m.ler / m.groupby("circuit_key").ler.transform("min")))


def timeout_bound_ms(timeout_s):
    return np.asarray(timeout_s, float) * 1000 / FIRST_BATCH


def _matrix(frame, features):
    return frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _with_decoder(X, decoders, vocab):
    index = {d: i for i, d in enumerate(vocab)}
    return np.column_stack([X, np.array([index.get(d, np.nan) for d in decoders], dtype=float)])


def _predict_quantiles(models, X, quantiles):
    return np.sort(np.column_stack([models[q].predict(X) for q in quantiles]), axis=1)


def _p_above(model, X):
    return model.predict(X) if isinstance(model, Constant) else model.predict_proba(X)[:, 1]


def _applicable(features):
    """decoders.applicable at ec4f622"""
    def flag(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else bool(v)
    dec = flag(features.get("decomposable"))
    if dec is None:
        dec = flag(features.get("graphlike")) or False
    return {name: (dec if name == "mwpm" else bool(flag(features.get("color_annotated"))) if name == "chromobius"
                   else True) for name in DECODER_ORDER}


# ---- the experiment knobs ----
@dataclass
class Config:
    name: str = "baseline"
    params: dict = field(default_factory=dict)          # overrides of BASE_PARAMS for every model
    per_model: dict = field(default_factory=dict)       # {model: {param: value}} on top of params
    monotone: dict = field(default_factory=dict)        # {model: {feature: +1 or -1}}
    drop: tuple = ()                                    # features left out of every model
    code_fraction: float = 1.0                          # share of each fold's training codes used
    subsample_seed: int = 0
    circuit_fraction: float = 1.0                       # share of each training code's circuits used

    def model_params(self, model):
        return {**BASE_PARAMS, **self.params, **self.per_model.get(model, {})}

    def monotonic_cst(self, model, features, categorical_last):
        cons = self.monotone.get(model)
        if not cons:
            return None
        cst = [int(cons.get(f, 0)) for f in features] + ([0] if categorical_last else [])
        return cst if any(cst) else None


def _hgb_params(cfg, model, features, categorical_last):
    p = cfg.model_params(model)
    cst = cfg.monotonic_cst(model, features, categorical_last)
    if cst is not None:
        p["monotonic_cst"] = cst
    return p


def _fit_quantiles(X, y, quantiles, categorical, random_state, params):
    if len(y) == 0:
        return {q: Constant(math.nan) for q in quantiles}
    return {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=random_state,
                                             categorical_features=categorical, **params).fit(X, y) for q in quantiles}


def train(df, cfg: Config, random_state=0, quantiles=QUANTILES):
    """models.train at ec4f622 with the Config's knobs; returns (models, meta)"""
    quantiles = sorted(quantiles)
    features = [c for c in FEATURE_NAMES if c in df.columns and df[c].notna().any() and c not in cfg.drop]
    circuits = circuit_features(df)
    labels = circuit_labels(df)

    X = _matrix(circuits.loc[labels.index], features)
    above = labels.above.to_numpy(int)
    if len(np.unique(above)) < 2:
        regime = Constant(float(above[0]) if len(above) else math.nan)
    else:
        regime = HistGradientBoostingClassifier(random_state=random_state,
                                                **_hgb_params(cfg, "regime", features, False)).fit(X, above)

    measured = labels.best_status.to_numpy() == "measured"
    y = np.log10(np.clip(labels.best_ler.to_numpy()[measured], LER_FLOOR, 1))
    ler = _fit_quantiles(X[measured], y, quantiles, None, random_state, _hgb_params(cfg, "ler", features, False))

    reg = regret_rows(df)
    regret_vocab = sorted(reg.decoder.unique())
    categorical = [False] * len(features) + [True]
    regret = _fit_quantiles(_with_decoder(_matrix(reg, features), reg.decoder, regret_vocab), reg.regret.to_numpy(),
                            [0.5], categorical, random_state, _hgb_params(cfg, "regret", features, True))[0.5]

    tp = _hgb_params(cfg, "time", features, True)
    timed = df[df.ms_per_shot > 0]
    censored = df[(df.status == "timeout") & (df.timeout_s > 0) & ~(df.ms_per_shot > 0)]
    time_vocab = sorted(set(timed.decoder) | set(censored.decoder))
    Xt = _with_decoder(_matrix(timed, features), timed.decoder, time_vocab)
    yt = np.log10(timed.ms_per_shot.to_numpy(float))
    tmodels = _fit_quantiles(Xt, yt, quantiles, categorical, random_state, tp)
    if len(censored) and len(timed):
        Xc = _with_decoder(_matrix(censored, features), censored.decoder, time_vocab)
        yc = np.maximum(tmodels[0.5].predict(Xc), np.log10(timeout_bound_ms(censored.timeout_s)))
        tmodels = _fit_quantiles(np.vstack([Xt, Xc]), np.concatenate([yt, yc]), quantiles, categorical, random_state, tp)
    meta = {"features": features, "quantiles": quantiles, "decoders": {"regret": regret_vocab, "time": time_vocab}}
    return {"regime": regime, "ler": ler, "regret": regret, "time": tmodels}, meta


def predict(models, meta, frame, decoders):
    """ModelBundle.predict at ec4f622 (frame: one row per circuit with a circuit_key column)"""
    keys = frame["circuit_key"].to_numpy()
    n, k = len(frame), len(decoders)
    X = _matrix(frame, meta["features"])
    q = meta["quantiles"]
    ler = 10 ** _predict_quantiles(models["ler"], X, q)
    Xd, dec = np.repeat(X, k, axis=0), np.tile(np.asarray(decoders, dtype=object), n)
    rv, tv = meta["decoders"]["regret"], meta["decoders"]["time"]
    regret = 10 ** np.maximum(models["regret"].predict(_with_decoder(Xd, dec, rv)), 0)
    regret[~np.isin(dec, rv)] = np.nan
    ms = 10 ** _predict_quantiles(models["time"], _with_decoder(Xd, dec, tv), q)
    ms[~np.isin(dec, tv)] = np.nan
    out = pd.DataFrame({"circuit_key": np.repeat(keys, k), "decoder": dec,
                        "p_above": np.repeat(_p_above(models["regime"], X), k)})
    for j, name in enumerate(("q10", "q50", "q90")):
        out[f"ler_{name}"] = np.repeat(ler[:, j], k)
    out["regret_q50"] = regret
    out["ler_decoder_q50"] = out.ler_q50 * regret
    for j, name in enumerate(("q10", "q50", "q90")):
        out[f"ms_{name}"] = ms[:, j]
    everything = frame.reindex(columns=FEATURE_NAMES).apply(pd.to_numeric, errors="coerce").astype(float)
    fits = [_applicable(f) for f in everything.to_dict("records")]
    out["applicable"] = [fit.get(d, True) for fit in fits for d in decoders]
    return out


# ---- cross-validation ----
def setup(df):
    circuits = circuit_features(df)
    codes = df.groupby("circuit_key", sort=False).code.first().loc[circuits.index].to_numpy()
    present = set(df.decoder)
    decoders = [d for d in DECODER_ORDER if d in present] + sorted(present - set(DECODER_ORDER))
    splits = list(GroupKFold(n_splits=min(10, len(set(codes)))).split(circuits, groups=codes))
    return circuits, codes, decoders, splits


def _fold(df, circuits, codes, decoders, fold, train_idx, test_idx, cfg, random_state, permute=None):
    train_keys = circuits.index[train_idx]
    if cfg.code_fraction < 1:
        train_codes = np.array(sorted(set(codes[train_idx])))
        rng = np.random.default_rng(1000 * cfg.subsample_seed + fold)
        keep = rng.choice(train_codes, size=max(2, round(cfg.code_fraction * len(train_codes))), replace=False)
        train_keys = circuits.index[train_idx][np.isin(codes[train_idx], keep)]
    if cfg.circuit_fraction < 1:
        rng = np.random.default_rng(1000 * cfg.subsample_seed + fold + 500)
        by_code = pd.Series(codes, index=circuits.index).loc[train_keys]
        train_keys = pd.Index([k for _, ks in by_code.groupby(by_code, sort=True) for k in
                               rng.choice(ks.index.to_numpy(), size=max(1, round(cfg.circuit_fraction * len(ks))), replace=False)])
    models, meta = train(df[df.circuit_key.isin(train_keys)], cfg, random_state)
    test = circuits.iloc[test_idx].reset_index()
    pred = predict(models, meta, test, decoders).assign(fold=fold)
    perm = permute(models, meta, test, decoders, fold) if permute else None
    return pred, perm


def cross_val(df, cfg: Config, random_state=0, n_jobs=5, permute=None, prepared=None):
    """Held-out predictions of every circuit and decoder (as evaluate.cross_val_predictions), folds in parallel;
    with permute(models, meta, test_frame, decoders, fold), also returns its per-fold outputs"""
    circuits, codes, decoders, splits = prepared or setup(df)
    res = Parallel(n_jobs=n_jobs)(delayed(_fold)(df, circuits, codes, decoders, f, tr, te, cfg, random_state, permute)
                                  for f, (tr, te) in enumerate(splits))
    pred = pd.concat([r[0] for r in res], ignore_index=True)
    return (pred, [r[1] for r in res]) if permute else pred


# ---- held-out metrics: same definitions as evaluate.py, kept per unit with its code ----
def recommend_choice(rows, objective, tolerance=1.25):
    """recommend.recommend(...).decoder at ec4f622 for objectives accurate / cheapest"""
    fits = rows[rows.applicable.astype(bool)]
    c = fits.dropna(subset=["ler_decoder_q50", "ms_q50"])
    if c.empty:
        return None
    if float(c.p_above.iloc[0]) > 0.5:
        return c.sort_values(["ms_q50", "ler_decoder_q50"]).iloc[0].decoder
    if objective == "accurate":
        return c.sort_values(["ler_decoder_q50", "ms_q50"]).iloc[0].decoder
    lowest = c.ler_decoder_q50.min()
    return c.assign(eligible=c.ler_decoder_q50 <= tolerance * lowest).sort_values(
        ["eligible", "ms_q50", "ler_decoder_q50"], ascending=[False, True, True]).iloc[0].decoder


def units(df, pred, with_recommendation=True):
    """Per-unit held-out results with their code: regime (circuit), ler (circuit), regret (decode), time
    (decode), rec (circuit)"""
    code_of = df.groupby("circuit_key").code.first()
    labels = circuit_labels(df)
    first = pred.groupby("circuit_key").first()

    p = first.p_above.reindex(labels.index).to_numpy()
    regime = pd.DataFrame({"code": code_of.loc[labels.index].to_numpy(), "truth": labels.above.to_numpy(),
                           "p": p, "guess": p > 0.5}, index=labels.index)

    lm = labels[labels.best_status == "measured"]
    y = np.log10(np.clip(lm.best_ler.to_numpy(), LER_FLOOR, 1))
    q = np.log10(first.reindex(lm.index)[["ler_q10", "ler_q50", "ler_q90"]].to_numpy(float))
    ler = pd.DataFrame({"code": code_of.loc[lm.index].to_numpy(), "y": y, "pred": q[:, 1],
                        "inside": (q[:, 0] <= y) & (y <= q[:, 2]), "lo": q[:, 0], "hi": q[:, 2]}, index=lm.index)

    r = regret_rows(df).merge(pred[["circuit_key", "decoder", "regret_q50"]], on=["circuit_key", "decoder"])
    r = r.dropna(subset=["regret_q50"])
    regret = pd.DataFrame({"code": r.code.to_numpy(), "err": np.abs(np.log10(r.regret_q50.to_numpy()) - r.regret.to_numpy())})

    cols = ["circuit_key", "decoder", "ms_q10", "ms_q50", "ms_q90"]
    t = df[df.ms_per_shot > 0][["circuit_key", "code", "decoder", "ms_per_shot"]].merge(pred[cols], on=["circuit_key", "decoder"])
    t = t.dropna(subset=["ms_q50"])
    yt = np.log10(t.ms_per_shot.to_numpy())
    qt = np.log10(t[["ms_q10", "ms_q50", "ms_q90"]].to_numpy(float))
    timeu = pd.DataFrame({"code": t.code.to_numpy(), "decoder": t.decoder.to_numpy(), "y": yt, "pred": qt[:, 1],
                          "inside": (qt[:, 0] <= yt) & (yt <= qt[:, 2]), "lo": qt[:, 0], "hi": qt[:, 2]})
    out = {"regime": regime, "ler": ler, "regret": regret, "time": timeu}
    if with_recommendation:
        out["rec"] = rec_units_fast(df, pred)
    return out


def rec_units(df, pred):
    m = df[(df.status == "measured") & (df.ms_per_shot > 0)]
    m = m[m.groupby("circuit_key").decoder.transform("size") >= 2]
    m = m[m.circuit_key.isin(set(pred.circuit_key))]
    predicted = dict(tuple(pred.groupby("circuit_key", sort=False)))
    rows = []
    for key, s in m.groupby("circuit_key", sort=False):
        ler, ms = dict(zip(s.decoder, s.ler)), dict(zip(s.decoder, s.ms_per_shot))
        best = s.loc[s.ler.idxmin()]
        good = set(s[~(s.ler_low > best.ler_high)].decoder)
        cheapest_ms = s[s.decoder.isin(good)].ms_per_shot.min()
        row = {"circuit_key": key, "code": s.code.iloc[0]}
        for name in ("accurate", "cheapest"):
            d = recommend_choice(predicted[key], name)
            row[f"{name}_choice"] = d
            row[f"{name}_fallback"] = d not in ler
            if d not in ler:
                d = "bposd" if "bposd" in ler else min(ler, key=ler.get)
            row[f"{name}_good"] = d in good
            row[f"{name}_regret"] = ler[d] / best.ler
            row[f"{name}_time"] = ms[d] / cheapest_ms
        rows.append(row)
    return pd.DataFrame(rows)


def _first_by(frame, keys):
    """Per circuit, the row that sorts first by keys (stable: ties keep the original, decoder, order)"""
    return frame.sort_values(keys, kind="stable").groupby("circuit_key", sort=False).head(1).set_index("circuit_key")


def choices_fast(pred, tolerance=1.25):
    """recommend_choice for every circuit at once: {objective: Series circuit_key -> decoder}"""
    c = pred[pred.applicable.astype(bool)].dropna(subset=["ler_decoder_q50", "ms_q50"]).copy()
    c["order"] = np.arange(len(c))
    c["lowest"] = c.groupby("circuit_key").ler_decoder_q50.transform("min")
    c["not_eligible"] = ~(c.ler_decoder_q50 <= tolerance * c.lowest)
    above = _first_by(c, ["circuit_key", "ms_q50", "ler_decoder_q50", "order"]).decoder
    acc = _first_by(c, ["circuit_key", "ler_decoder_q50", "ms_q50", "order"]).decoder
    cheap = _first_by(c, ["circuit_key", "not_eligible", "ms_q50", "ler_decoder_q50", "order"]).decoder
    is_above = c.groupby("circuit_key").p_above.first() > 0.5
    return {"accurate": acc.where(~is_above.reindex(acc.index), above.reindex(acc.index)),
            "cheapest": cheap.where(~is_above.reindex(cheap.index), above.reindex(cheap.index))}


def rec_units_fast(df, pred):
    """rec_units, vectorised (identical results)"""
    m = df[(df.status == "measured") & (df.ms_per_shot > 0)]
    m = m[m.groupby("circuit_key").decoder.transform("size") >= 2]
    m = m[m.circuit_key.isin(set(pred.circuit_key))].copy()
    m["order"] = np.arange(len(m))
    best = m.loc[m.groupby("circuit_key", sort=False).ler.idxmin()].set_index("circuit_key")
    m["best_ler"] = m.circuit_key.map(best.ler)
    m["good"] = ~(m.ler_low > m.circuit_key.map(best.ler_high))
    cheapest_ms = m[m.good].groupby("circuit_key").ms_per_shot.min()
    keys = best.index
    out = pd.DataFrame({"circuit_key": keys, "code": best.code.to_numpy()})
    lookup = m.set_index(["circuit_key", "decoder"])
    has_bposd = set(m[m.decoder == "bposd"].circuit_key)
    ch = choices_fast(pred)
    for name in ("accurate", "cheapest"):
        d = ch[name].reindex(keys)
        run_pairs = pd.MultiIndex.from_arrays([keys, d.to_numpy()])
        ran = run_pairs.isin(lookup.index)
        used = np.where(ran, d.to_numpy(), np.where(keys.isin(has_bposd), "bposd", best.decoder.to_numpy()))
        rows = lookup.loc[pd.MultiIndex.from_arrays([keys, used])]
        out[f"{name}_choice"] = d.to_numpy()
        out[f"{name}_fallback"] = ~ran
        out[f"{name}_good"] = rows.good.to_numpy()
        out[f"{name}_regret"] = rows.ler.to_numpy() / best.ler.to_numpy()
        out[f"{name}_time"] = rows.ms_per_shot.to_numpy() / cheapest_ms.reindex(keys).to_numpy()
    return out


def _spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def summarize(u):
    """The headline held-out metrics from units(); same definitions as evaluate.py"""
    g, l, r, t = u["regime"], u["ler"], u["regret"], u["time"]
    le = np.abs(l.pred.to_numpy() - l.y.to_numpy())
    te = np.abs(t.pred.to_numpy() - t.y.to_numpy())
    out = {
        "regime_acc": float((g.truth == g.guess).mean()),
        "regime_recall": float((g.truth & g.guess).sum() / max(g.truth.sum(), 1)),
        "regime_precision": float((g.truth & g.guess).sum() / max(g.guess.sum(), 1)),
        "ler_med": float(np.median(le)), "ler_p90": float(np.quantile(le, 0.9)),
        "ler_spearman": _spearman(l.pred, l.y), "ler_cov": float(l.inside.mean()),
        "regret_med": float(np.median(r.err)), "regret_p90": float(np.quantile(r.err, 0.9)),
        "time_med": float(np.median(te)), "time_p90": float(np.quantile(te, 0.9)),
        "time_3x": float(np.mean(te <= np.log10(3))), "time_spearman": _spearman(t.pred, t.y),
        "time_cov": float(t.inside.mean()),
    }
    for d in ("relay_bp", "bposd", "tesseract"):
        s = t[t.decoder == d]
        out[f"time_3x_{d}"] = float(np.mean(np.abs(s.pred - s.y) <= np.log10(3)))
    if "rec" in u:
        c = u["rec"]
        out.update(rec_acc_good=float(c.accurate_good.mean()), rec_cheap_good=float(c.cheapest_good.mean()),
                   rec_cheap_time=float(np.median(c.cheapest_time)), rec_acc_p90_regret=float(np.quantile(c.accurate_regret, 0.9)),
                   rec_cheap_p90_regret=float(np.quantile(c.cheapest_regret, 0.9)))
    return out


HEADLINE = ["regime_acc", "ler_med", "ler_p90", "ler_spearman", "ler_cov", "regret_med", "regret_p90", "time_med",
            "time_p90", "time_3x", "time_3x_relay_bp", "time_cov", "rec_acc_good", "rec_cheap_good", "rec_cheap_time"]
# direction: +1 higher is better, -1 lower is better
BETTER = {"regime_acc": 1, "ler_med": -1, "ler_p90": -1, "ler_spearman": 1, "ler_cov": 0, "regret_med": -1,
          "regret_p90": -1, "time_med": -1, "time_p90": -1, "time_3x": 1, "time_3x_relay_bp": 1, "time_cov": 0,
          "rec_acc_good": 1, "rec_cheap_good": 1, "rec_cheap_time": -1}


def _subset(u, sample_codes):
    """units restricted to a bootstrap sample of codes (with repetition)"""
    out = {}
    for k, frame in u.items():
        groups = frame.groupby("code").indices
        idx = np.concatenate([groups[c] for c in sample_codes if c in groups])
        out[k] = frame.iloc[idx]
    return out


def paired_bootstrap(u_base, u_alt, metrics=HEADLINE, B=300, seed=0):
    """95% interval of (alt - base) for each metric, resampling codes with replacement (paired)"""
    codes = np.array(sorted(set(u_base["regime"].code)))
    rng = np.random.default_rng(seed)
    diffs = {m: [] for m in metrics}
    for _ in range(B):
        sample = rng.choice(codes, size=len(codes), replace=True)
        a, b = summarize(_subset(u_base, sample)), summarize(_subset(u_alt, sample))
        for m in metrics:
            diffs[m].append(b[m] - a[m])
    return {m: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for m, v in diffs.items()}


def run(df, cfg, n_jobs=5, with_recommendation=True, random_state=0, prepared=None, cache_dir=None):
    """Cross-validate one Config; returns (summary dict, units, seconds). Units are cached on disk by name"""
    if cache_dir:
        path = os.path.join(cache_dir, f"{cfg.name}.pkl")
        if os.path.exists(path):
            u = pd.read_pickle(path)
            return summarize(u), u, 0.0
    t0 = time.time()
    pred = cross_val(df, cfg, random_state, n_jobs, prepared=prepared)
    u = units(df, pred, with_recommendation)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        pd.to_pickle(u, os.path.join(cache_dir, f"{cfg.name}.pkl"))
    return summarize(u), u, time.time() - t0


def load():
    df, manifest = load_dataset(DATA)
    return df, manifest
