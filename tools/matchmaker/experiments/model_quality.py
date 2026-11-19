"""Model-quality experiments for qec-matchmaker (held-out-code metrics, same 10-fold GroupKFold as evaluate.py).

    python experiments/model_quality.py run hp          # hyperparameters (stage A, then B)
    python experiments/model_quality.py run mono        # monotonic constraints
    python experiments/model_quality.py run drop        # drop-one-feature retraining
    python experiments/model_quality.py run curve       # learning curve over the share of training codes
    python experiments/model_quality.py importance      # permutation importance + monotonicity probe (baseline)
    python experiments/model_quality.py table <stage>   # metrics table with paired code-bootstrap intervals

Per-unit held-out results are cached in experiments/cache/<config>.pkl; tables go to experiments/results/.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mq_core as mq  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
N_JOBS = int(os.environ.get("MQ_JOBS", 6))

C = mq.Config
LER_MONO = {f: 1 for f in ("det_fire_mean", "det_fire_median", "det_fire_p90", "error_mass_per_detector", "logical_mass")}
TIME_MONO = {"mechanisms": 1, "detectors": 1}
# every feature that moves when the noise rises / the circuit grows (the joint probe's directions)
LER_MONO_FULL = {**LER_MONO, "det_fire_max": 1, "error_mass": 1}
TIME_MONO_FULL = {f: 1 for f in ("detectors", "mechanisms", "qubits", "measurements", "two_qubit_gates", "error_mass", "ticks")}

STAGES = {
    "hp": [
        C("baseline"),
        C("lr0.03", params=dict(learning_rate=0.03)),
        C("lr0.1", params=dict(learning_rate=0.1)),
        C("it400", params=dict(max_iter=400)),
        C("leaves7", params=dict(max_leaf_nodes=7)),
        C("leaves31", params=dict(max_leaf_nodes=31)),
        C("msl10", params=dict(min_samples_leaf=10)),
        C("msl40", params=dict(min_samples_leaf=40)),
    ],
    # stage B: stage A showed fewer leaves hurts and more capacity helps time / LER a little
    "hp_b": [
        C("baseline"),
        C("it400_leaves31", params=dict(max_iter=400, max_leaf_nodes=31)),
        C("lr0.1_it400", params=dict(learning_rate=0.1, max_iter=400)),
        C("lr0.1_leaves31", params=dict(learning_rate=0.1, max_leaf_nodes=31)),
        C("lr0.1_it400_leaves31", params=dict(learning_rate=0.1, max_iter=400, max_leaf_nodes=31)),
        C("l2_0", params=dict(l2_regularization=0.0)),
    ],
    # final: capacity only where it helped (ler, time); regime and regret keep the baseline parameters.
    # base_iv is the baseline again, re-run so that its units keep q10/q90 (interval widths)
    "final": [
        C("base_iv"),
        C("cap_ler_time", per_model={m: dict(learning_rate=0.1, max_iter=400, max_leaf_nodes=31) for m in ("ler", "time")}),
        C("cap200_ler_time", per_model={m: dict(learning_rate=0.1, max_leaf_nodes=31) for m in ("ler", "time")}),
        C("cap63_ler_time", per_model={m: dict(learning_rate=0.1, max_iter=400, max_leaf_nodes=63) for m in ("ler", "time")}),
    ],
    "mono": [
        C("baseline"),
        C("mono_ler", monotone={"ler": LER_MONO}),
        C("mono_ler_regime", monotone={"ler": LER_MONO, "regime": LER_MONO}),
        C("mono_time", monotone={"time": TIME_MONO}),
        C("mono_all", monotone={"ler": LER_MONO, "regime": LER_MONO, "time": TIME_MONO}),
        C("mono_full", monotone={"ler": LER_MONO_FULL, "regime": LER_MONO_FULL, "time": TIME_MONO_FULL}),
    ],
    "curve": [C("baseline")] + [C(f"frac{f}_s{s}", code_fraction=f, subsample_seed=s)
                                for f in (0.25, 0.5, 0.75) for s in (0, 1, 2)],
}
# all training codes, but only a share of each code's circuits: the value of more results for known codes
STAGES["curve_circuits"] = [C("baseline")] + [C(f"cfrac{f}_s{s}", circuit_fraction=f, subsample_seed=s)
                                              for f in (0.25, 0.5) for s in (0, 1, 2)]
# final2: the capacity setting combined with the features drop-one suggested removing
_CAP = {m: dict(learning_rate=0.1, max_iter=400, max_leaf_nodes=31) for m in ("ler", "time")}
STAGES["final2"] = [C("base_iv"), C("cap_ler_time", per_model=_CAP),
                    C("cap_drop_em", per_model=_CAP, drop=("error_mass",)),
                    C("cap_drop_em_det", per_model=_CAP, drop=("error_mass", "detectors")),
                    C("drop_em_iv", drop=("error_mass",))]
FILLED = ["detectors", "observables", "mechanisms", "mechanism_weight_mean", "error_mass", "error_mass_per_detector",
          "det_fire_mean", "det_fire_median", "det_fire_max", "undetectable_faults", "undetectable_probability",
          "qubits", "two_qubit_gates", "swaps", "ticks", "measurements"]
STAGES["drop"] = [C("baseline")] + [C(f"drop_{f}", drop=(f,)) for f in FILLED]


def configs(stage):
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage}; one of {', '.join(STAGES)}")
    return STAGES[stage]


def run_stage(stage):
    df, _ = mq.load()
    prepared = mq.setup(df)
    rows = []
    for cfg in configs(stage):
        s, _, sec = mq.run(df, cfg, n_jobs=N_JOBS, prepared=prepared, cache_dir=CACHE)
        print(f"{cfg.name:24s} {sec:5.0f}s  " + "  ".join(f"{k}={s[k]:.3f}" for k in mq.HEADLINE), flush=True)
        rows.append({"config": cfg.name, **s})
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"{stage}.csv"), index=False)


def table(stage, names=None, base="baseline", B=300, metrics=None):
    """Markdown table: each config's metrics, with a mark where the paired code-bootstrap 95% interval of the
    difference from the base excludes 0 (+ better, - worse)"""
    metrics = metrics or mq.HEADLINE
    names = names or [c.name for c in configs(stage)]
    ub = pd.read_pickle(os.path.join(CACHE, f"{base}.pkl"))
    sb = mq.summarize(ub)
    head = "| config | " + " | ".join(metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 1)]
    for name in names:
        u = pd.read_pickle(os.path.join(CACHE, f"{name}.pkl"))
        s = mq.summarize(u)
        ci = mq.paired_bootstrap(ub, u, metrics, B=B) if name != base else {}
        cells = []
        for m in metrics:
            mark = ""
            if m in ci and mq.BETTER.get(m, 0):
                lo, hi = ci[m]
                if lo > 0 or hi < 0:
                    mark = "+" if (lo > 0) == (mq.BETTER[m] > 0) else "-"
            cells.append(f"{s[m]:.3f}{mark}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- permutation importance and monotonicity probe (on each fold's held-out codes) ----
GROUPS = {
    "size (detectors, mechanisms, qubits, measurements, two_qubit_gates, error_mass)":
        ["detectors", "mechanisms", "qubits", "measurements", "two_qubit_gates", "error_mass"],
    "noise rate (det_fire_*, error_mass_per_detector)":
        ["det_fire_mean", "det_fire_median", "det_fire_max", "error_mass_per_detector"],
}
REPEATS = 5
PROBE = {"ler": ["det_fire_mean", "det_fire_median", "error_mass_per_detector"], "time": ["mechanisms", "detectors"]}
PROBE_FACTOR = 1.25


def _permute_hook(models, meta, test, decoders, fold):
    rng = np.random.default_rng(fold)
    out = {"perm": [], "probe": []}
    targets = [(f, [f]) for f in meta["features"]] + [(g, [f for f in fs if f in meta["features"]]) for g, fs in GROUPS.items()]
    for name, cols in targets:
        for r in range(REPEATS):
            t = test.copy()
            order = rng.permutation(len(t))
            t[cols] = t[cols].to_numpy()[order]
            out["perm"].append(mq.predict(models, meta, t, decoders).assign(fold=fold, target=name, repeat=r))
    base = mq.predict(models, meta, test, decoders)
    for model, feats in PROBE.items():
        for f in feats + (["+".join(feats)] if model == "time" else []):
            t = test.copy()
            for g in f.split("+"):
                t[g] = t[g] * PROBE_FACTOR
            p = mq.predict(models, meta, t, decoders)
            col = "ler_q50" if model == "ler" else "ms_q50"
            delta = np.log10(p[col].to_numpy()) - np.log10(base[col].to_numpy())
            if model == "ler":
                delta = delta[::len(decoders)]
            delta = delta[~np.isnan(delta)]
            out["probe"].append({"fold": fold, "model": model, "feature": f, "n": len(delta),
                                 "decreases": int((delta < -1e-9).sum()), "mean_delta": float(delta.mean()),
                                 "min_delta": float(delta.min())})
    return out


def importance(cfg_name="baseline", cfg=None):
    df, _ = mq.load()
    cfg = cfg or mq.Config(cfg_name)
    t0 = time.time()
    pred, hooks = mq.cross_val(df, cfg, n_jobs=N_JOBS, permute=_permute_hook)
    print(f"cross-validation with permutations: {time.time() - t0:.0f}s", flush=True)
    ub = mq.units(df, pred)
    base = mq.summarize(ub)
    base_ll = _logloss(ub)
    perm = pd.concat([p for h in hooks for p in h["perm"]], ignore_index=True)
    rows = []
    for (target, r), p in perm.groupby(["target", "repeat"], sort=False):
        s = mq.summarize(mq.units(df, p.drop(columns=["target", "repeat"])))
        u = mq.units(df, p.drop(columns=["target", "repeat"]), with_recommendation=False)
        rows.append({"target": target, "repeat": r, **s, "regime_logloss": _logloss(u),
                     "ler_mae": float(np.mean(np.abs(u["ler"].pred - u["ler"].y))),
                     "time_mae": float(np.mean(np.abs(u["time"].pred - u["time"].y)))})
    res = pd.DataFrame(rows)
    base_row = {**base, "regime_logloss": base_ll, "ler_mae": float(np.mean(np.abs(ub["ler"].pred - ub["ler"].y))),
                "time_mae": float(np.mean(np.abs(ub["time"].pred - ub["time"].y)))}
    probe = pd.DataFrame([p for h in hooks for p in h["probe"]])
    os.makedirs(RESULTS, exist_ok=True)
    res.to_csv(os.path.join(RESULTS, f"importance_{cfg.name}.csv"), index=False)
    probe.to_csv(os.path.join(RESULTS, f"probe_{cfg.name}.csv"), index=False)
    with open(os.path.join(RESULTS, f"importance_{cfg.name}_base.json"), "w") as f:
        json.dump(base_row, f, indent=1)
    print(importance_table(cfg.name))
    print(probe_table(cfg.name))


def _logloss(u):
    g = u["regime"]
    p = np.clip(g.p.to_numpy(float), 1e-6, 1 - 1e-6)
    t = g.truth.to_numpy(bool)
    return float(-np.mean(np.where(t, np.log(p), np.log(1 - p))))


IMP_COLS = [("regime_acc", -1), ("regime_logloss", 1), ("ler_mae", 1), ("ler_med", 1), ("time_mae", 1), ("time_3x", -1),
            ("rec_acc_good", -1), ("rec_cheap_good", -1)]


def importance_table(name="baseline"):
    """Mean increase in error from permuting a feature (sign: positive = the feature helps)"""
    res = pd.read_csv(os.path.join(RESULTS, f"importance_{name}.csv"))
    with open(os.path.join(RESULTS, f"importance_{name}_base.json")) as f:
        base = json.load(f)
    agg = []
    for target, g in res.groupby("target", sort=False):
        row = {"feature": target}
        for col, sign in IMP_COLS:
            d = sign * (g[col] - base[col])
            row[col] = d.mean()
            row[col + "_sd"] = d.std(ddof=1)
        agg.append(row)
    agg = pd.DataFrame(agg).sort_values("ler_mae", ascending=False)
    lines = ["| permuted | " + " | ".join(c for c, _ in IMP_COLS) + " |", "|" + "---|" * (len(IMP_COLS) + 1)]
    for _, r in agg.iterrows():
        lines.append(f"| {r.feature} | " + " | ".join(f"{r[c]:+.3f}" for c, _ in IMP_COLS) + " |")
    return "\n".join(lines)


def probe_table(name="baseline"):
    p = pd.read_csv(os.path.join(RESULTS, f"probe_{name}.csv"))
    g = p.groupby(["model", "feature"], sort=False).agg(n=("n", "sum"), decreases=("decreases", "sum"),
                                                        mean_delta=("mean_delta", "mean"), min_delta=("min_delta", "min"))
    lines = [f"| model | feature x{PROBE_FACTOR} | predictions | share decreasing | mean change (decades) | worst (decades) |",
             "|---|---|---|---|---|---|"]
    for (model, feat), r in g.iterrows():
        lines.append(f"| {model} | {feat} | {r.n} | {r.decreases / r.n:.1%} | {r.mean_delta:+.3f} | {r.min_delta:+.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "run":
        run_stage(sys.argv[2])
    elif cmd == "table":
        print(table(sys.argv[2]))
    elif cmd == "importance":
        importance(*(sys.argv[2:3] or ["baseline"]))
    else:
        print(__doc__)
