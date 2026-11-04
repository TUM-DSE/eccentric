"""Evaluation on held-out codes: every circuit is predicted by models trained without its code.

Codes are split into GroupKFold folds, all circuits of a code in one fold; with as many folds as codes
this is leave-one-code-out. The decoder choice is scored as in ECCentric's analysis/decoding.py, on
held-out circuits with at least two measured decodes that have a time: a decoder is good enough when it is
not significantly worse than the best measured one (its ler_low <= the best's ler_high, 95% Wilson), and
the cheapest good-enough decoder is the good-enough one with the lowest ms per shot. A choice that was not
run on a circuit falls back to BP-OSD (to the best decoder run, without BP-OSD) and is counted.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .decoders import DECODERS
from .models import LER_FLOOR, circuit_features, circuit_labels, regret_rows, timeout_bound_ms, train
from .recommend import recommend

POLICIES = {"accurate": "recommend, accurate", "cheapest": "recommend, cheapest", "bposd": "always bposd",
            "tesseract": "always tesseract"}
QUANTILE_COLUMNS = ("q10", "q50", "q90")


def _ordered(decoders):
    present = set(decoders)
    return [d for d in DECODERS if d in present] + sorted(present - set(DECODERS))


def cross_val_predictions(df: pd.DataFrame, folds: int = 10, random_state: int = 0, calibrate: bool = False) -> pd.DataFrame:
    """Held-out predictions (`ModelBundle.predict` columns plus `fold`) for every circuit and every decoder
    of the table; calibrate conformalizes each fold's intervals within its own training codes"""
    circuits = circuit_features(df)
    codes = df.groupby("circuit_key", sort=False).code.first().loc[circuits.index].to_numpy()
    decoders = _ordered(df.decoder)
    out = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=min(folds, len(set(codes)))).split(circuits, groups=codes)):
        bundle = train(df[df.circuit_key.isin(circuits.index[train_idx])], random_state=random_state, calibrate=calibrate)
        out.append(bundle.predict(circuits.iloc[test_idx].reset_index(), decoders).assign(fold=fold))
    return pd.concat(out, ignore_index=True)


def _errors(predicted, true):
    """Median and p90 |log10 error| and rank correlation of log10 predictions against log10 truth"""
    err = np.abs(predicted - true)
    return {"median_abs_log10_error": float(np.median(err)), "p90_abs_log10_error": float(np.quantile(err, 0.9)),
            "spearman": float(pd.Series(predicted).corr(pd.Series(true), method="spearman"))}


def _intervals(frame, prefix, true):
    """log10 of the q10/q50/q90 columns, and the share of truths inside [q10, q90]"""
    q = np.log10(frame[[f"{prefix}_{c}" for c in QUANTILE_COLUMNS]].to_numpy(float))
    return q, float(np.mean((q[:, 0] <= true) & (true <= q[:, 2])))


def _regime(df, pred):
    labels = circuit_labels(df)
    guess = (pred.groupby("circuit_key").p_above.first().reindex(labels.index) > 0.5).to_numpy()
    truth = labels.above.to_numpy()
    tp = int((truth & guess).sum())
    return {"circuits": len(truth), "base_rate": float(truth.mean()), "accuracy": float((truth == guess).mean()),
            "recall": tp / max(int(truth.sum()), 1), "precision": tp / max(int(guess.sum()), 1)}


def _ler(df, pred):
    labels = circuit_labels(df)
    labels = labels[labels.best_status == "measured"]
    y = np.log10(np.clip(labels.best_ler.to_numpy(), LER_FLOOR, 1))
    q, coverage = _intervals(pred.groupby("circuit_key").first().reindex(labels.index), "ler", y)
    return {"circuits": len(y), **_errors(q[:, 1], y), "coverage_q10_q90": coverage}


def _regret(df, pred):
    r = regret_rows(df).merge(pred[["circuit_key", "decoder", "regret_q50"]], on=["circuit_key", "decoder"])
    r = r.dropna(subset=["regret_q50"])
    err = np.abs(np.log10(r.regret_q50.to_numpy()) - r.regret.to_numpy())
    return {"decodes": len(r), "median_abs_log10_error": float(np.median(err)),
            "p90_abs_log10_error": float(np.quantile(err, 0.9))}


def _time(df, pred):
    cols = ["circuit_key", "decoder"] + [f"ms_{c}" for c in QUANTILE_COLUMNS]
    t = df[df.ms_per_shot > 0][["circuit_key", "decoder", "ms_per_shot"]].merge(pred[cols], on=["circuit_key", "decoder"])
    unpredicted = int(t.ms_q50.isna().sum())
    t = t.dropna(subset=["ms_q50"])
    per_decoder = {}
    for name in ["all"] + _ordered(t.decoder):
        s = t if name == "all" else t[t.decoder == name]
        y = np.log10(s.ms_per_shot.to_numpy())
        q, coverage = _intervals(s, "ms", y)
        per_decoder[name] = {"decodes": len(s), **_errors(q[:, 1], y),
                             "within_3x": float(np.mean(np.abs(q[:, 1] - y) <= np.log10(3))), "coverage_q10_q90": coverage}
    timeouts = df[(df.status == "timeout") & (df.timeout_s > 0)].merge(pred[cols], on=["circuit_key", "decoder"])
    above = timeouts.ms_q50.to_numpy() >= timeout_bound_ms(timeouts.timeout_s)
    return {"decoders": per_decoder, "unpredicted": unpredicted,
            "timeouts": {"decodes": len(timeouts), "predicted_above_bound": float(above.mean()) if len(above) else None}}


def _recommendation(df, pred):
    m = df[(df.status == "measured") & (df.ms_per_shot > 0)]
    m = m[m.groupby("circuit_key").decoder.transform("size") >= 2]
    m = m[m.circuit_key.isin(set(pred.circuit_key))]
    predicted = dict(tuple(pred.groupby("circuit_key", sort=False)))
    scores = {name: {"good": [], "regret": [], "time": [], "fallbacks": 0, "chosen": {}} for name in POLICIES}
    for key, s in m.groupby("circuit_key", sort=False):
        ler, ms = dict(zip(s.decoder, s.ler)), dict(zip(s.decoder, s.ms_per_shot))
        best = s.loc[s.ler.idxmin()]
        good = s[~(s.ler_low > best.ler_high)]
        cheapest_ms = good.ms_per_shot.min()
        choices = {"accurate": recommend(predicted[key], "accurate").decoder,
                   "cheapest": recommend(predicted[key], "cheapest").decoder, "bposd": "bposd", "tesseract": "tesseract"}
        for name, d in choices.items():
            score = scores[name]
            score["chosen"][d] = score["chosen"].get(d, 0) + 1
            if d not in ler:
                score["fallbacks"] += 1
                d = "bposd" if "bposd" in ler else min(ler, key=ler.get)
            score["good"].append(d in set(good.decoder))
            score["regret"].append(ler[d] / best.ler)
            score["time"].append(ms[d] / cheapest_ms)
    policies = {POLICIES[name]: {
        "good_enough": float(np.mean(s["good"])), "median_regret": float(np.median(s["regret"])),
        "p90_regret": float(np.quantile(s["regret"], 0.9)), "median_time_vs_cheapest_good": float(np.median(s["time"])),
        "fallbacks": s["fallbacks"], "chosen": {str(d): n for d, n in sorted(s["chosen"].items(), key=lambda x: -x[1])}}
        for name, s in scores.items()}
    ruled_out = m.merge(pred[["circuit_key", "decoder", "applicable"]], on=["circuit_key", "decoder"])
    ruled_out = ruled_out[~ruled_out.applicable.astype(bool)].decoder.value_counts()
    return {"circuits": int(m.circuit_key.nunique()), "policies": policies,
            "measured_but_inapplicable": {str(d): int(n) for d, n in ruled_out.items()}}


def evaluate(df: pd.DataFrame, folds: int = 10, random_state: int = 0, calibrate: bool = False) -> dict:
    """Held-out-code metrics of the regime, LER, regret and time models and of the decoder choice; writes
    nothing"""
    pred = cross_val_predictions(df, folds, random_state, calibrate)
    codes = int(df.code.nunique())
    return {
        "setup": {"folds": int(pred.fold.nunique()), "codes": codes, "circuits": int(df.circuit_key.nunique()),
                  "decodes": len(df), "split": "leave-one-code-out" if folds >= codes else "GroupKFold over codes",
                  "calibrated": calibrate},
        "regime": _regime(df, pred),
        "ler": _ler(df, pred),
        "regret": _regret(df, pred),
        "time": _time(df, pred),
        "recommendation": _recommendation(df, pred),
    }


def _f(x, digits=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def report_markdown(metrics: dict) -> str:
    """The metrics of evaluate() as a Markdown report"""
    s, g, ler, reg, t, rec = (metrics[k] for k in ("setup", "regime", "ler", "regret", "time", "recommendation"))
    lines = [
        "# Matchmaker evaluation", "",
        f"Held-out codes: {s['split']}, {s['folds']} folds over {s['codes']} codes (all circuits of a code are held "
        f"out together); {s['circuits']} circuits, {s['decodes']} decodes. Errors are in decades (|log10 error|). "
        f"Intervals [q10, q90] {'conformalized within each fold' if s.get('calibrated') else 'not conformalized'}.", "",
        "## Regime (above threshold: best LER > 0.5)", "",
        "| circuits | above (base rate) | accuracy | recall | precision |", "|---|---|---|---|---|",
        f"| {g['circuits']} | {_pct(g['base_rate'])} | {_pct(g['accuracy'])} | {_pct(g['recall'])} | {_pct(g['precision'])} |", "",
        "## Best LER (circuits whose best decode is measured)", "",
        "| circuits | median error | p90 error | Spearman | in [q10, q90] |", "|---|---|---|---|---|",
        f"| {ler['circuits']} | {_f(ler['median_abs_log10_error'])} | {_f(ler['p90_abs_log10_error'])} | "
        f"{_f(ler['spearman'])} | {_pct(ler['coverage_q10_q90'])} |", "",
        "## Regret (a decoder's LER over the best measured)", "",
        f"{reg['decodes']} measured decodes: median error {_f(reg['median_abs_log10_error'])}, "
        f"p90 {_f(reg['p90_abs_log10_error'])}.", "",
        "## Time per shot", "",
        "| decoder | decodes | median error | p90 error | within 3x | Spearman | in [q10, q90] |",
        "|---|---|---|---|---|---|---|",
    ]
    for d, m in t["decoders"].items():
        lines.append(f"| {d} | {m['decodes']} | {_f(m['median_abs_log10_error'])} | {_f(m['p90_abs_log10_error'])} | "
                     f"{_pct(m['within_3x'])} | {_f(m['spearman'])} | {_pct(m['coverage_q10_q90'])} |")
    lines += ["", f"Timed-out decodes: {t['timeouts']['decodes']}, predicted median above their bound for "
              f"{_pct(t['timeouts']['predicted_above_bound'])}. Decodes without a prediction (decoder not in the "
              f"fold's training): {t['unpredicted']}.", "",
              "## Decoder choice", "",
              f"On {rec['circuits']} held-out circuits with at least two measured decoders. Good enough: not "
              "significantly worse than the best; regret: LER over the best's; time: over the cheapest good-enough "
              "decoder's; fallbacks: choices not run on the circuit, scored as BP-OSD.", "",
              "| policy | good enough | median regret | p90 regret | median time vs cheapest good-enough | fallbacks |",
              "|---|---|---|---|---|---|"]
    for name, p in rec["policies"].items():
        lines.append(f"| {name} | {_pct(p['good_enough'])} | {_f(p['median_regret'])} | {_f(p['p90_regret'])} | "
                     f"{_f(p['median_time_vs_cheapest_good'])} | {p['fallbacks']} |")
    lines.append("")
    for name, p in rec["policies"].items():
        if name.startswith("recommend"):
            lines.append(f"Chosen by {name}: " + ", ".join(f"{d} {n}" for d, n in p["chosen"].items()) + ".")
    if rec["measured_but_inapplicable"]:
        lines += ["", "Measured decodes that `applicable()` rules out, so never recommended: "
                  + ", ".join(f"{d} {n}" for d, n in rec["measured_but_inapplicable"].items()) + "."]
    return "\n".join(lines) + "\n"
