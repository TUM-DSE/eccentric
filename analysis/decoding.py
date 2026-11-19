"""The numbers behind analysis/DECODING.md: which decoder for which code, accuracy against time, and
whether the choice can be automated. python analysis/decoding.py

Reads experiment_results/Sweep/results.csv. A circuit is one configuration (code, schedule, distance,
rounds, device, noise model, compiler setting), decoded by several decoders on the same shots. Only
decodes with >= 100 logical errors ("measured") are compared, and a decoder counts as better or worse
than another only when their 95% Wilson intervals do not overlap. Time per shot is the decode time
divided by the shots, on one core, including the decoder's construction.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.export_web import family  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
CONFIG = ["slice", "code", "schedule", "distance", "cycles", "backend", "backend_size", "error_type",
          "error_probability", "layout_method", "routing_method", "translating_method"]
MATCHABLE = {"Surface", "Heavy-hex", "Bacon-Shor", "Repetition", "XZZX surface", "Toric", "Honeycomb",
             "Planar honeycomb", "Hyperbolic Floquet", "Semi-hyperbolic Floquet"}
BUDGETS_MS = [0.1, 1, 10, 100, 1000, 10000]


def load():
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "experiment_results", "Sweep", "results.csv"), low_memory=False)
    df["fam"] = df.code.map(lambda c: family(c)[0])
    df["group"] = df.fam.map(lambda f: "matchable" if f in MATCHABLE else "color" if f == "Color"
                             else "Steane" if f == "Concatenated Steane" else "qLDPC")
    df["ms"] = 1000 * df.decode_seconds / df.shots
    df["cid"] = df[CONFIG].astype(str).agg("|".join, axis=1)
    return df


def section(title):
    print(f"\n### {title}\n")


def circuits(m):
    """Per circuit: the decoders not significantly worse than the best ("good enough"), the cheapest
    of them, and the best"""
    out = []
    for cid, s in m.groupby("cid"):
        if len(s) < 2:
            continue
        best = s.loc[s.logical_error_rate.idxmin()]
        good = s[~(s.ler_low > best.ler_high)]
        cheapest = good.loc[good.ms.idxmin()]
        r = s.iloc[0]
        out.append(dict(cid=cid, code=r.code, fam=r.fam, group=r.group, detectors=r.detectors, k=r.observables,
                        rounds=r.rounds, routed=float(r.swaps > 0), device_noise=float(r.error_type not in ("sd6", "modsi1000")),
                        best=best.decoder, cheapest_good=cheapest.decoder, good=set(good.decoder),
                        ler={d: v for d, v in zip(s.decoder, s.logical_error_rate)},
                        ms={d: v for d, v in zip(s.decoder, s.ms)}, best_ler=best.logical_error_rate,
                        # What a short pilot run with the fastest decoder tells about the noise regime
                        pilot=s.loc[s.ms.idxmin()].logical_error_rate))
    return pd.DataFrame(out)


# A small classification tree (CART, Gini), so the selection needs nothing beyond numpy
def gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return 1 - (p * p).sum()


def grow(X, y, depth, min_leaf):
    values, counts = np.unique(y, return_counts=True)
    leaf = values[counts.argmax()]
    if depth == 0 or len(y) < 2 * min_leaf or len(values) == 1:
        return leaf
    best = None
    for j in range(X.shape[1]):
        for t in np.unique(X[:, j])[:-1]:
            left = X[:, j] <= t
            if left.sum() < min_leaf or (~left).sum() < min_leaf:
                continue
            score = left.mean() * gini(y[left]) + (~left).mean() * gini(y[~left])
            if best is None or score < best[0]:
                best = (score, j, t, left)
    if best is None or best[0] >= gini(y):
        return leaf
    _, j, t, left = best
    return (j, t, grow(X[left], y[left], depth - 1, min_leaf), grow(X[~left], y[~left], depth - 1, min_leaf))


def predict(tree, x):
    while isinstance(tree, tuple):
        j, t, left, right = tree
        tree = left if x[j] <= t else right
    return tree


def show(tree, names, indent="  "):
    if not isinstance(tree, tuple):
        return f"{indent}-> {tree}\n"
    j, t, left, right = tree
    return (f"{indent}if {names[j]} <= {t:g}:\n" + show(left, names, indent + "    ")
            + f"{indent}else:\n" + show(right, names, indent + "    "))


def evaluate(c, choose):
    """Share of circuits where the choice is good enough, its LER regret against the best, and its time
    against the cheapest good-enough decoder; a choice not run on a circuit falls back to BP-OSD"""
    ok, regret, cost, total = [], [], [], 0.0
    for _, r in c.iterrows():
        d = choose(r)
        if d not in r.ler:
            d = "bposd" if "bposd" in r.ler else min(r.ler, key=r.ler.get)
        ok.append(d in r.good)
        regret.append(r.ler[d] / r.best_ler)
        cost.append(r.ms[d] / r.ms[r.cheapest_good])
        total += r.ms[d]
    return dict(good_enough=np.mean(ok), median_regret=np.median(regret), p90_regret=np.quantile(regret, 0.9),
                median_time_vs_cheapest=np.median(cost), total_time_h=total / 3.6e6)


def main():
    df = load()
    # A few refined results carry no decode time (their base result predates the field)
    m = df[(df.status == "measured") & df.ms.notna()].copy()

    section("Decoders run per code family, and how their decodes ended")
    print(df[df.status != "does not fit"].pivot_table(index="fam", columns="decoder", values="code", aggfunc="size", fill_value=0).to_string())
    print(df[df.status != "does not fit"].groupby(["decoder", "status"]).size().unstack(fill_value=0).to_string())

    section("Accuracy per family: median and 90th-percentile LER against the circuit's best, share significantly worse, median ms per shot")
    m["n_dec"] = m.groupby("cid").decoder.transform("size")
    mm = m[m.n_dec >= 2].copy()
    mm["regret"] = mm.logical_error_rate / mm.groupby("cid").logical_error_rate.transform("min")
    best_high = mm.loc[mm.groupby("cid").logical_error_rate.idxmin()].set_index("cid").ler_high
    mm["sig_worse"] = mm.ler_low > mm.cid.map(best_high)
    print(mm.groupby(["fam", "decoder"]).agg(n=("regret", "size"), median_regret=("regret", "median"),
                                             p90_regret=("regret", lambda s: s.quantile(0.9)), sig_worse=("sig_worse", "mean"),
                                             ms=("ms", "median")).round(3).to_string())

    section("Head to head on the same circuits: A significantly better / tie / worse than B, median LER and time A/B")
    w = m.pivot_table(index="cid", columns="decoder", values=["logical_error_rate", "ler_low", "ler_high", "ms"], aggfunc="first")
    grp = m.groupby("cid").group.first()
    rows = []
    for a, b in itertools.combinations(sorted(m.decoder.unique()), 2):
        sub = w.dropna(subset=[("logical_error_rate", a), ("logical_error_rate", b)])
        for g, s in sub.groupby(grp):
            better = int((s[("ler_high", a)] < s[("ler_low", b)]).sum())
            worse = int((s[("ler_low", a)] > s[("ler_high", b)]).sum())
            rows.append((g, a, b, len(s), better, len(s) - better - worse, worse,
                         (s[("logical_error_rate", a)] / s[("logical_error_rate", b)]).median(), (s[("ms", a)] / s[("ms", b)]).median()))
    print(pd.DataFrame(rows, columns=["group", "A", "B", "n", "A better", "tie", "A worse", "LER A/B", "time A/B"]).round(3).to_string(index=False))

    section("Time per shot (ms) by circuit size (detectors), and its growth exponent")
    m["size"] = pd.cut(m.detectors, [0, 100, 300, 1000, 3000, 1e5], labels=["<100", "100-300", "300-1k", "1k-3k", ">3k"])
    print(m.pivot_table(index="decoder", columns="size", values="ms", aggfunc="median", observed=False).round(2).to_string())
    for d, s in m[m.ms > 0].groupby("decoder"):
        print(f"  {d}: time ~ detectors^{np.polyfit(np.log(s.detectors), np.log(s.ms), 1)[0]:.2f} (n = {len(s)})")

    section("Time per shot (ms) by LER regime, same decoder")
    m["regime"] = pd.cut(m.logical_error_rate, [0, 1e-2, 0.3, 1.01], labels=["<1%", "1-30%", ">30%"])
    print(m.pivot_table(index="decoder", columns="regime", values="ms", aggfunc="median", observed=False).round(2).to_string())

    section("Pareto front: share of circuits (>= 3 measured decoders) on which a decoder is not dominated")
    seen, front = {}, {}
    for _, s in m.groupby("cid"):
        if len(s) < 3:
            continue
        for i, r in s.iterrows():
            dominated = any(o.ms < r.ms and not (o.ler_low > r.ler_high) for j, o in s.iterrows() if j != i)
            seen[r.decoder] = seen.get(r.decoder, 0) + 1
            front[r.decoder] = front.get(r.decoder, 0) + (not dominated)
    print({d: f"{front[d]}/{seen[d]} = {front[d] / seen[d]:.0%}" for d in sorted(seen)})

    c = circuits(m)
    section("Best decoder under a time budget per shot: share of circuits with any decoder within budget, median LER regret")
    for budget in BUDGETS_MS:
        reg, any_ = [], 0
        for _, r in c.iterrows():
            within = {d: v for d, v in r.ler.items() if r.ms[d] <= budget}
            if within:
                any_ += 1
                reg.append(min(within.values()) / r.best_ler)
        print(f"  {budget:>7g} ms: {any_ / len(c):.0%} of circuits, median regret {np.median(reg):.2f}, p90 {np.quantile(reg, 0.9):.2f}")

    section("Automatic choice: policies, leave-one-code-out for the tree")
    print("circuits:", len(c), "| best decoder:", c.best.value_counts().to_dict(), "| cheapest good-enough:", c.cheapest_good.value_counts().to_dict())
    names = ["matchable", "color", "detectors", "k", "rounds", "routed", "device_noise", "pilot_ler"]

    def features(r):
        return [r.group == "matchable", r.group == "color", r.detectors, r.k, r.rounds, r.routed, r.device_noise, r.pilot]

    X = np.array([features(r) for _, r in c.iterrows()], dtype=float)
    policies = {
        "always BP-OSD": lambda r: "bposd",
        "always Relay-BP (BP-OSD if not run)": lambda r: "relay_bp",
        "always Tesseract": lambda r: "tesseract",
        "fastest: MWPM / Chromobius / Relay-BP": lambda r: "mwpm" if "mwpm" in r.ler else "chromobius" if "chromobius" in r.ler else "relay_bp",
        "rule: Tesseract below 300 detectors, else MWPM if matchable, else BP-OSD":
            lambda r: "tesseract" if r.detectors < 300 else "mwpm" if r.group == "matchable" else "bposd",
        "oracle: cheapest good-enough": lambda r: r.cheapest_good,
    }
    for target in ("cheapest_good", "best"):
        preds = {}
        for code in c.code.unique():
            train = (c.code != code).values
            tree = grow(X[train], c[target].values[train], depth=4, min_leaf=15)
            for i in np.where(~train)[0]:
                preds[c.cid.iloc[i]] = predict(tree, X[i])
        policies[f"tree (leave-one-code-out) for the {target.replace('_', ' ')}"] = lambda r, p=preds: p[r.cid]
    print(pd.DataFrame({name: evaluate(c, f) for name, f in policies.items()}).T.round(3).to_string())
    print("\nthe tree for the cheapest good-enough decoder, trained on all circuits:")
    print(show(grow(X, c.cheapest_good.values, depth=3, min_leaf=15), names))

    section("Where decode CPU went")
    s = df[df.status.isin(["measured", "low statistics", "upper bound", "timeout"])].copy()
    s["ler"] = s.logical_error_rate.fillna(s.ler_high)
    total = s.decode_seconds.sum()
    for lo in (0.3, 0.5, 0.9):
        print(f"  LER > {lo}: {s[s.ler > lo].decode_seconds.sum() / total:.0%} of decode CPU, {(s.ler > lo).mean():.0%} of decodes")
    top = s.decode_seconds.sort_values(ascending=False)
    print(f"  top 1% of decodes: {top.head(len(top) // 100).sum() / total:.0%} of decode CPU; top 10%: {top.head(len(top) // 10).sum() / total:.0%}")
    print("  share by decoder:", (s.groupby("decoder").decode_seconds.sum() / total).round(2).to_dict())


if __name__ == "__main__":
    main()
