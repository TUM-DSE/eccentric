"""The numbers behind analysis/INSIGHTS.md: python analysis/insights.py [<old results.csv>]

Reads experiment_results/Sweep/results.csv (and the routing and variance experiments). The optional
argument is a sweep table collected before the waiting-qubit fix (CHANGELOG fix 9), for the
before/after comparison. The logical error rate (LER) is per shot over all rounds; where no error was
seen, the 95% upper bound stands in. "Best" means the lowest LER over decoders and schedules.
"""
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiment_results")
CONFIG = ["slice", "code", "schedule", "distance", "cycles", "backend", "backend_size", "error_type",
          "error_probability", "layout_method", "routing_method", "translating_method"]
HAS_RATE = ["measured", "low statistics", "upper bound"]


def load():
    df = pd.read_csv(os.path.join(RESULTS, "Sweep", "results.csv"), low_memory=False)
    df["ler"] = df.logical_error_rate.fillna(df.ler_high)
    return df


def best_per_config(df, by=CONFIG):
    """The row with the lowest LER of each configuration (over decoders, and schedules if not in by)"""
    rated = df[df.status.isin(HAS_RATE)]
    return rated.loc[rated.groupby(by, dropna=False).ler.idxmin()]


def section(title):
    print(f"\n### {title}\n")


def thresholds(b, by):
    t = b.groupby(by).agg(configs=("ler", "size"), below_1e2=("ler", lambda s: (s < 1e-2).sum()),
                          below_1e3=("ler", lambda s: (s < 1e-3).sum()), below_1e4=("ler", lambda s: (s < 1e-4).sum()),
                          best=("ler", "min"))
    t["best_code"] = b.loc[b.groupby(by).ler.idxmin()].set_index(by).code
    return t


def main():
    df = load()
    section("Coverage")
    print(df.status.value_counts().to_string())
    print("codes", df.code.nunique(), "devices", df.backend.nunique(), "noise models", df.error_type.nunique())

    by_code = ["slice", "code", "backend", "error_type", "error_probability"]
    best = best_per_config(df, by_code)
    for sl in ["qldpc_ibm", "bb_native", "qldpc_bicycle", "qldpc_bicycle_chain", "qldpc_ibm_projected", "qldpc_atoms",
               "readout_700", "willow_surface"]:
        section(f"{sl}: codes below 1e-2/1e-3/1e-4 per device and noise model")
        print(thresholds(best[best.slice == sl], ["backend", "error_type"]).to_string())

    section("Native layout against routing onto Flamingo (bivariate bicycle codes)")
    native = best[(best.slice == "bb_native") & (best.backend == "native")].set_index(["code", "error_type"]).ler
    routed = best[(best.slice == "qldpc_ibm") & (best.backend == "real_flamingo")].set_index(["code", "error_type"]).ler
    swaps = df[(df.slice == "qldpc_ibm") & (df.backend == "real_flamingo")].groupby("code").swaps.min()
    t = pd.DataFrame({"native": native, "routed": routed}).dropna()
    t["routed/native"] = t.routed / t.native
    print(t.join(swaps, on="code").to_string())

    section("SWAP share of two-qubit gates, per device (median over compiled circuits)")
    circuits = df.drop_duplicates(["run", "job_id"]).dropna(subset=["swaps"])
    circuits = circuits[circuits.two_qubit_gates > 0]
    share = (circuits.swaps / circuits.two_qubit_gates).groupby(circuits.backend).median().sort_values()
    print(share.round(2).to_string())

    section("Readout 700 ns against 1.56 us (same code, device, decoder; both measured)")
    r7 = df[(df.slice == "readout_700") & (df.status == "measured")].copy()
    r7["base"] = r7.error_type.str.replace("_m700", "")
    base = df[df.slice.isin(["qldpc_ibm", "qldpc_bicycle"]) & (df.status == "measured")]
    j = r7.merge(base, left_on=["code", "schedule", "backend", "base", "decoder"],
                 right_on=["code", "schedule", "backend", "error_type", "decoder"], suffixes=("_700", "_156"))
    j["ratio"] = j.ler_156 / j.ler_700
    unsat = j[j.ler_156 < 0.3]
    print(pd.DataFrame({"all": j.groupby("base").ratio.median(), "n": j.groupby("base").size(),
                        "LER<0.3": unsat.groupby("base").ratio.median(), "n<0.3": unsat.groupby("base").size()}).round(2).to_string())

    section("Decoders: wins and median LER and time relative to BP-OSD on the same circuit (measured only)")
    measured = df[df.status == "measured"]
    wins = measured.loc[measured.groupby(CONFIG, dropna=False).logical_error_rate.idxmin()].decoder.value_counts()
    ref = measured[measured.decoder == "bposd"][CONFIG + ["logical_error_rate", "decode_seconds"]]
    rel = measured.merge(ref, on=CONFIG, suffixes=("", "_bposd"))
    rel["ler_ratio"] = rel.logical_error_rate / rel.logical_error_rate_bposd
    rel["time_ratio"] = rel.decode_seconds / rel.decode_seconds_bposd
    print(pd.DataFrame({"wins": wins, "rows": measured.decoder.value_counts(),
                        "LER vs bposd": rel.groupby("decoder").ler_ratio.median(),
                        "time vs bposd": rel.groupby("decoder").time_ratio.median()}).round(3).to_string())
    print("\nnot decoded:", df[df.status.isin(["timeout", "decoder not applicable"])].groupby(["status", "decoder"]).size().to_string())

    section("Circuits with undetectable logical faults (circuit distance 1)")
    u = circuits[circuits.undetectable_faults > 0]
    print(u.groupby(["code", "schedule", "backend"]).undetectable_probability.agg(["size", "min", "max"]).to_string())

    section("Paper codes on sparse topologies (modsi1000), best LER")
    pt = best_per_config(df[df.slice == "paper_topology"], ["code", "distance", "backend", "error_probability"])
    print(pt.pivot_table(index=["code", "distance", "error_probability"], columns="backend", values="ler").to_string())

    section("Paper DQC: multi-QPU against one QPU (bposd)")
    dqc = best_per_config(df[df.slice == "paper_dqc"], ["code", "distance", "backend", "error_type"])
    print(dqc.pivot_table(index=["code", "distance", "error_type"], columns="backend", values="ler").to_string())

    section("Reference (all-to-all, SD6): best LER at p = 1e-3 with qubits and logical qubits")
    ref = best_per_config(df[(df.slice == "reference") & (df.error_probability == 0.001)], ["code", "distance"])
    ref["k"] = ref.observables
    print(ref.sort_values("ler")[["code", "distance", "k", "circuit_qubits", "decoder", "ler", "ler_per_round", "status"]].to_string(index=False))

    section("Reference: per-round LER against distance (Lambda = eps_d / eps_{d+2})")
    fam = best_per_config(df[(df.slice == "reference") & df.code.isin(["surface", "color", "hh", "bacon"])],
                          ["code", "distance", "error_probability"])
    print(fam.pivot_table(index=["code", "error_probability"], columns="distance", values="ler_per_round").to_string())

    section("Willow: device layout against native, measured and spec-sheet noise")
    w = best_per_config(df[df.slice == "willow_surface"], ["distance", "cycles", "backend", "error_type"])
    print(w.pivot_table(index=["distance", "cycles"], columns=["backend", "error_type"], values="ler").to_string())

    section("Neutral atoms and ions: best LER per code")
    at = best[best.slice == "qldpc_atoms"]
    print(at.pivot_table(index="code", columns="backend", values="ler").sort_values("real_apollo").to_string())

    section("Bicycle modules: best LER per code")
    bi = best[best.slice == "qldpc_bicycle"]
    print(bi.pivot_table(index="code", columns=["backend", "error_type"], values="ler").to_string())

    section("Routing experiments (SWAP overhead, mean over samples)")
    for name in ["Routing_grid", "Routing_grid_placed", "Routing_hh", "Routing_hh_placed", "Routing_diamond"]:
        r = pd.read_csv(os.path.join(RESULTS, name, "results.csv"))
        print(f"\n{name}")
        print(r.pivot_table(index=["code", "distance"], columns="routing_method", values="swap_overhead_mean").round(1).to_string())

    if len(sys.argv) > 1:
        section("Waiting-qubit fix (CHANGELOG 9): new against old LER, same configuration and decoder")
        old = pd.read_csv(sys.argv[1], low_memory=False)
        old["ler"] = old.logical_error_rate.fillna(old.ler_high)
        keys = CONFIG + ["decoder"]
        a, b = old.copy(), df.copy()
        for d in (a, b):
            for k in keys:
                d[k] = d[k].astype(str)
        m = a.merge(b, on=keys, suffixes=("_old", "_new"))
        both = m[(m.status_old == "measured") & (m.status_new == "measured")].copy()
        both["ratio"] = both.ler_new / both.ler_old
        unsat = both[both.ler_old < 0.3]
        print(pd.DataFrame({"n": both.groupby("error_type").size(), "median": both.groupby("error_type").ratio.median(),
                            "p90": both.groupby("error_type").ratio.quantile(0.9),
                            "n (old LER<0.3)": unsat.groupby("error_type").size(),
                            "median (old LER<0.3)": unsat.groupby("error_type").ratio.median()}).round(2).to_string())
        rated = m[m.status_old.isin(HAS_RATE) & m.status_new.isin(HAS_RATE)]
        cfg = [k for k in CONFIG]
        for et, g in rated.groupby("error_type"):
            lo, ln = g.groupby(cfg).ler_old.min(), g.groupby(cfg).ler_new.min()
            print(f"{et}: {len(lo)} configs, below 1e-2 {np.mean(lo < 1e-2):.0%} -> {np.mean(ln < 1e-2):.0%}, "
                  f"below 1e-3 {np.mean(lo < 1e-3):.0%} -> {np.mean(ln < 1e-3):.0%}")


if __name__ == "__main__":
    main()
