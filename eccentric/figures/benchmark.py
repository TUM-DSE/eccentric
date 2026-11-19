"""The experiments of the SIGMETRICS'26 paper, redrawn from the current results.

The sweep reruns the paper's topology, technology and distributed experiments with every decoder and more
shots (slices paper_topology, paper_dqc); the routing, translation and accumulated-error experiments come from
experiment_results/<experiment>. The code-size and variance experiments are not redrawn: with 1,000 shots per
configuration, the best codes show no logical error at all, so they cannot resolve the differences they are about.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import style as S

GROUP = "benchmark"
RESULTS = Path(__file__).resolve().parents[2] / "experiment_results"


def _rate(g):
    """(per-round rate, is_upper_bound) of a group of rows: the best measured rate, else the best upper bound;
    NaN when every row is at or above threshold"""
    if g["ler_per_round"].notna().any():
        return g["ler_per_round"].min(), False
    if "ler_per_round_high" in g and (g["status"] == "upper bound").any():
        return g.loc[g["status"] == "upper bound", "ler_per_round_high"].min(), True
    return np.nan, False


def _code_axis(ax, codes):
    ax.set_yticks(range(len(codes)), [S.CODE_LABELS[c] for c in codes])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(len(codes) - 0.5, -0.5)


def _dots(ax, df, value, series, colors, labels, codes):
    """One row per code, one dot per series"""
    for k, (name, color) in enumerate(zip(series, colors)):
        d = df[df["series"] == name].set_index("code").reindex(codes)
        y = np.arange(len(codes)) + (k - (len(series) - 1) / 2) * 0.18
        bound = d["is_bound"].fillna(False).astype(bool).values if "is_bound" in d else np.zeros(len(d), bool)
        ax.scatter(d[value][~bound], y[~bound], s=46, color=color, edgecolor=S.SURFACE, linewidth=1.2, zorder=3,
                   label=labels[k])
        if bound.any():
            ax.scatter(d[value][bound], y[bound], s=46, facecolor=S.SURFACE, edgecolor=color, linewidth=1.8,
                       zorder=3)
    _code_axis(ax, codes)


def _best_common_distance(t, backends):
    """Per code and backend, the best per-round rate at the largest distance every backend has"""
    rows = []
    for code, g in t.groupby("code"):
        common = set.intersection(*[set(g[g["backend"] == b]["distance"]) for b in backends])
        if not common:
            continue
        d = max(common)
        for b in backends:
            rate, bound = _rate(g[(g["backend"] == b) & (g["distance"] == d)])
            rows.append({"code": code, "series": b, "distance": d, "ler_per_round": rate, "is_bound": bound})
    return pd.DataFrame(rows)


def topology(t, reg):
    t = t[(t["slice"] == "paper_topology") & (t["error_type"] == "modsi1000") & (t["error_probability"] == 0.002)]
    backends = ["custom_full", "custom_cube", "custom_grid"]
    df = _best_common_distance(t, backends)
    codes = [c for c in S.PAPER_CODES if c in set(df["code"])]
    fig, ax = S.figure(7.5, 3.8)
    _dots(ax, df, "ler_per_round", backends, S.SERIES[:3], [S.DEVICE_LABELS[b] for b in backends], codes)
    ax.set_xscale("log")
    ax.set_xlabel("Logical error per round (best decoder)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("Connectivity: the same codes on all-to-all, 3D and 2D devices")
    piv = df.pivot(index="code", columns="series", values="ler_per_round")
    grid, cube = (piv["custom_grid"] / piv["custom_full"]).median(), (piv["custom_cube"] / piv["custom_full"]).median()
    reg.save(fig, "topology", group=GROUP, title="Connectivity",
             finding=f"Restricting connectivity raises the logical error per round by a median {grid:.0f}× on a 2D "
                     f"grid and {cube:.0f}× on a 3D cube over all-to-all: the third dimension does not help.",
             caption="SI1000 noise at p = 0.002, the largest distance that fits on all three devices, best of all "
                     "decoders, Z-basis memory; hollow markers are 95% upper bounds (no logical error seen). Codes at or "
                     "above threshold on a device are not drawn. Sweep slice paper_topology.",
             data=df.rename(columns={"series": "device"}))


def technologies(t, reg):
    pairs = [("Quantinuum Apollo", "real_apollo", "real_nsapollo", "real_apollo"),
             ("Infleqtion", "real_infleqtion", "real_nsinfleqtion", "real_infleqtion")]
    fig, axes = S.plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    frames = []
    for ax, (name, shuttle, static, noise) in zip(axes, pairs):
        x = t[(t["slice"] == "paper_topology") & (t["error_type"] == noise) & (t["operation"] == "memory")
              & (t["basis"] == "Z")]
        df = _best_common_distance(x, [shuttle, static])
        df["platform"] = name
        frames.append(df)
        codes = [c for c in S.PAPER_CODES if c in set(df["code"])]
        _dots(ax, df, "ler_per_round", [shuttle, static], S.SERIES[:2], ["With shuttling", "Without shuttling"], codes)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("Logical error per round (best decoder)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Shuttling: what moving qubits buys on ions and atoms", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.tight_layout()
    df = pd.concat(frames)
    wins = {}  # (codes where shuttling is lower, codes that can be ranked)
    for name, shuttle, static, _ in pairs:
        x = df[df["platform"] == name].set_index(["code", "series"])
        won = ranked = 0
        for code in x.index.get_level_values("code").unique():
            (a, a_bound), (b, b_bound) = [(x.loc[(code, dev), "ler_per_round"], x.loc[(code, dev), "is_bound"])
                                          if (code, dev) in x.index else (np.nan, False) for dev in (shuttle, static)]
            if pd.isna(a) and pd.isna(b) or (a_bound and b_bound):
                continue  # both at threshold, or both only bounded: no ranking
            if pd.isna(b) or (not pd.isna(a) and a < b and not (b_bound and not a_bound)):
                won, ranked = won + 1, ranked + 1  # static at threshold, or shuttling lower
            elif pd.isna(a) or (b < a and not (a_bound and not b_bound)):
                ranked += 1
        wins[name] = (won, ranked)
    reg.save(fig, "technologies", group=GROUP, title="Shuttling on trapped-ion and neutral-atom devices",
             finding=f"Shuttling lowers the logical error rate for {wins['Infleqtion'][0]} of "
                     f"{wins['Infleqtion'][1]} codes on Infleqtion and {wins['Quantinuum Apollo'][0]} of "
                     f"{wins['Quantinuum Apollo'][1]} on Quantinuum Apollo.",
             caption="Each device under its own noise model, with and without moving qubits, at the largest distance "
                     "both versions fit; best of all decoders. Hollow markers are 95% upper bounds (no logical error "
                     "seen); a code at or above threshold on one version is not drawn. Sweep slice paper_topology.",
             data=df.rename(columns={"series": "device"}))


def distributed(t, reg):
    t = t[(t["slice"] == "paper_dqc") & (t["operation"] == "memory") & (t["basis"] == "Z")]
    panels = [("IBM Flamingo", "real_flamingo", "real_flamingo_1_qpu", "3 QPUs"),
              ("IBM Loon", "real_loon", "real_loon_1_qpu", "2 QPUs")]
    noises = [("real_flamingo", "today's noise"), ("real_flamingo_downscaled", "noise ÷10")]
    fig, axes = S.plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    rows = []
    for i, (noise, nlabel) in enumerate(noises):
        for j, (name, multi, single, n_qpus) in enumerate(panels):
            x = t[t["error_type"] == noise]
            for b, series in [(multi, f"{n_qpus}, larger codes"), (single, "1 QPU")]:
                for code, g in x[x["backend"] == b].groupby("code"):
                    rate, bound = _rate(g)
                    rows.append({"panel": f"{name}, {nlabel}", "code": code, "series": series, "device": b,
                                 "noise": noise, "ler_per_round": rate, "is_bound": bound})
            df = pd.DataFrame([r for r in rows if r["panel"] == f"{name}, {nlabel}"])
            codes = [c for c in S.PAPER_CODES if c in set(df["code"])]
            ax = axes[i, j]
            _dots(ax, df, "ler_per_round", ["1 QPU", f"{n_qpus}, larger codes"], S.SERIES[:2],
                  ["1 QPU", f"{n_qpus} with remote gates"], codes)
            ax.set_xscale("log")
            ax.set_title(f"{name}, {nlabel}")
            if i == 1:
                ax.set_xlabel("Logical error per round (best distance and decoder)")
    handles = [S.plt.Line2D([], [], ls="", marker="o", ms=7, color=c) for c in S.SERIES[:2]]
    fig.legend(handles, ["1 QPU", "All QPUs, with remote gates between them"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Distributing a code over QPUs: bigger codes, noisier links", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.tight_layout()
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["panel", "code"], columns="series", values="ler_per_round")
    single_better = (piv["1 QPU"] < piv.drop(columns="1 QPU").min(axis=1)).mean() * 100
    reg.save(fig, "distributed", group=GROUP, title="Single vs multi-QPU execution",
             finding=f"Spreading a larger code over several QPUs connected by remote gates does not pay off: a single "
                     f"QPU gives the lower logical error rate in {single_better:.0f}% of code and device combinations.",
             caption="Each code at the best distance that fits on one QPU, or on all QPUs with remote gates between "
                     "them, best of all decoders. Today's IBM Flamingo noise, and the same with gate errors ÷10 and "
                     "T1/T2 ×3. Hollow markers are 95% upper bounds. Sweep slice paper_dqc.",
             data=df)


def routing(_, reg):
    frames = []
    for name, label in [("Routing_grid", "2D grid"), ("Routing_hh", "Heavy-hex")]:
        d = pd.read_csv(RESULTS / name / "results.csv")
        d = d.groupby(["code", "distance", "layout_method"])["swap_overhead_mean"].min().reset_index()
        d["device"] = label
        frames.append(d)
    df = pd.concat(frames)
    layouts = ["trivial", "dense", "sabre"]
    fig, axes = S.plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, dev in zip(axes, ["2D grid", "Heavy-hex"]):
        x = df[df["device"] == dev].rename(columns={"layout_method": "series"})
        x = x[x["swap_overhead_mean"] > 0]
        codes = [c for c in S.PAPER_CODES if c in set(x["code"])]
        _dots(ax, x, "swap_overhead_mean", layouts, S.SERIES[:3], ["Trivial layout", "Dense layout", "SABRE layout"],
              codes)
        ax.set_xscale("log")
        ax.set_title(dev)
        ax.set_xlabel("SWAP gates added by routing")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Routing: the initial layout decides how many SWAPs a code needs", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    piv = df.pivot_table(index=["device", "code"], columns="layout_method", values="swap_overhead_mean")
    piv = piv[(piv > 0).all(axis=1)]
    gain = (piv["trivial"] / piv["sabre"]).median()
    reg.save(fig, "routing", group=GROUP, title="Routing overhead by initial layout",
             finding=f"Choosing the initial layout with SABRE instead of placing qubits in order needs a median "
                     f"{gain:.1f}× fewer SWAPs; no layout removes them.",
             caption="SWAPs Qiskit's routing adds to each code's circuit on a 289-qubit 2D grid and a heavy-hex device, "
                     "best routing method for each initial layout. Codes that need no SWAPs on a device are not drawn. "
                     "Source: experiment_results/Routing_grid, Routing_hh.",
             data=df)


def translation(_, reg):
    d = pd.read_csv(RESULTS / "Translation" / "results.csv")
    d["added_per_gate"] = d["gate_overhead_mean"] / d["original_total_gates"]
    methods = ["qiskit", "qiskit_optimized", "tket_optimized"]
    labels = ["Qiskit (level 0)", "Qiskit (level 3)", "TKET (level 2)"]
    fig, axes = S.plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, gs, title in zip(axes, ["ibm_heron", "h2"], ["IBM Heron gate set", "Quantinuum H2 gate set"]):
        x = d[d["gate_set"] == gs].rename(columns={"translating_method": "series"})
        codes = [c for c in S.PAPER_CODES if c in set(x["code"])]
        _dots(ax, x, "added_per_gate", methods, S.SERIES[:3], labels, codes)
        ax.set_title(title)
        ax.set_xlabel("Gates added per original gate")
        ax.set_xlim(left=0)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Gate-set translation multiplies the circuit", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    best = d[d["translating_method"].isin(["qiskit_optimized", "tket_optimized"])].groupby(["code", "gate_set"])[
        "added_per_gate"].min()
    reg.save(fig, "translation", group=GROUP, title="Gate-set translation overhead",
             finding=f"Even with optimization, translating a code's circuit to native gates adds "
                     f"{best.min():.1f}–{best.max():.1f} gates for every gate of the circuit.",
             caption="Gates added when translating each code's circuit (all-to-all device) to the IBM Heron and "
                     "Quantinuum H2 native gate sets. Source: experiment_results/Translation.",
             data=d[["code", "gate_set", "translating_method", "original_total_gates", "gate_overhead_mean",
                     "added_per_gate"]])


def break_even(_, reg):
    d = pd.read_csv(RESULTS / "Accumulated" / "results.csv")
    d["ratio"] = d["corrected_error_rate"] / d["idle_error_rate"]
    fig, ax = S.figure(7.5, 4.2)
    for code in S.PAPER_CODES:
        c = d[d["code"] == code].sort_values("error_probability")
        c = c[c["corrected_error_rate"] > 0]
        ax.plot(c["error_probability"], c["ratio"], marker="o", ms=5, color=S.CODE_COLORS[code],
                label=S.CODE_LABELS[code])
    ax.axhline(1, color=S.INK_2, lw=1)
    ax.text(0.98, 0.04, "Above 1: an unencoded idle qubit does better", transform=ax.transAxes, ha="right",
            fontsize=8, color=S.INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p (SI1000)")
    ax.set_ylabel("Logical error / idle qubit's error")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title("Break-even: when does encoding beat leaving a qubit alone?")
    last_ahead = {}  # the highest p before the code first falls behind the idle qubit
    for code, c in d.sort_values("error_probability").groupby("code"):
        behind = c[c["ratio"] >= 1]["error_probability"]
        first_behind = behind.min() if len(behind) else np.inf
        ahead = c[c["error_probability"] < first_behind]["error_probability"]
        last_ahead[code] = ahead.max() if len(ahead) else np.nan
    best = max(last_ahead, key=lambda c: last_ahead[c])
    worst = min(last_ahead, key=lambda c: last_ahead[c])
    reg.save(fig, "break-even", group=GROUP, title="Break-even against an idle qubit",
             finding=f"Encoding pays off only at low noise: the {S.CODE_NAMES[best]} code stays ahead of an "
                     f"unencoded qubit idling for the same time up to p = {last_ahead[best]:g}, the "
                     f"{S.CODE_NAMES[worst]} code only up to p = {last_ahead[worst]:g}.",
             caption="Logical error rate of each code's memory over the idle error of one unencoded qubit for the "
                     "same duration, all-to-all device, SI1000 noise, BP-OSD, 10,000 shots. Points with no logical "
                     "error are not drawn. Source: experiment_results/Accumulated.",
             data=d[["code", "error_probability", "distance", "corrected_error_rate", "idle_error_rate", "ratio"]])


ALL = [topology, technologies, distributed, routing, translation, break_even]
