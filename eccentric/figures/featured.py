"""Featured findings of the ECCentric dataset (experiment_results/Sweep/results.csv)"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .. import dataset
from . import style as S

GROUP = "featured"

DECODER_LABELS = {
    "mwpm": "MWPM", "mwpm_corr": "Correlated MWPM", "bposd": "BP-OSD", "bposd_tuned": "BP-OSD (tuned)",
    "bplsd": "BP-LSD", "belief_find": "BeliefFind", "relay_bp": "Relay-BP", "tesseract": "Tesseract",
    "gari": "GARI", "gari_pp": "GARI++", "chromobius": "Chromobius", "mwpf": "MWPF",
    "hier_bposd": "Level-by-level BP-OSD", "hier_bposd_hard": "Level-by-level BP-OSD (hard)",
}


def _memory_z(t):
    return t[(t["operation"] == "memory") & (t["basis"] == "Z")]


def _best_rate(rows):
    """Lowest logical error rate of a group of rows, falling back on the lowest upper bound"""
    measured = rows["logical_error_rate"].dropna()
    if len(measured):
        return measured.min(), False
    return rows["ler_high"].min(), True


def routing_vs_native(t, reg):
    """Bivariate bicycle codes on their native layout against routed onto IBM Flamingo or Loon, same noise"""
    t = _memory_z(t)
    t = t[(t["error_type"] == "real_flamingo_downscaled") & t["code"].str.match(r"bb\d+$")]
    rows = []
    for code, g in t.groupby("code"):
        native, routed = g[g["backend"] == "native"], g[g["backend"].isin(["real_flamingo", "real_loon"])]
        if native.empty or routed.empty:
            continue
        (n, n_bound), (r, r_bound) = _best_rate(native), _best_rate(routed)
        rows.append({"code": code, "native_ler": n, "native_is_bound": n_bound, "routed_ler": r,
                     "routed_is_bound": r_bound, "factor": r / n})
    df = pd.DataFrame(rows).sort_values("factor")
    fig, ax = S.figure(7.5, 3.6)
    y = np.arange(len(df))
    ax.hlines(y, df["native_ler"], df["routed_ler"], color=S.NEUTRAL, lw=2, zorder=1)
    ax.scatter(df["native_ler"], y, s=60, color=S.SERIES[0], zorder=3, edgecolor=S.SURFACE, linewidth=1.5,
               label="Native layout (no SWAPs)")
    ax.scatter(df["routed_ler"], y, s=60, color=S.SERIES[1], zorder=3, edgecolor=S.SURFACE, linewidth=1.5,
               label="Routed onto IBM Flamingo / Loon")
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(r["routed_ler"] * 1.35, yi, f"{r['factor']:,.0f}×", va="center", fontsize=9, color=S.INK_2)
    ax.set_yticks(y, [c.upper() if c == "gross" else c for c in df["code"]])
    ax.set_xscale("log")
    ax.set_xlim(right=8)
    ax.set_xlabel("Logical error rate (d rounds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Same codes, same noise: routing costs up to five orders of magnitude")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    reg.save(fig, "routing-vs-native", group=GROUP, title="Routing vs native layout for bivariate bicycle codes",
             finding=f"Routed onto IBM's heavy-hex devices, bivariate bicycle codes are {df['factor'].min():,.0f}× to "
                     f"{df['factor'].max():,.0f}× worse than on a layout that gives each of their checks its couplers.",
             caption="Best logical error rate over all decoders, Z-basis memory, Flamingo noise with gate errors ÷10 and "
                     "T1/T2 ×3 on both. Routed: the better of IBM Flamingo (3 QPUs) and Loon (2 QPUs).",
             data=df)


def decoders(t, reg):
    """How often each decoder is within statistical error of the best one, against its median time per shot"""
    t = t.copy()
    t["seconds_per_shot"] = t["decode_seconds"] / t["shots"]
    t["lo"] = t["ler_low"].fillna(0.0)
    keys = list(dataset.CONFIG)
    g = t.groupby(keys, dropna=False)
    t = t[g["decoder"].transform("nunique") >= 3]
    g = t.groupby(keys, dropna=False)
    best_high = g["ler_high"].transform("min")
    best = g["logical_error_rate"].transform("min")
    t = t[best.isna() | (best < 0.4)]  # below threshold: the choice of decoder matters
    t = t.assign(good=t["lo"] <= best_high.loc[t.index])
    n_configs = t.groupby(keys, dropna=False).ngroups
    df = (t.groupby("decoder").agg(configurations=("good", "size"), good_enough=("good", "mean"),
                                   median_ms_per_shot=("seconds_per_shot", "median"))
          .reset_index())
    df["median_ms_per_shot"] *= 1000
    df["good_enough"] *= 100
    df = df[df["configurations"] >= 100].sort_values("good_enough")
    df["label"] = df["decoder"].map(DECODER_LABELS)
    fig, ax = S.figure(7.5, 4.6)
    ax.scatter(df["median_ms_per_shot"], df["good_enough"], s=64, color=S.SERIES[0], edgecolor=S.SURFACE,
               linewidth=1.5, zorder=3)
    offsets = {"relay_bp": (7, 4), "bposd_tuned": (7, -11), "gari_pp": (-8, 6), "gari": (7, -9)}
    for _, r in df.iterrows():
        dx, dy = offsets.get(r["decoder"], (7, -3))
        ax.annotate(r["label"], (r["median_ms_per_shot"], r["good_enough"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9, color=S.INK, ha="right" if dx < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(df["median_ms_per_shot"].min() / 3, df["median_ms_per_shot"].max() * 6)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Median decoding time per shot (ms, one core)")
    ax.set_ylabel("Configurations where it is as good as the best (%)")
    by = df.set_index("decoder")
    bposd, tess = by.loc["bposd", "good_enough"], by.loc["tesseract", "good_enough"]
    speed = by.loc["bposd", "median_ms_per_shot"] / by.loc["tesseract", "median_ms_per_shot"]
    ax.set_title(f"BP-OSD, the go-to qLDPC decoder, is as good as the best only {bposd:.0f}% of the time")
    reg.save(fig, "decoders", group=GROUP, title="Decoder accuracy against decoding time",
             finding=f"BP-OSD, the usual choice for qLDPC codes, is as accurate as the best decoder on only {bposd:.0f}% "
                     f"of configurations; Tesseract on {tess:.1f}%, and at {speed:.1f}× less median time per shot.",
             caption=f"{n_configs:,} configurations below threshold (best logical error rate < 0.4), each decoded by at "
                     "least three decoders. A decoder counts as good enough when its 95% interval reaches the best "
                     "decoder's. Decoders with fewer than 100 configurations are left out. Tesseract's timeouts (no "
                     "result) are not counted, which favours it on the largest circuits.",
             data=df, columns=["decoder", "label", "configurations", "good_enough", "median_ms_per_shot"])


NOISE_DEVICES = ["real_willow", "real_infleqtion", "real_harvard", "real_bicycle", "real_loon", "real_flamingo",
                 "real_flamingo_1_qpu", "real_torino"]


def below_threshold(t, reg):
    """Error suppression Λ (d = 3 to 5) of the surface and color codes as each device's noise is scaled"""
    t = _memory_z(t)
    t = t[(t["slice"] == "noise_scaling") & t["code"].isin(["surface", "color"]) & (t["cycles"] == t["distance"])]
    t = t.assign(scale=t["error_type"].str.extract(r"_x([\d.]+)$")[0].astype(float).fillna(1.0))
    best = t.groupby(["backend", "code", "scale", "distance"])["ler_per_round"].min().unstack("distance")
    best["lambda"] = best[3] / best[5]
    df = best["lambda"].reset_index()
    fig, axes = S.plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, code in zip(axes, ["surface", "color"]):
        for i, dev in enumerate(NOISE_DEVICES):
            d = df[(df["code"] == code) & (df["backend"] == dev)].sort_values("scale")
            ax.plot(d["scale"], d["lambda"], marker="o", ms=5, color=S.SERIES[i], label=S.DEVICE_LABELS[dev])
        ax.axhline(1, color=S.INK_2, lw=1)
        ax.axhspan(0.2, 1, color=S.GRID, alpha=0.5, lw=0)
        ax.text(0.105, 0.32, "above threshold: larger codes are worse", fontsize=8, color=S.INK_2, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 70)
        ax.set_xticks([0.1, 0.25, 0.5, 1], ["×0.1", "×0.25", "×0.5", "today"])
        ax.minorticks_off()
        ax.set_yticks([0.5, 1, 3, 10, 30], ["0.5", "1", "3", "10", "30"])
        ax.set_xlabel("Device noise, scaled")
        ax.set_title(f"{S.CODE_LABELS[code]} code")
    axes[0].set_ylabel("Error suppression Λ (d = 3 → 5)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    today = df[df["scale"] == 1.0]
    below = {c: int((today[today["code"] == c]["lambda"] > 1).sum()) for c in ["surface", "color"]}
    fig.suptitle("Below threshold today? The surface code mostly is, the color code mostly is not",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    reg.save(fig, "below-threshold", group=GROUP, title="Error suppression as device noise is scaled",
             finding=f"At today's noise, the surface code is below threshold on {below['surface']} of 8 device models "
                     f"and the color code on {below['color']}. IBM Heron needs about 40% of its noise for either.",
             caption="Λ is the per-round logical error rate at d = 3 divided by that at d = 5 (d rounds, best decoder); "
                     "Λ > 1 means a larger code helps. Every error probability of the device's noise model is scaled "
                     "by the factor on the x-axis; the compiled circuit is the same at every scale.",
             data=df.rename(columns={"backend": "device"}))


BASIS_GROUPS = [
    ("Google Willow", ["real_willow"]), ("Quantinuum Apollo", ["real_apollo"]),
    ("Harvard/MIT atoms", ["real_harvard"]), ("All-to-all, SD6", ["custom_full"]),
    ("IBM Heron", ["real_torino"]), ("IBM Flamingo / Loon", ["real_flamingo", "real_flamingo_1_qpu", "real_loon",
                                                             "real_loon_1_qpu"]),
    ("IBM bicycle module", ["real_bicycle", "real_bicycle_lr"]), ("Native layouts", ["native", "native_lr"]),
    ("Infleqtion", ["real_infleqtion"]),
]


def x_vs_z(t, reg):
    """Per-round logical error rate in the X basis over the Z basis, same configuration and decoder"""
    t = t[(t["operation"] == "memory") & (t["status"] != "upper bound")].dropna(subset=["ler_per_round"])
    key = [c for c in dataset.CONFIG if c not in ("basis", "slice")]
    pairs = t[t["basis"] == "Z"].merge(t[t["basis"] == "X"], on=key + ["decoder"], suffixes=("_z", "_x"))
    best = pairs.groupby(key, dropna=False).agg(z=("ler_per_round_z", "min"), x=("ler_per_round_x", "min")).reset_index()
    best["ratio"] = best["x"] / best["z"]
    group_of = {b: name for name, backends in BASIS_GROUPS for b in backends}
    best["group"] = best["backend"].map(group_of)
    best = best.dropna(subset=["group"])
    order = best.groupby("group")["ratio"].median().sort_values().index.tolist()
    rng = np.random.default_rng(0)
    fig, ax = S.figure(7.5, 4.4)
    for i, grp in enumerate(order):
        r = best[best["group"] == grp]["ratio"]
        ax.scatter(r, i + rng.uniform(-0.28, 0.28, len(r)), s=9, color=S.SERIES[0], alpha=0.35, lw=0)
        med = r.median()
        ax.plot([med, med], [i - 0.38, i + 0.38], color=S.INK, lw=2.5, solid_capstyle="round")
        ax.text(260, i, f"{med:.1f}×  (n = {len(r)})", va="center", fontsize=9, color=S.INK_2)
    ax.axvline(1, color=S.INK_2, lw=1)
    ax.set_yticks(range(len(order)), order)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 250)
    ax.set_xticks([0.1, 1, 10, 100], ["0.1×", "1×", "10×", "100×"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("X-basis / Z-basis logical error rate per round (bar: median)")
    ax.text(0.9, len(order) - 0.4, "X better ←", ha="right", fontsize=8, color=S.INK_2)
    ax.text(1.1, len(order) - 0.4, "→ X worse", ha="left", fontsize=8, color=S.INK_2)
    ax.set_title("Z-basis memory experiments flatter the code")
    worse = (best["ratio"] > 1).mean() * 100
    reg.save(fig, "x-vs-z", group=GROUP, title="X-basis against Z-basis memory",
             finding=f"Measured in the X basis, {worse:.0f}% of configurations do worse than in the usual Z basis; "
                     f"on Infleqtion by a median {best[best['group'] == 'Infleqtion']['ratio'].median():.1f}×.",
             caption=f"{len(best):,} configurations run in both bases with the same compiled circuit; per basis, the "
                     "lowest per-round logical error rate over the decoders run in both. Upper bounds are left out.",
             data=best[["group", "code", "backend", "error_type", "distance", "z", "x", "ratio"]])


# Willow surface-code memory, Google's experiment with its ensembled matching decoder (logical error per cycle):
# Google Quantum AI, "Quantum error correction below the surface code threshold", Nature 638, 920 (2025), Table S1
GOOGLE_MATCHING = {3: 0.0071, 5: 0.0035, 7: 0.00171}


def willow(t, reg):
    t = t[(t["slice"] == "willow_surface") & (t["backend"] == "real_willow") & (t["error_type"] == "real_willow")
          & (t["cycles"] == 10)]
    ours = t.groupby(["decoder", "distance"])["ler_per_round"].min().unstack("decoder")
    df = pd.DataFrame({"distance": [3, 5, 7], "google_matching": [GOOGLE_MATCHING[d] for d in (3, 5, 7)],
                       "eccentric_mwpm": ours.loc[[3, 5, 7], "mwpm"].values,
                       "eccentric_tesseract": ours.loc[[3, 5, 7], "tesseract"].values})
    fig, ax = S.figure(6.5, 4.2)
    series = [("google_matching", "Google's experiment (matching)", S.SERIES[0]),
              ("eccentric_mwpm", "ECCentric Willow model (MWPM)", S.SERIES[1]),
              ("eccentric_tesseract", "ECCentric Willow model (Tesseract)", S.SERIES[2])]
    for col, label, color in series:
        ax.plot(df["distance"], df[col] * 100, marker="o", ms=7, color=color, label=label)
        ax.annotate(label, (7, df[col].iloc[-1] * 100), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=S.INK)
    ax.set_yscale("log")
    ax.set_xticks([3, 5, 7])
    ax.set_xlim(2.6, 10.8)
    ax.minorticks_off()
    ax.set_yticks([0.03, 0.1, 0.3, 1], ["0.03%", "0.1%", "0.3%", "1%"])
    ax.set_xlabel("Code distance")
    ax.set_ylabel("Logical error per cycle")
    ax.set_title("The Willow model against Google's experiment")
    ratio = df["google_matching"] / df["eccentric_mwpm"]
    lam_g = df["google_matching"].iloc[1] / df["google_matching"].iloc[2]
    lam_m = df["eccentric_mwpm"].iloc[1] / df["eccentric_mwpm"].iloc[2]
    reg.save(fig, "willow", group=GROUP, title="The Willow model against Google's experiment",
             finding=f"With matching decoders on both sides, the model is {ratio.min():.1f}–{ratio.max():.1f}× optimistic "
                     f"and reproduces the error suppression (Λ = {lam_m:.2f} against Google's {lam_g:.2f}).",
             caption="Rotated surface-code memory over 10 cycles on the Willow device model (real_willow noise). "
                     "Google: ensembled matching decoder, Nature 638, 920 (2025), Table S1. The gap to Tesseract is "
                     "the decoder: a stronger decoder lowers the same circuit's error by another 2×.",
             data=df)


HEATMAP_DEVICES = ["real_apollo", "real_infleqtion", "real_willow", "real_harvard", "real_bicycle", "real_loon",
                   "real_loon_1_qpu", "real_flamingo", "real_flamingo_1_qpu", "real_torino"]


def where_codes_work(t, reg):
    """Best per-round logical error rate of each code family on each device, at today's noise"""
    t = _memory_z(t)
    t = t[t["backend"].isin(HEATMAP_DEVICES)]
    t = t[t["error_type"] == t["backend"].map(S.TODAY)]
    t = t.assign(family=t["code"].map(lambda c: dataset.family(c)[0]))
    t = t[~t["family"].str.startswith("Lattice surgery")]
    rows = []
    for (fam, dev), g in t.groupby(["family", "backend"]):
        rate = g["ler_per_round"].min()
        bound = False
        if pd.isna(rate):
            rate = g["ler_per_round_high"].min()
            bound = not pd.isna(rate)
        rows.append({"family": fam, "device": dev, "ler_per_round": rate, "is_bound": bound,
                     "above_threshold": pd.isna(rate)})
    df = pd.DataFrame(rows)
    grid = df.pivot(index="family", columns="device", values="ler_per_round").reindex(columns=HEATMAP_DEVICES)
    grid = grid.loc[grid.min(axis=1).sort_values(na_position="last").index]
    tested = df.pivot(index="family", columns="device", values="above_threshold").reindex(
        index=grid.index, columns=HEATMAP_DEVICES)
    bounds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8][::-1]
    cmap = ListedColormap(S.BLUES[::-1][:6])
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = S.figure(9, 8.5)
    ax.set_facecolor(S.SURFACE)
    for i, fam in enumerate(grid.index):
        for j, dev in enumerate(HEATMAP_DEVICES):
            v, above = grid.loc[fam, dev], tested.loc[fam, dev]
            if pd.isna(above):
                continue  # not run
            color = S.NEUTRAL if above or v >= 0.1 else cmap(norm(max(v, 1.1e-8)))
            ax.add_patch(S.plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor=S.SURFACE, lw=2))
    ax.set_xlim(0, len(HEATMAP_DEVICES))
    ax.set_ylim(len(grid), 0)
    ax.set_xticks(np.arange(len(HEATMAP_DEVICES)) + 0.5, [S.DEVICE_LABELS[d] for d in HEATMAP_DEVICES],
                  rotation=40, ha="right")
    ax.set_yticks(np.arange(len(grid)) + 0.5, grid.index)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    labels = ["< 10⁻⁶", "10⁻⁶–10⁻⁵", "10⁻⁵–10⁻⁴", "10⁻⁴–10⁻³", "10⁻³–10⁻²", "10⁻²–10⁻¹"]
    handles = [Patch(facecolor=cmap(k), label=lab) for k, lab in enumerate(labels)]
    handles.append(Patch(facecolor=S.NEUTRAL, label="≥ 10⁻¹ or above threshold"))
    handles.append(Patch(facecolor=S.SURFACE, edgecolor=S.AXIS, label="not run"))
    ax.legend(handles=handles, title="Logical error per round", loc="upper left", bbox_to_anchor=(1.01, 1),
              title_fontsize=9, alignment="left")
    ax.set_title("Where each code family works at today's noise")
    good = df[(df["ler_per_round"] < 1e-3)].groupby("device")["family"].nunique()
    reg.save(fig, "where-codes-work", group=GROUP, title="Code families × devices at today's noise",
             finding=f"Trapped ions protect the most code families: below 10⁻³ per round, {good.get('real_apollo', 0)} "
                     f"families on Quantinuum Apollo against {good.get('real_torino', 0)} on IBM Heron.",
             caption="Lowest per-round logical error rate of any code of the family, any distance and decoder, Z-basis "
                     "memory, under the device's own noise model. Where no logical error was seen, the 95% upper bound "
                     "is used. IBM devices use the Flamingo noise model, Heron its own.",
             data=df)


ALL = [where_codes_work, routing_vs_native, below_threshold, decoders, x_vs_z, willow]
