"""
make_readout_tradeoff.py
========================
For the Group-1 six-code memory experiments, compute the LER % difference between
readout lengths, per (code, distance, noise model). Baseline = the slowest readout
(1000 ns); each shorter readout is "X% faster" with a "Y% LER difference".

  e.g. 400 ns readout = 60% faster than 1000 ns -> look up its ler_pct_diff_vs_1000ns.

Reads every six-code-memory CSV in experiment_results/noise_models/ and writes:
  - readout_length_tradeoff.csv  (flat lookup table)
  - readout_length_tradeoff.md   (per-model tables for eyeballing)
"""
import os, csv

ND = "experiment_results/noise_models"
RO = [200, 400, 600, 800, 1000]
BASE = 1000  # slowest readout = baseline for "% faster" and "% LER diff"

# csv file -> (noise_model key in that file, friendly label, readout note)
SOURCES = [
    ("ler_noise_models.csv", "heron_herqules",
     "Model 1 (ibm_boston gates + HERQULES readout + decoherence)", "HERQULES readout"),
    ("ler_noise_models.csv", "readout_decoherence",
     "Model 2 (HERQULES readout + decoherence, no gate errors)", "HERQULES readout"),
    ("ler_noise_models.csv", "decoherence_only",
     "Model 3 (decoherence only, no readout flip)", "no readout flip (length acts via decoherence only)"),
    ("ler_noise_models_futuristic.csv", "heron_herqules_futuristic",
     "Futuristic (Model 1, all errors /10, T1/T2 x3)", "HERQULES readout /10"),
    ("ler_noise_models_highfid.csv", "heron_highfid",
     "High-fidelity readout (Model 1, readout error /10 only)", "HERQULES readout /10"),
    ("ler_noise_models_boston_lowt_backlog.csv", "boston_lowt",
     "boston_lowt_backlog (ibm_boston gates, T1/T2=190/130us, fixed readout, backlog ON)",
     "FIXED readout 3.54e-3 (length acts via decoherence/backlog only)"),
    ("ler_noise_models_futuristic_fixedreadout.csv", "futuristic_fixedreadout",
     "Futuristic gates (/10) + T1/T2 x3 + FIXED readout 3.54e-4 (boston/10, no HERQULES)",
     "FIXED readout 3.54e-4 (length acts via decoherence only)"),
]


def load(path, model):
    """Return {(code, d): {ro: ler}} for one noise model in one CSV."""
    data = {}
    if not os.path.exists(path):
        return data
    for r in csv.DictReader(open(path)):
        if r["noise_model"] != model:
            continue
        v = r["logical_error_rate"]
        if v in ("", "None"):
            continue
        key = (r["code"], int(r["distance"]))
        data.setdefault(key, {})[int(r["readout_ns"])] = float(v)
    return data


def pct_faster(ro):
    return round((BASE - ro) / BASE * 100)


def pct_diff(ler, ler_base):
    if ler_base is None or ler_base == 0.0:
        return None
    return (ler - ler_base) / ler_base * 100.0


CODE_ORDER = {c: i for i, c in enumerate(["surface", "bacon", "color", "hh", "steane", "gross"])}

rows_csv = []
md = ["# Readout-length vs LER tradeoff (Group 1, six-code memories)\n",
      "Baseline = **1000 ns** (slowest readout). For each shorter readout, "
      "`% faster` = (1000 − ns)/1000, and `Δ%` = (LER − LER@1000ns)/LER@1000ns "
      "(positive = worse). Cells show `LER (Δ%)`.\n",
      "Quick read: e.g. *distance 7, 400 ns (60% faster)* -> find the code's row, "
      "the 400 ns column.\n"]

for path, model, label, note in SOURCES:
    data = load(os.path.join(ND, path), model)
    if not data:
        continue
    md.append(f"\n## {label}")
    md.append(f"_Readout: {note}. Columns: readout ns (% faster vs 1000 ns)._\n")
    md.append("| code | d | 200 (80% faster) | 400 (60%) | 600 (40%) | 800 (20%) | 1000 (base) |")
    md.append("|---|---|---|---|---|---|---|")
    for (code, d) in sorted(data, key=lambda k: (CODE_ORDER.get(k[0], 9), k[1])):
        lers = data[(code, d)]
        base = lers.get(BASE)
        cells = []
        for ro in RO:
            ler = lers.get(ro)
            if ler is None:
                cells.append("—"); continue
            if ro == BASE:
                cells.append(f"{ler:.5f}")
            else:
                dp = pct_diff(ler, base)
                dp_s = "n/a" if dp is None else f"{dp:+.1f}%"
                cells.append(f"{ler:.5f} ({dp_s})")
            # flat CSV row
            dp = pct_diff(ler, base)
            rows_csv.append([code, d, model, ro, pct_faster(ro), f"{ler:.6f}",
                             ("" if base is None else f"{base:.6f}"),
                             ("" if dp is None else f"{dp:.1f}")])
        md.append(f"| {code} | {d} | " + " | ".join(cells) + " |")

# write CSV
with open(os.path.join(ND, "readout_length_tradeoff.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["code", "distance", "noise_model", "readout_ns", "pct_faster_vs_1000ns",
                "ler", "ler_1000ns", "ler_pct_diff_vs_1000ns"])
    w.writerows(rows_csv)

# write MD
with open(os.path.join(ND, "readout_length_tradeoff.md"), "w") as f:
    f.write("\n".join(md) + "\n")

print(f"wrote readout_length_tradeoff.csv ({len(rows_csv)} rows) and .md")
