"""The system diagrams, in a light and a dark version each (docs/figures/<name>-{light,dark}.png):

  overview   what `advise` does to a circuit: input, features, models, recommendation, output
  runtime    predict, pilot, commit (qec_matchmaker.runtime.execute)
  training   from ECCentric's sweep to the packaged model and the report

python scripts/make_diagrams.py [--data data] [--out docs/figures]

Drawn by hand in inches on a blank canvas; the counts come from the training table.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from make_figures import HERE, THEMES, style

FILLS = {"light": dict(box="#f6f5f1", tint="#e6f0fd"), "dark": dict(box="#161b22", tint="#10294d")}
PAD, LINE = 0.11, 0.185  # a box's inner margin and line pitch, inches
MONO = ["DejaVu Sans Mono", "monospace"]


class Box:
    def __init__(self, x, top, w, h):
        self.x, self.w, self.h, self.top_y, self.bottom_y = x, w, h, top, top - h
        self.cx, self.cy = x + w / 2, top - h / 2
        self.left, self.right = (x, self.cy), (x + w, self.cy)
        self.top, self.bottom = (self.cx, top), (self.cx, top - h)


def canvas(theme, w, h, title):
    t = dict(style(theme), **FILLS[theme])
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(0.15, h - 0.3, title, fontsize=12.5, fontweight="semibold", color=t["ink"], va="center")
    return fig, ax, t


def height(title, body):
    return 2 * PAD + (bool(title) + len(body)) * LINE + (0.03 if title and body else 0)


def box(ax, t, x, top, w, title=None, body=(), kind="step", h=None, center=False):
    """A box with its top-left corner at (x, top). Kinds: "step" (something happens), "question" (a decision),
    "result" (what comes out), "option" (an optional input), "panel" (a frame around other boxes). A body line starting with ` is a file name"""
    content = height(title, body)
    h = h or content
    face, edge, lw, radius = {"step": (t["box"], t["axis"], 0.8, 0.07), "question": (t["surface"], t["ink2"], 1.0, 0.2),
                              "result": (t["tint"], t["accent"], 1.2, 0.07),
                              "option": (t["surface"], t["muted"], 0.8, 0.2),
                              "panel": (t["surface"], t["axis"], 0.8, 0.09)}[kind]
    ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=face,
                                edgecolor=edge, linewidth=lw, zorder=3))
    y = top - PAD - LINE / 2 - (0 if kind == "panel" else (h - content) / 2)
    tx, ha = (x + w / 2, "center") if center else (x + PAD, "left")
    if title:
        ax.text(tx, y, title, fontsize=9.5, fontweight="semibold", color=t["ink"], va="center", ha=ha, zorder=4)
        y -= LINE + (0.03 if body else 0)
    for line in body:
        if line.startswith("`"):
            ax.text(tx, y, line.strip("`"), fontsize=7.8, family=MONO, color=t["ink2"], va="center", ha=ha, zorder=4)
        else:
            ax.text(tx, y, line, fontsize=9, color=t["ink2"], va="center", ha=ha, zorder=4)
        y -= LINE
    return Box(x, top, w, h)


def arrow(ax, t, *points, head=True):
    xs, ys = zip(*points)
    if len(points) > 2 or not head:
        end = len(points) if not head else len(points) - 1
        ax.plot(xs[:end], ys[:end], color=t["ink2"], lw=1.1, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    if head:
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=9, lw=1.1,
                                     color=t["ink2"], shrinkA=0, shrinkB=0, zorder=2))


def label(ax, t, x, y, text, mono=False, **kw):
    ax.text(x, y, text, fontsize=8.3 if not mono else 7.8, family=MONO if mono else None, color=t["ink2"],
            va=kw.pop("va", "center"), ha=kw.pop("ha", "center"), zorder=5,
            bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1.2), **kw)


def stage(ax, t, x, y, text):
    ax.text(x, y, text.upper(), fontsize=7.8, color=t["muted"], va="center", ha="left")


def save(fig, out, name, theme):
    fig.savefig(os.path.join(out, f"{name}-{theme}.png"), dpi=200)
    plt.close(fig)


def fig_overview(counts, theme, out):
    w, h = 10, 5.4
    fig, ax, t = canvas(theme, w, h, "From a circuit to a decoder, without decoding a shot")
    widths, gap, top = [1.6, 1.75, 2.0, 1.6, 1.75], 0.25, 4.45
    xs = [0.15 + sum(widths[:i]) + i * gap for i in range(5)]
    for x, name in zip(xs, ["Input", "Features", "Models", "Choice", "Output"]):
        stage(ax, t, x, top + 0.17, name)

    minis = [("Regime", "P(above threshold)"), ("Best error rate", "quantiles 10 / 50 / 90%"),
             ("Regret, per decoder", "error rate over the best"), ("Time, per decoder", "ms per shot, quantiles")]
    mini_h, head = height("a", ["b"]), height("a", ["b"]) - PAD + 0.05
    panel_h = head + len(minis) * (mini_h + 0.07) + 0.04
    models = box(ax, t, xs[2], top, widths[2], "Trees on scaling laws", [f"trained on {counts['circuits']:,} circuits"],
                 kind="panel", h=panel_h)
    for i, (title, line) in enumerate(minis):
        box(ax, t, xs[2] + 0.1, top - head - i * (mini_h + 0.07), widths[2] - 0.2, title, [line])

    def centered(i, title, body, kind="step"):
        return box(ax, t, xs[i], models.cy + height(title, body) / 2, widths[i], title, body, kind=kind)

    source = centered(0, "Noisy circuit", ["Stim circuit (.stim)", "error model (.dem)", "decoder-bench (.h5)"])
    features = centered(1, "Detector error model", [f"reduced to {counts['features']} features:", "detector firing rates",
                                                    "error and logical mass", "mechanisms, their weight",
                                                    "matchable? colored?", "circuit size, rounds"])
    choice = centered(3, "Recommendation", ["of the decoders that", "apply: the cheapest", "good enough, the most",
                                            "accurate, or the best", "within a time budget"])
    advice = centered(4, "Advice", ["P(above threshold)", "error rate, 80% interval", "ms per shot, per decoder",
                                    "the decoder to use", "and why; warnings"], kind="result")
    for a, b in [(source, features), (features, models), (models, choice), (choice, advice)]:
        arrow(ax, t, a.right, b.left)

    stage(ax, t, 0.15, 0.49, "Options")
    for x, pw, text, ax_x, target in [(3.3, 1.75, "code distance, if known", 4.55, models.bottom_y),
                                      (5.2, 1.8, "your machine's speed", 5.6, models.bottom_y),
                                      (7.15, 2.1, "objective, tolerance, budget", 7.5, choice.bottom_y)]:
        option = box(ax, t, x, 0.7, pw, body=[text], kind="option", center=True)
        arrow(ax, t, (ax_x, option.top_y), (ax_x, target))
    save(fig, out, "overview", theme)


def fig_runtime(theme, out):
    w, h = 10, 4.55
    fig, ax, t = canvas(theme, w, h, "At run time: predict, pilot, commit")
    cw, gap = 1.36, 0.308
    xs = [0.15 + i * (cw + gap) for i in range(6)]
    r1, h1, r2, h2, r3 = 3.8, 1.0, 2.3, 0.8, 0.95

    predict = box(ax, t, xs[0], r1, cw, "Predict", ["rank the candidate", "decoders for the", "objective"], h=h1)
    skip = box(ax, t, xs[1], r1, cw, body=["Above threshold", "with P > 0.9?"], kind="question", h=h1, center=True)
    pilot = box(ax, t, xs[2], r1, cw, "Pilot", ["next candidate: its", "first 100 shots, cap", "10× predicted time"], h=h1)
    done = box(ax, t, xs[3], r1, cw, body=["Finished within", "the cap?"], kind="question", h=h1, center=True)
    errors = box(ax, t, xs[4], r1, cw, body=["Over half of the", "shots logical", "errors?"], kind="question", h=h1,
                 center=True)
    commit = box(ax, t, xs[5], r1, cw, "Commit", ["full decode with", "the piloted decoder"], h=h1)

    skipped = box(ax, t, xs[1], r2, cw, "Skipped", ["nothing decoded"], kind="result", h=h2)
    more = box(ax, t, xs[2], r2, cw, body=["Candidates left,", "under 4 piloted?"], kind="question", h=h2, center=True)
    weak = box(ax, t, xs[4], r2, cw, body=["Most accurate", "decoder failed,", "or two did?"], kind="question", h=h2,
               center=True)
    decoded = box(ax, t, xs[5], r2, cw, "Decoded", ["to the error target"], kind="result", h=h2)
    gave_up = box(ax, t, xs[2], r3, cw, "Gave up", ["only pilots paid"], kind="result")
    above = box(ax, t, xs[4], r3, cw, "Above threshold", ["only pilots paid"], kind="result")

    for a, b, text in [(predict, skip, None), (skip, pilot, "no"), (pilot, done, None), (done, errors, "yes"),
                       (errors, commit, "no")]:
        arrow(ax, t, a.right, b.left)
        if text:
            label(ax, t, (a.right[0] + b.left[0]) / 2 - 0.02, a.cy + 0.13, text)
    for a, b, text in [(skip, skipped, "yes"), (errors, weak, "yes"), (commit, decoded, None), (weak, above, "yes"),
                       (more, gave_up, "no")]:
        arrow(ax, t, a.bottom, b.top)
        if text:
            label(ax, t, a.cx + 0.2, (a.bottom_y + b.top_y) / 2, text)
    arrow(ax, t, more.top, pilot.bottom)
    label(ax, t, more.cx + 0.2, (more.top_y + pilot.bottom_y) / 2, "yes")
    # Both ways back to the next candidate: too slow for its cap, or failed where a better decoder may not
    arrow(ax, t, weak.left, more.right)
    arrow(ax, t, done.bottom, (done.cx, weak.cy), head=False)
    ax.plot([done.cx], [weak.cy], marker="o", ms=3.5, color=t["ink2"], zorder=3)
    label(ax, t, done.cx + 0.17, (done.bottom_y + weak.cy) / 2 + 0.1, "no")
    label(ax, t, (done.cx + weak.x) / 2, weak.cy + 0.13, "no")
    ax.text(done.cx, weak.cy - 0.2, "slower than predicted, or a weak\ndecoder: try the next one, the\nmost accurate first after a failure",
            fontsize=8.3, color=t["muted"], va="top", ha="center", linespacing=1.45)
    save(fig, out, "runtime", theme)


def fig_training(counts, theme, out):
    w, h = 10, 4.25
    fig, ax, t = canvas(theme, w, h, "Learning from ECCentric, and keeping up with it")
    widths, gap = [2.0, 1.75, 1.75, 1.85], 0.78
    xs = [0.15 + sum(widths[:i]) + i * gap for i in range(4)]
    up, down = 2.85, 1.45
    mid = (up + down) / 2
    stage(ax, t, xs[0], 3.6, "ECCentric benchmark")
    stage(ax, t, xs[1], 3.6, "This repository")

    def at(i, cy, title, body, kind="step"):
        return box(ax, t, xs[i], cy + height(title, body) / 2, widths[i], title, body, kind=kind)

    circuits = at(0, up, f"{counts['circuits']:,} circuits", [f"{counts['codes']} codes × {counts['devices']} devices",
                                                             f"× {counts['noise']} noise models,", "compiled and simulated"])
    decodes = at(0, down, f"{counts['rows']:,} decodes", [f"{counts['decoders']} decoders on the same", "shots: error rate, time",
                                                         "`results.csv`"])
    store = at(1, up, "Feature store", [f"{counts['features']} features per circuit", "`data/features.jsonl`"])
    table = at(2, mid, "Training table", ["one row per circuit", "and decoder", "`data/dataset.csv`", "`data/MANIFEST.json`"])
    model = at(3, up, "Packaged model", ["trees and laws, shipped", "with the package", "`qec_matchmaker/model/`"],
               kind="result")
    report = at(3, down, "Report", ["held-out codes, 10 folds", "`reports/REPORT.md`"], kind="result")

    arrow(ax, t, circuits.right, store.left)
    label(ax, t, (circuits.right[0] + store.left[0]) / 2 - 0.03, up + 0.14, "extract", mono=True)
    join = table.x - gap + 0.2
    arrow(ax, t, store.right, (join, up), (join, mid), head=False)
    arrow(ax, t, decodes.right, (join, down), (join, mid), table.left)
    label(ax, t, (join + table.x) / 2 - 0.03, mid + 0.14, "ingest", mono=True)
    fork = table.x + table.w + 0.14
    arrow(ax, t, table.right, (fork, mid), (fork, up), model.left)
    arrow(ax, t, (fork, mid), (fork, down), report.left)
    centre = (fork + model.x) / 2 - 0.02
    label(ax, t, centre, up + 0.14, "train", mono=True)
    label(ax, t, centre, down - 0.14, "evaluate", mono=True)

    y = 0.62
    ax.plot([circuits.right[0] + 0.1, circuits.right[0] + 0.1, w - 0.15, w - 0.15], [y + 0.08, y, y, y + 0.08],
            color=t["axis"], lw=0.9)
    ax.text((circuits.right[0] + 0.1 + w - 0.15) / 2, y - 0.2,
            "scripts/update_from_eccentric.sh runs the four steps and commits what changed", fontsize=8.6,
            color=t["ink2"], ha="center", va="center")
    save(fig, out, "training", theme)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "data"))
    p.add_argument("--out", default=os.path.join(HERE, "docs", "figures"))
    a = p.parse_args()
    from qec_matchmaker.dataset import load_dataset
    from qec_matchmaker.features import FEATURE_NAMES

    df, _ = load_dataset(a.data)
    counts = dict(rows=len(df), circuits=df.circuit_key.nunique(), codes=df.code.nunique(), devices=df.backend.nunique(),
                  noise=df.error_type.nunique(), decoders=df.decoder.nunique(), features=len(FEATURE_NAMES))
    os.makedirs(a.out, exist_ok=True)
    for theme in THEMES:
        fig_overview(counts, theme, a.out)
        fig_runtime(theme, a.out)
        fig_training(counts, theme, a.out)
    print("diagrams in", a.out, counts)


if __name__ == "__main__":
    main()
