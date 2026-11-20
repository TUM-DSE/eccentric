"""Shared look of the figures and the helper that saves each one with its data.

Colors follow the entity, never its rank: every code, device and series keeps its slot in every figure.
Palette: the reference categorical palette of the data-viz guidelines (adjacent pairs pass the CVD and
normal-vision checks in this order; dot and scatter plots use at most the first three slots).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b9b8b1"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Fixed identity colors
CODE_COLORS = {"surface": SERIES[0], "color": SERIES[1], "hh": SERIES[2], "bacon": SERIES[3],
               "gross": SERIES[4], "steane": SERIES[5]}
CODE_LABELS = {"surface": "Surface", "color": "Color", "hh": "Heavy-hex", "bacon": "Bacon-Shor",
               "gross": "Gross", "steane": "Concat. Steane"}
# In running text
CODE_NAMES = {"surface": "surface", "color": "color", "hh": "heavy-hex", "bacon": "Bacon-Shor", "gross": "gross",
              "steane": "concatenated Steane"}
PAPER_CODES = list(CODE_COLORS)

DEVICE_LABELS = {
    "real_torino": "IBM Heron", "real_flamingo": "IBM Flamingo (3 QPUs)", "real_flamingo_1_qpu": "IBM Flamingo (1 QPU)",
    "real_loon": "IBM Loon (2 QPUs)", "real_loon_1_qpu": "IBM Loon (1 QPU)", "real_bicycle": "IBM bicycle module",
    "real_bicycle_lr": "IBM bicycle module (noisy LR)", "real_willow": "Google Willow", "real_apollo": "Quantinuum Apollo",
    "real_nsapollo": "Apollo, no shuttling", "real_infleqtion": "Infleqtion", "real_nsinfleqtion": "Infleqtion, no shuttling",
    "real_harvard": "Harvard/MIT atoms", "native": "Native layout", "native_lr": "Native layout (noisy LR)",
    "custom_full": "All-to-all", "custom_cube": "3D cube", "custom_grid": "2D grid",
}
# Today's noise model of each device
TODAY = {"real_torino": "real_heron", "real_flamingo": "real_flamingo", "real_flamingo_1_qpu": "real_flamingo",
         "real_loon": "real_flamingo", "real_loon_1_qpu": "real_flamingo", "real_bicycle": "real_flamingo",
         "real_willow": "real_willow", "real_apollo": "real_apollo", "real_infleqtion": "real_infleqtion",
         "real_harvard": "real_harvard"}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 6,
        "svg.fonttype": "none", "svg.hashsalt": "eccentric",  # stable ids: a rebuild changes only what changed
    })


def figure(width=7.5, height=4.2, **kw):
    fig, ax = plt.subplots(figsize=(width, height), **kw)
    return fig, ax


class Registry:
    """Collects the figures of a build: their files, data and descriptions"""

    def __init__(self, out: Path, source: dict):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.entries = []

    def save(self, fig, fid, *, group, title, finding, caption, data, columns=None):
        """Write <fid>.svg, <fid>.png and <fid>.csv (the plotted values) and record the figure"""
        fig.savefig(self.out / f"{fid}.svg", bbox_inches="tight", metadata={"Date": None})
        fig.savefig(self.out / f"{fid}.png", dpi=200, bbox_inches="tight", metadata={"Software": None})
        plt.close(fig)
        data = data[columns] if columns else data
        data.to_csv(self.out / f"{fid}.csv", index=False, float_format="%.6g")
        self.entries.append({"id": fid, "group": group, "title": title, "finding": finding, "caption": caption,
                             "files": {"svg": f"{fid}.svg", "png": f"{fid}.png", "csv": f"{fid}.csv"}})

    def write_manifest(self):
        manifest = {"source": self.source, "figures": self.entries}
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
