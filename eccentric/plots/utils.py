import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mtick
from matplotlib.ticker import LogLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Rectangle

marker_styles = {
    'gross': 'o',
    'bacon': 's',
    'hh': '^',
    'surface': 'D',
    'steane': 'X',
    'color': '*',
    'other': ''
}

code_rename_map = {
    'bacon': 'Bacon-Shor',
    'hh': 'Heavy-hex',
    'gross': 'Gross'
}

error_type_map = {
    'Constant': 'constant',
    'SI1000': 'modsi1000'
}

decoder_map = {
    'mwpm': 'MWPM',
    'bposd_batch': 'BP-OSD Batch',
    'bposd_chk': 'BP-OSD PCM',
    'bposd': 'BP-OSD',
    'chromobius': 'Chromobius'
}

backend_rename_map = {
    "real_willow": "Willow",
    "real_willow_ns": "Willow",
    "real_infleqtion": "Infleqtion\n(w/ s.)",
    "real_nsinfleqtion": "Infleqtion",
    "real_harvard": "Harvard/MIT",
    "real_nsapollo": "Apollo\n(w/o s.)",
    "real_apollo": "Apollo",
    "real_flamingo": "DQC",
    "real_flamingo_1_qpu": "1 QPU",
    "real_nighthawk": "DQC",
    "real_nighthawk_1_qpu": "1 QPU"
}

code_palette = sns.color_palette("pastel", n_colors=8)
code_hatches = ["/", "\\", "//", "++", "xx", "**"]

WIDE_FIGSIZE = 6
HEIGHT_FIGSIZE = 2.2
FONTSIZE = 12
BAR_WIDTH = 0.2
group_spacing = 0.4

HIGHLIGHT = {
    ('surface', 'Constant'): [400, 500],
    ('hh', 'Constant'): [450],
    ('color', 'Constant'): [400, 500],
    ('bacon', 'Constant'): [400, 450],
    ('surface', 'SI1000'): [400, 500],
    ('hh', 'SI1000'): [450],
    ('color', 'SI1000'): [400, 500],
    ('bacon', 'SI1000'): [400, 450],
}


tex_fonts = {
    "font.family": "serif",
    "axes.labelsize": FONTSIZE,
    "font.size": FONTSIZE,
    "legend.fontsize": FONTSIZE - 2,
    "xtick.labelsize": FONTSIZE - 2,
    "ytick.labelsize": FONTSIZE - 2,
    "axes.titlesize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.5,
    "lines.markeredgecolor": "black",
    "errorbar.capsize": 3,
}

plt.rcParams.update(tex_fonts)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
