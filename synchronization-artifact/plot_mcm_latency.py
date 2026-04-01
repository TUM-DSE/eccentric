"""
plot_mcm_latency.py
===================
Plots results from evaluate_mcm_latency.py and/or evaluate_mcm_error.py.

Usage
-----
  python3 plot_mcm_latency.py                        # latency sweep only
  python3 plot_mcm_latency.py --error                # error-rate sweep only
  python3 plot_mcm_latency.py --both                 # side-by-side
  python3 plot_mcm_latency.py <latency.csv>          # custom latency CSV
  python3 plot_mcm_latency.py --error <error.csv>    # custom error CSV
"""

import sys
import os
import csv
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LATENCY_CSV = 'mcm_latency_ler.csv'
DEFAULT_ERROR_CSV   = 'mcm_error_ler.csv'
OUTPUT_LATENCY_PNG  = 'mcm_latency_ler.png'
OUTPUT_ERROR_PNG    = 'mcm_error_ler.png'
OUTPUT_BOTH_PNG     = 'mcm_sweep_both.png'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> dict:
    """Return {group_key: {x_val: ler}} — group key is the first column value."""
    data = defaultdict(dict)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        # first col = grouping (distance), second = x-axis, third = LER
        grp_col, x_col, y_col = cols[0], cols[1], cols[2]
        for row in reader:
            grp = int(row[grp_col])
            x   = float(row[x_col])
            y   = float(row[y_col])
            data[grp][x] = y
    return data, x_col


def _plot_axis(ax, data: dict, x_label: str, title: str):
    distances = sorted(data.keys())
    colors = plt.cm.viridis_r(
        [i / max(len(distances) - 1, 1) for i in range(len(distances))]
    )
    for d, color in zip(distances, colors):
        pts = sorted(data[d].items())
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        ax.plot(xs, ys, marker='o', linewidth=2, markersize=6,
                color=color, label=f'd = {d}')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax.legend(title='Code distance', fontsize=10, title_fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_latency(csv_path=DEFAULT_LATENCY_CSV, out=OUTPUT_LATENCY_PNG):
    data, x_col = load_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_axis(ax, data,
               x_label='MCM Latency (ns)',
               title='LER vs. MCM Latency\n(merge=True, fixed gate error & CNOT latency)')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Plot saved to {out}')
    plt.show()


def plot_error(csv_path=DEFAULT_ERROR_CSV, out=OUTPUT_ERROR_PNG):
    data, x_col = load_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_axis(ax, data,
               x_label='Measurement Error Rate',
               title='LER vs. Measurement Error Rate\n(merge=True, MCM latency = 2 µs)')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Plot saved to {out}')
    plt.show()


def plot_both(lat_csv=DEFAULT_LATENCY_CSV, err_csv=DEFAULT_ERROR_CSV,
              out=OUTPUT_BOTH_PNG):
    lat_data, _ = load_csv(lat_csv)
    err_data, _ = load_csv(err_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _plot_axis(ax1, lat_data,
               x_label='MCM Latency (ns)',
               title='LER vs. MCM Latency\n(fixed gate error, latency sweep)')
    _plot_axis(ax2, err_data,
               x_label='Measurement Error Rate',
               title='LER vs. Measurement Error\n(fixed MCM latency = 2 µs)')
    fig.suptitle('Effect of MCM Parameters on Logical Error Rate (merge=True)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Plot saved to {out}')
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot MCM sweep results.')
    parser.add_argument('csv', nargs='?', default=None,
                        help='Path to CSV (latency or error sweep)')
    parser.add_argument('--error', nargs='?', const=DEFAULT_ERROR_CSV, default=None,
                        metavar='CSV',
                        help='Plot error-rate sweep (optional custom CSV path)')
    parser.add_argument('--both', action='store_true',
                        help='Side-by-side plot of both sweeps')
    args = parser.parse_args()

    if args.both:
        lat_csv = args.csv or DEFAULT_LATENCY_CSV
        err_csv = args.error or DEFAULT_ERROR_CSV
        for p in [lat_csv, err_csv]:
            if not os.path.exists(p):
                print(f'Error: {p!r} not found.')
                sys.exit(1)
        plot_both(lat_csv, err_csv)
    elif args.error is not None:
        csv_path = args.error
        if not os.path.exists(csv_path):
            print(f'Error: {csv_path!r} not found.')
            sys.exit(1)
        plot_error(csv_path)
    else:
        csv_path = args.csv or DEFAULT_LATENCY_CSV
        if not os.path.exists(csv_path):
            print(f'Error: {csv_path!r} not found.')
            sys.exit(1)
        plot_latency(csv_path)
