import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
WIDE_FIGSIZE = 6
HEIGHT_FIGSIZE = 2
FONTSIZE = 12
BAR_WIDTH = 0.2
group_spacing = 0.4
code_palette = sns.color_palette("pastel", n_colors=8)

def plot_four_panel(csv_a1, csv_a2, csv_b1, csv_b2, patch_csv, mcm_error_csv):

    # Read CSVs
    df_mcm_a = pd.read_csv(csv_a1)   # MCMit
    df_herq_a = pd.read_csv(csv_a2)  # HERQULES
    df_mcm_b = pd.read_csv(csv_b1)   # MCMit
    df_herq_b = pd.read_csv(csv_b2)  # HERQULES
    df_patch = pd.read_csv(patch_csv)
    df_mcm_err = pd.read_csv(mcm_error_csv)

    for df in [df_mcm_a, df_herq_a, df_mcm_b, df_herq_b, df_patch, df_mcm_err]:
        df['logical_error_rate'] = pd.to_numeric(df['logical_error_rate'])

    fig, axes = plt.subplots(
        1, 4,
        figsize=(WIDE_FIGSIZE * 2.8, HEIGHT_FIGSIZE * 1.3)
    )

    LOWER_TEXT_Y = 1.25

    # ==================================================
    # (a) Comparison 1 (MCMit vs HERQULES)
    # ==================================================
    distances = sorted(df_mcm_a['distance'].unique())

    for i, d in enumerate(distances):
        subset = df_mcm_a[df_mcm_a['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[0].plot(
            x_vals,
            subset['logical_error_rate'],
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"MCMit d={d}"
        )

        subset = df_herq_a[df_herq_a['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[0].plot(
            x_vals,
            subset['logical_error_rate'],
            '--s',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"HERQULES d={d}"
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Duration (ns)", fontsize=FONTSIZE)
    axes[0].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[0].set_title("(a) IBM Heron", fontsize=FONTSIZE, fontweight="bold")
    axes[0].grid(color='gray')

    # X-ticks for plot a
    durations_a = sorted(set(
        df_mcm_a.get('mcm_latency_ns', df_mcm_a.get('measure_latency_ns')).tolist() +
        df_herq_a.get('mcm_latency_ns', df_herq_a.get('measure_latency_ns')).tolist()
    ))
    axes[0].set_xticks(durations_a)
    axes[0].set_xticklabels([str(int(d)) for d in durations_a])

    # ==================================================
    # (b) Comparison 2 (MCMit vs HERQULES)
    # ==================================================
    distances = sorted(df_mcm_b['distance'].unique())

    for i, d in enumerate(distances):
        subset = df_mcm_b[df_mcm_b['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[1].plot(
            x_vals,
            subset['logical_error_rate'],
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"MCMit d={d}"
        )

        subset = df_herq_b[df_herq_b['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[1].plot(
            x_vals,
            subset['logical_error_rate'],
            '--s',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"HERQULES d={d}"
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Duration (ns)", fontsize=FONTSIZE)
    axes[1].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[1].set_title("(b) Futuristic noise model", fontsize=FONTSIZE, fontweight="bold")
    axes[1].grid(color='gray')

    # X-ticks for plot b
    durations_b = sorted(set(
        df_mcm_b.get('mcm_latency_ns', df_mcm_b.get('measure_latency_ns')).tolist() +
        df_herq_b.get('mcm_latency_ns', df_herq_b.get('measure_latency_ns')).tolist()
    ))
    axes[1].set_xticks(durations_b)
    axes[1].set_xticklabels([str(int(d)) for d in durations_b])

    # ==================================================
    # (c) Patches vs duration (MCMit)
    # ==================================================
    patch_vals = sorted(df_patch['num_patches'].unique())
    distances = sorted(df_patch['distance'].unique())

    for i, dist in enumerate(distances):
        y_vals = []
        for n in patch_vals:
            subset = df_patch[
                (df_patch['num_patches'] == n) &
                (df_patch['distance'] == dist)
            ]
            y_vals.append(
                subset['logical_error_rate'].values[0]
                if not subset.empty else np.nan
            )

        axes[2].plot(
            patch_vals,
            y_vals,
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"d = {int(dist)}"
        )

    axes[2].set_yscale("log")
    axes[2].set_xlabel("Number of QEC patches", fontsize=FONTSIZE)
    axes[2].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[2].set_title("(c) QEC patches", fontsize=FONTSIZE, fontweight="bold")
    axes[2].grid(color='gray')

    # X-ticks for plot c
    axes[2].set_xticks(patch_vals)
    axes[2].set_xticklabels([str(int(p)) for p in patch_vals])
    axes[2].legend(fontsize=FONTSIZE-2,
                   loc="center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.46),
    frameon=False)
    # ==================================================
    # (d) MCM error vs latency (MCMit)
    # ==================================================
    x_vals = sorted(df_mcm_err['mcm_error'].unique())
    latencies = sorted(df_mcm_err['measure_latency_ns'].unique())

    for i, lat in enumerate(latencies):
        y_vals = []
        for xm in x_vals:
            subset = df_mcm_err[
                (df_mcm_err['mcm_error'] == xm) &
                (df_mcm_err['measure_latency_ns'] == lat)
            ]
            y_vals.append(subset['logical_error_rate'].values[0] if not subset.empty else np.nan)

        axes[3].plot(
            x_vals,
            y_vals,
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"{int(lat)} ns"
        )

    axes[3].set_yscale("log")
    axes[3].set_xscale("log")
    axes[3].set_xlabel("MCM error probability", fontsize=FONTSIZE)
    axes[3].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[3].set_title("(d) QEC threshold", fontsize=FONTSIZE, fontweight="bold")
    axes[3].grid(color='gray')
    axes[3].text(2.55, 1.25, "Lower is better ↓",
                 transform=axes[0].transAxes,
                 fontsize=FONTSIZE, fontweight="bold",
                 color="blue", va="top", ha="center")
    xmin, xmax = axes[3].get_xlim()
    ymin, ymax = axes[3].get_ylim()

    # Determine overlap range
    low = max(xmin, ymin)
    high = min(xmax, ymax)

    # Generate diagonal line
    x_diag = np.logspace(np.log10(low), np.log10(high), 200)
    axes[3].plot(
        x_diag,
        x_diag,
        linestyle='--',
        linewidth=2,
        color='black',
        label='Threshold'
    )
    axes[3].legend(fontsize=FONTSIZE-2,
                   loc="center",
    ncol=3,
    bbox_to_anchor=(0.48, -0.55),
    frameon=False
                   )

    # ==================================================
    # Legends between plots
    # ==================================================
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    #handles_c, labels_c = axes[3].get_legend_handles_labels()

    # Legend between (a) and (b)
    fig.legend(
        handles_a, labels_a,
        loc="center",
        ncol=3,
        bbox_to_anchor=(0.28, 0.08),
        fontsize=FONTSIZE - 2,
        frameon=False
    )

    # Legend between (c) and (d)
    #fig.legend(
    #    handles_c, labels_c,
    #    loc="center",
    #    ncol=2,
    #    bbox_to_anchor=(0.78, 0.08),
    #    fontsize=FONTSIZE - 2,
    #    frameon=False
    #)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('gray')  # change border color to gray
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.35, top=0.85, wspace=0.35)
    fig.patch.set_edgecolor("blue")
    fig.patch.set_linewidth(3)
    plt.savefig("../data/mcm_qec_evaluation.pdf", format="pdf")
    plt.close(fig)

plot_four_panel("../experiment_results/mcm_tradeoff_mcm_ibm.csv", "../experiment_results/mcm_tradeoff_herqules_ibm.csv", "../experiment_results/mcm_tradeoff_mcm_futuristic_2.csv", "../experiment_results/mcm_tradeoff_herqules_futuristic_2.csv", "../experiment_results/mcm_patch_dist.csv", "../experiment_results/mcm_error_latency_ibm.csv")
print("rendered ../data/mcm_qec_evaluation.pdf")
