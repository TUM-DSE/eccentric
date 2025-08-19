import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

marker_styles = {
    'gross': 'o',
    'bacon': '+',
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

backend_rename_map = {
    "real_willow": "Willow",
    "real_infleqtion": "Infleqtion",
    "real_nsinfleqtion": "Infleqtion w/o\nshuttling",
    "real_apollo": "Apollo",
    "real_flamingo": "Flamingo"
}

code_palette = sns.color_palette("pastel", n_colors=6)
code_hatches = ["/", "\\", "//", "++", "xx", "**"]

def generate_size_plot(df_path):
    df = pd.read_csv(df_path)
    error_types = ['Constant', 'SI1000']
    error_probs = [0.004]
    df_filtered = df[~df['code'].str.contains('heavyhex', case=False, na=False)]
    df_filtered = df_filtered[~df_filtered['backend'].str.contains('heavyhex', case=False, na=False)]
    backends = df_filtered['backend'].unique()

    os.makedirs("../data", exist_ok=True)

    for backend in backends:
        n_rows = 1
        n_cols = len(error_probs) * len(error_types)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5), sharey=True)
        if n_cols == 1:
            axes = [axes]
        all_handles_labels = []

        for idx, (i, p) in enumerate(enumerate(error_probs)):
            for j, et in enumerate(error_types):
                ax = axes[idx * len(error_types) + j]
                original_et = error_type_map.get(et, et.lower())
                subset = df_filtered[
                    (df_filtered['backend'] == backend) &
                    (df_filtered['error_type'] == original_et) &
                    (df_filtered['error_probability'] == p)
                ]

                for code, group in subset.groupby('code'):
                    code_key = code.lower()
                    code_display = code_rename_map.get(code_key, code.capitalize())
                    marker = marker_styles.get(code_key, marker_styles['other'])
                    group_sorted = group.sort_values('backend_size')

                    line = ax.plot(
                        group_sorted['backend_size'],
                        group_sorted['logical_error_rate'],
                        label=code_display,
                        marker=marker
                    )

                    if code_key == 'gross':
                        line_color = line[0].get_color()
                        gross_div12 = group_sorted['logical_error_rate'] / 12
                        ax.plot(
                            group_sorted['backend_size'],
                            gross_div12,
                            linestyle='--',
                            color=line_color,
                            label='Gross / 12'
                        )

                ax.set_xlabel('Backend Size')
                xticks = sorted(subset['backend_size'].unique())
                ax.set_xticks(xticks)
                ax.set_title(f'{et}', loc='left', fontsize=11, fontweight='bold')

                if idx == 0 and j == 0:
                    ax.set_ylabel('Logical Error Rate')
                    ax.text(1.0, 1.05, 'Lower is better ↓', transform=ax.transAxes,
                            fontsize=10, fontweight='bold', color="blue", va='top', ha='right')
                elif idx == 0 and j == 1:
                    ax.text(1.0, 1.05, 'Lower is better ↓', transform=ax.transAxes,
                            fontsize=10, fontweight='bold', color="blue", va='top', ha='right')

                ax.grid(True)
                ax.set_ylim(0, 1.05)
                handles, labels = ax.get_legend_handles_labels()
                all_handles_labels.extend(zip(handles, labels))

        unique_labels = {}
        for h, l in all_handles_labels:
            if l not in unique_labels:
                unique_labels[l] = h

        fig.legend(
            handles=list(unique_labels.values()),
            labels=list(unique_labels.keys()),
            title='Code',
            fontsize='small',
            title_fontsize='small',
            loc='upper right',
            bbox_to_anchor=(0.95, 0.88)
        )

        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plt.savefig(f"data/size_{backend}.pdf")
        plt.close(fig)

def generate_connectivity_plot(df_path):
    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
    backend_order = ['custom_grid', 'custom_cube', 'custom_full']
    df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 3))

    codes = sorted(df["code"].unique())
    backends = backend_order
    x = np.arange(len(backends))
    bar_width = 0.15

    for i, code in enumerate(codes):
        subset = df[df["code"] == code]
        means = []
        stds = []

        for backend in backends:
            row = subset[subset["backend"] == backend]
            if not row.empty:
                means.append(row["logical_error_rate"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        plt.bar(
            x + i * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            color=code_palette[i % len(code_palette)],
            hatch=code_hatches[i % len(code_hatches)],
            edgecolor="black",
            label=code
        )

    plt.xticks(x + bar_width * (len(codes) - 1) / 2, [b.replace("custom_", "").capitalize() for b in backends])
    #plt.xlabel("Backend Connectivity")
    plt.ylabel("Logical Error Rate (Log)")
    plt.title("Logical Error Rate by Backend Connectivity", loc='left', fontweight='bold')
    plt.yscale("log")
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(1.15, 1.0),
        #ncols=len(codes),
        fontsize='small',
        frameon=True
    )
    plt.text(1.00, 1.09, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=10, fontweight='bold', color="blue", va='top', ha='right')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/connectivity.pdf", format="pdf")
    plt.close()

def generate_topology_plot(df_path):
    df = pd.read_csv(df_path)
    df["backend"] = df["backend"].replace(backend_rename_map)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 4))

    backends = sorted(df["backend"].unique())
    codes = sorted(df["code"].unique())
    x = np.arange(len(backends))
    bar_width = 0.15

    for i, code in enumerate(codes):
        subset = df[df["code"] == code]
        means = []
        stds = []

        for backend in backends:
            row = subset[subset["backend"] == backend]
            if not row.empty:
                means.append(row["logical_error_rate"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        plt.bar(
            x + i * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            color=code_palette[i % len(code_palette)],
            hatch=code_hatches[i % len(code_hatches)],
            edgecolor="black",
            label=code
        )

    plt.xticks(x + bar_width * (len(codes) - 1) * 3/5, backends, rotation=0, ha="right")
    #plt.xlabel("Backend")
    plt.ylabel("Logical Error Rate (Log)")
    plt.title("Logical Error Rate by Backend Topology", loc='left', fontweight='bold')
    plt.yscale("log")
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    plt.legend(
        title="QEC Code",
        loc='lower center',
        bbox_to_anchor=(1.06, 0.4),
        #ncol=len(codes),
        fontsize='small',
        #frameon=False
    )
    plt.text(1.00, 1.07, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=10, fontweight='bold', color="blue", va='top', ha='right')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/topology.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def generate_technology_plot(path):
    technologies = ["Willow", "Apollo", "Infleqtion", "DQC"]
    dfs = []

    for tech in technologies:
        tech_path = os.path.join(path, tech, "results.csv")
        df = pd.read_csv(tech_path)
        df["backend"] = df["backend"].replace(backend_rename_map)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Normalize code labels
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # Ensure consistent order
    backends = sorted(df["backend"].unique())
    codes = sorted(df["code"].unique())

    df["backend"] = pd.Categorical(df["backend"], categories=backends, ordered=True)
    df["code"] = pd.Categorical(df["code"], categories=codes, ordered=True)

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 3))

    x = np.arange(len(backends))
    bar_width = 0.8 / len(codes)

    for i, code in enumerate(codes):
        means = []
        for backend in backends:
            subset = df[(df["backend"] == backend) & (df["code"] == code)]
            if not subset.empty:
                means.append(subset["logical_error_rate"].values[0])
            else:
                means.append(0)

        plt.bar(
            x + i * bar_width,
            means,
            width=bar_width,
            color=code_palette[i % len(code_palette)],
            hatch=code_hatches[i % len(code_hatches)],
            edgecolor="black",
            label=code
        )

    plt.xticks(x + bar_width * (len(codes) - 1) * 3/4, backends, rotation=0, ha="right")
    #plt.xlabel("Backend")
    plt.ylabel("Logical Error Rate (Log)")
    plt.title("Logical Error Rate by Backend and QEC Code", loc='left', fontweight='bold')
    plt.yscale("log")
    plt.legend(
        title="QEC Code",
        loc='lower center',
        bbox_to_anchor=(1.065, 0.15),
        #ncol=len(codes),
        fontsize='small',
        #frameon=False
    )

    plt.text(1.00, 1.08, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=10, fontweight='bold', color="blue", va='top', ha='right')

    plt.subplots_adjust(bottom=0.25)  # Reserve space for rotated labels + legend
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/technologies.pdf", format="pdf")
    plt.close()

def generate_dqc_plot(path):
    datasets = [
        ("DQC_LOWER", "DQC Full Size"),
        ("DQC_1_QPU_LOWER", "DQC 1 QPU Size")
    ]
    dfs = []

    for folder, label in datasets:
        tech_path = os.path.join(path, folder, "results.csv")
        df = pd.read_csv(tech_path)
        df["backend"] = df["backend"].replace(backend_rename_map)
        df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))
        df["dataset"] = label
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Use codes as X-axis
    codes = sorted(df["code"].unique())
    datasets_labels = [label for _, label in datasets]

    df["code"] = pd.Categorical(df["code"], categories=codes, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=datasets_labels, ordered=True)

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 3))

    x = np.arange(len(codes))
    bar_width = 0.35  # narrower bars for side-by-side plotting

    # Loop through datasets (two bars per code)
    for j, dataset_label in enumerate(datasets_labels):
        means = []
        for code in codes:
            subset = df[(df["code"] == code) & (df["dataset"] == dataset_label)]
            if not subset.empty:
                means.append(subset["corrected_error_rate"].mean())  # mean over backends
            else:
                means.append(0)

        plt.bar(
            x + (j - 0.5) * bar_width,
            means,
            width=bar_width,
            color=code_palette[0 % len(code_palette)] if j == 0 else code_palette[1 % len(code_palette)],
            hatch='/' if j == 0 else '\\',
            edgecolor="black",
            label=dataset_label
        )

    plt.xticks(x, codes, rotation=0, ha="center", fontsize=12)
    plt.ylabel("Logical Error Rate (Log)", fontsize=12)
    plt.title("Logical Error Rate by QEC Code and Patch Size", loc='left', fontweight='bold', fontsize=14)
    plt.yscale("log")

    plt.legend(
        title="Patch Size",
        loc='lower center',
        bbox_to_anchor=(1.10, 0.55),
        #ncol=2,
        fontsize=12,
        #frameon=False
    )

    plt.text(1.00, 1.10, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', color="blue", va='top', ha='right')

    plt.subplots_adjust(bottom=0.25)
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/dqc.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def generate_logical_by_code_plot(df_path):
    pass

def generate_swap_overhead_plot(df_path, backend_label, total_columns=3):

    def format_code(code):
        code = code.lower()
        return {
            "hh": "Heavy-Hex",
            "surface": "Surface",
            "color": "Color"
        }.get(code, code.capitalize())

    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(format_code)
    
    routing_methods = df["routing_method"].dropna().unique()
    layout_methods = df["layout_method"].dropna().unique()
    codes = sorted(df["code"].unique())

    n_routing = len(routing_methods)
    n_cols = total_columns
    n_rows = int(np.ceil(n_routing / n_cols))

    plot_width_per_col = 6
    plot_height_per_row = 5

    bar_width = 0.2
    palette = sns.color_palette("pastel", n_colors=len(layout_methods))
    #hatches = ['/', 'o', '*', '\\', '-']
    hatches = ['/', '\\', '//', 'o', '-']
    layout_styles = {
        layout: (palette[i % len(palette)], hatches[i % len(hatches)])
        for i, layout in enumerate(layout_methods)
    }

    #fig, axes = plt.subplots(n_rows, n_cols, figsize=(plot_width_per_col * n_cols, plot_height_per_row * n_rows), sharey=True, constrained_layout=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18,3), sharey=True, constrained_layout=True)
    axes = axes.flatten()

    for i, routing in enumerate(routing_methods):
        ax = axes[i]
        subset = df[df["routing_method"] == routing]

        x = np.arange(len(codes))
        for j, layout in enumerate(layout_methods):
            values = []
            errors = []
            for code in codes:
                entry = subset[(subset["layout_method"] == layout) & (subset["code"] == code)]
                mean = entry["swap_overhead_mean"].values[0] if not entry.empty else 0
                var = entry["swap_overhead_var"].values[0] if not entry.empty else 0
                values.append(mean)
                errors.append(np.sqrt(var))

            ax.bar(
                x + j * bar_width,
                values,
                width=bar_width,
                label=layout.capitalize(),
                color=layout_styles[layout][0],
                edgecolor="black",
                hatch=layout_styles[layout][1],
                yerr=errors,
                capsize=3,
            )

        ax.set_title(f"Routing: {routing.capitalize()}", fontsize=11, fontweight='bold', loc='left')
        ax.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax.set_xticklabels(codes, rotation=15, ha="center")
        #ax.set_xlabel("QEC Code")
        if i % n_cols == 0:
            ax.set_ylabel("SWAP Overhead")

        # ✅ Add horizontal grid lines
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    ylim = axes[0].get_ylim()
    xlim = axes[0].get_xlim()

    axes[0].text(
        xlim[1],
        ylim[1] * 1.08,
        "Lower is better ↓",
        fontsize=10, fontweight='bold', color="blue", va='top', ha='right'
    )
    axes[1].text(
        xlim[1],
        ylim[1] * 1.08,
        "Lower is better ↓",
        fontsize=10, fontweight='bold', color="blue", va='top', ha='right'
    )
    axes[2].text(
        xlim[1],
        ylim[1] * 1.08,
        "Lower is better ↓",
        fontsize=10, fontweight='bold', color="blue", va='top', ha='right'
    )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=layout_styles[layout][0],
                      edgecolor='black',
                      hatch=layout_styles[layout][1])
        for layout in layout_methods
    ]
    labels = [layout.capitalize() for layout in layout_methods]
    fig.legend(handles, labels, title="Layout Method", loc="lower center",    bbox_to_anchor=(1.043, 0.55), )

    #plt.title(f"SWAP Overhead on {backend_label} Architecture", fontsize=16, y=1.02, fontweight='bold', ha='left')
    #plt.tight_layout()
    plt.savefig("data/" + backend_label + "_swap_overhead.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def generate_plot_variance(df_path):
    df = pd.read_csv(df_path)  # Update with the correct file path

    # Ensure `backend` is categorical with proper order
    backend_order = ["variance_low", "variance_mid", "variance_high"]
    df["backend"] = pd.Categorical(df["backend"], categories=backend_order, ordered=True)
    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # Calculate standard deviation using Bernoulli trial std formula
    df["std"] = np.sqrt(df["logical_error_rate"] * (1 - df["logical_error_rate"]) / df["num_samples"])

    # Plot settings
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 3))

    # Unique codes sorted
    codes = sorted(df["code"].unique())
    x = np.arange(len(codes))
    bar_width = 0.2

    # Color palette
    palette = sns.color_palette("pastel", n_colors=3)
    hatches = ['/', '\\', '//']

    # Plot bars for each backend
    for i, backend in enumerate(backend_order):
        subset = df[df["backend"] == backend]
        means = []
        stds = []

        for code in codes:
            row = subset[subset["code"] == code]
            if not row.empty:
                means.append(row["logical_error_rate"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)

        plt.bar(
            x + i * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            color=palette[i],
            hatch=hatches[i],
            edgecolor="black",
            label=backend.replace("variance_", "").capitalize()
        )

    # Axes and labels
    plt.xticks(x + bar_width, codes, rotation=0, ha="center")
    #plt.xlabel("Quantum Error Correction Code")
    plt.ylabel("Logical Error Rate")
    plt.title("Logical Error Rate by Code (with Std Dev)", loc='left', fontweight='bold')
    plt.legend(title="Variance", loc='lower center' , bbox_to_anchor=(1.055, 0.5), fontsize='small', frameon=True)
    plt.text(1.00, 1.08, 'Lower is better ↓', transform=plt.gca().transAxes,
             fontsize=10, fontweight='bold', color="blue", va='top', ha='right')
    plt.tight_layout()

    # Save / Show
    plt.savefig("data/plot_variance.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def generate_normalized_gate_ovehead(df_path):
    df = pd.read_csv(df_path)
    method_label_map = {
        "tket": "tket",
        "qiskit": "qiskit",
        "qiskit_optimized": "qiskit_optimized",
    }
    df["translating_method"] = df["translating_method"].map(method_label_map)
    
    gate_sets = ["ibm_heron", "h2"]
    translation_methods = ["tket", "qiskit", "qiskit_optimized"]
    df = df[df["gate_set"].isin(gate_sets) & df["translating_method"].isin(translation_methods)]

    df["code"] = df["code"].apply(lambda x: code_rename_map.get(x.lower(), x.capitalize()))

    # Plot settings
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 3), sharey=True)

    # Set pastel palette base color
    base_palette = sns.color_palette("pastel", n_colors=2)
    qiskit_color = base_palette[0]  # shared base for both Qiskits
    tket_color = base_palette[1]
    optimized_qiskit_color = tuple(min(1, c + 0.2) for c in qiskit_color)

    # Colors and hatches
    color_map = {
        "tket": tket_color,
        "qiskit": qiskit_color,
        "qiskit_optimized": optimized_qiskit_color,
    }
    hatches = ["\\", "/", "//"]  # different hatches per method

    # Sort and format code labels
    codes = sorted(df["code"].unique())

    # Plot bars with hatching
    bar_width = 0.25
    x = np.arange(len(codes))
    method_labels = {
        "tket": "TKET",
        "qiskit": "Qiskit",
        "qiskit_optimized": "Qiskit Optimized"
    }
    for i, gate_set in enumerate(gate_sets):
        ax = axes[i]
        subset = df[df["gate_set"] == gate_set]

        pivot = subset.pivot(index="code", columns="translating_method", values="gate_overhead_mean")
        total_gates = subset.pivot(index="code", columns="translating_method", values="original_total_gates")

        # Normalize gate overhead by total gates
        for method in translation_methods:
            if method in pivot.columns and method in total_gates.columns:
                pivot[method] = pivot[method] / total_gates[method]

        for j, method in enumerate(translation_methods):
            values = pivot[method].values
            bars = ax.bar(
                x + j * bar_width,
                values,
                width=bar_width,
                color=color_map[method],
                hatch=hatches[j % len(hatches)],
                edgecolor="black",
                label=method_labels[method]
            )

        ax.set_title(f"Normalized Gate Overhead ({'IBM Heron' if gate_set == 'ibm_heron' else 'H2'})", fontsize=14, fontweight='bold', loc='left')
        #ax.set_xlabel("Quantum Error Correction Code")
        axes[0].set_ylabel("Normalized Gate Overhead")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(codes, rotation=0, ha="center")

        # Add "Lower is better ↓" in top-left corner
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        axes[0].text(
            xlim[1],                
            ylim[1] * 1.08,         # near top
            "Lower is better ↓",
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            color="blue"
        )

        axes[1].text(
            xlim[1],                
            ylim[1] * 1.08,         # near top
            "Lower is better ↓",
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            color="blue"
        )

    # Legend and layout
    axes[1].legend(labels=['TKET', 'Qiskit', 'Qiskit Optimized'])
    plt.tight_layout()

    # Uncomment to save
    plt.savefig("data/translation.pdf", format="pdf", bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    size = "experiment_results/Size_full/results.csv"
    connectivity = "experiment_results/Connectivity_small/results.csv"
    topology = "experiment_results/Topology/results.csv"
    path = "experiment_results"
    df_grid = "experiment_results/Routing_grid/results.csv"
    df_hh = "experiment_results/Routing_hh/results.csv"
    plot_variance = "experiment_results/Variance/results.csv"
    gate_overhead = "experiment_results/Translation/results.csv"
    generate_size_plot(size)
    generate_connectivity_plot(connectivity)
    generate_topology_plot(topology)
    generate_technology_plot(path)
    generate_dqc_plot(path)
    generate_swap_overhead_plot(df_grid, "Grid")
    generate_swap_overhead_plot(df_hh, "Heavy-Hex")
    generate_plot_variance(plot_variance)
    generate_normalized_gate_ovehead(gate_overhead)
