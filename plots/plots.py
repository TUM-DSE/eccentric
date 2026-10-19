from utils import *


def generate_size_plot(df_path):
    df = pd.read_csv(df_path)
    error_types = ["SI1000", "Constant"]
    error_probs = [0.002, 0.004, 0.008]
    backend = "custom_full"

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_codes = sorted([c.lower() for c in df["code"].unique()])
    code_color = {
        c: default_colors[i % len(default_colors)] for i, c in enumerate(all_codes)
    }

    letters = ["a)", "b)", "c)", "d)", "e)", "f)"]

    n_rows, n_cols = len(error_types), len(error_probs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(WIDE_FIGSIZE * 2, HEIGHT_FIGSIZE * 1.7),
        sharey="row",
        sharex="col",
        gridspec_kw={"wspace": 0.1, "hspace": 0.25},
    )

    axes = np.array(axes).reshape(n_rows, n_cols)

    unique_labels = {}

    for row, et in enumerate(error_types):
        original_et = error_type_map.get(et, et.lower())

        all_xticks = sorted(
            df.loc[
                (df["backend"] == backend) & (df["error_type"] == original_et),
                "backend_size",
            ].unique()
        )

        for col, p in enumerate(error_probs):
            ax = axes[row, col]

            subset = df[
                (df["backend"] == backend)
                & (df["error_type"] == original_et)
                & (df["error_probability"] == p)
            ]

            for code, group in subset.groupby("code"):
                code_key = code.lower()
                code_display = code_rename_map.get(code_key, code.capitalize())
                marker = marker_styles.get(code_key, marker_styles["other"])
                group_sorted = group.sort_values("backend_size")

                xs = group_sorted["backend_size"].to_numpy()
                ys = group_sorted["logical_error_rate"].to_numpy()

                line = ax.plot(
                    xs,
                    ys,
                    label=code_display,
                    marker=marker,
                    color=code_color[code_key],
                    markeredgecolor="none",
                )[0]

                highlight_x = HIGHLIGHT.get((code_key, et), [])
                if highlight_x:
                    sel = np.isin(xs, highlight_x)
                    ax.plot(
                        xs[sel],
                        ys[sel],
                        linestyle="None",
                        marker=marker,
                        markersize=line.get_markersize() * 1.4,
                        markerfacecolor=code_color[code_key],
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        color=code_color[code_key],
                        zorder=line.get_zorder() + 2,
                        label="_nolegend_",
                    )

            ax.set_xticks(all_xticks)
            ax.set_xticklabels(all_xticks, fontsize=FONTSIZE - 2)

            letter = letters[row * n_cols + col]
            title_et = "Const." if et == "Constant" else et
            ax.set_title(
                f"{letter} {title_et} {p}", loc="left", fontsize=12, fontweight="bold"
            )

            if col == 0:
                ax.set_ylabel("Logical Err. Rate", fontsize=FONTSIZE)

            ax.text(
                1.0,
                1.16,
                "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="blue",
                va="top",
                ha="right",
            )

            ax.grid(True)
            ax.set_ylim(0, 0.65)

            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in unique_labels and l != "_nolegend_":
                    unique_labels[l] = h

    plt.subplots_adjust(left=0.07, right=0.95, bottom=0.15, top=0.92)

    fig.legend(
        handles=list(unique_labels.values()),
        labels=list(unique_labels.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(unique_labels),
        frameon=False,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/size.pdf", format="pdf")
    plt.close(fig)


def generate_size_plot_two(df_path):
    df = pd.read_csv(df_path)
    error_types = ["SI1000", "Constant"]
    backend = "custom_full"
    for error_prob in [0.002, 0.004, 0.008]:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        all_codes = sorted([c.lower() for c in df["code"].unique()])
        code_color = {
            c: default_colors[i % len(default_colors)] for i, c in enumerate(all_codes)
        }

        letters = ["a)", "b)"]

        n_rows, n_cols = 1, 2
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(WIDE_FIGSIZE * 2, HEIGHT_FIGSIZE),
            sharey=True,
            sharex=False,
            gridspec_kw={"wspace": 0.05},
        )

        unique_labels = {}

        for col, et in enumerate(error_types):
            ax = axes[col]
            original_et = error_type_map.get(et, et.lower())

            all_xticks = sorted(
                df.loc[
                    (df["backend"] == backend) & (df["error_type"] == original_et),
                    "backend_size",
                ].unique()
            )

            subset = df[
                (df["backend"] == backend)
                & (df["error_type"] == original_et)
                & (df["error_probability"] == error_prob)
            ]

            for code, group in subset.groupby("code"):
                code_key = code.lower()
                code_display = code_rename_map.get(code_key, code.capitalize())
                marker = marker_styles.get(code_key, marker_styles["other"])
                group_sorted = group.sort_values("backend_size")

                xs = group_sorted["backend_size"].to_numpy()
                ys = group_sorted["logical_error_rate"].to_numpy()

                line = ax.plot(
                    xs,
                    ys,
                    label=code_display,
                    marker=marker,
                    color=code_color[code_key],
                    markeredgecolor="none",
                )[0]

                highlight_x = HIGHLIGHT.get((code_key, et), [])
                if highlight_x:
                    sel = np.isin(xs, highlight_x)
                    ax.plot(
                        xs[sel],
                        ys[sel],
                        linestyle="None",
                        marker=marker,
                        markersize=line.get_markersize() * 1.4,
                        markerfacecolor=code_color[code_key],
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        color=code_color[code_key],
                        zorder=line.get_zorder() + 2,
                        label="_nolegend_",
                    )

            ax.set_xticks(all_xticks)
            ax.set_xticklabels(all_xticks, fontsize=FONTSIZE - 2)
            ax.set_ylim(0, 0.65)
            ax.grid(True)

            letter = letters[col]
            title_et = "Uniform" if et == "Constant" else et
            ax.set_title(
                f"{letter} {title_et}, p={error_prob}",
                loc="left",
                fontsize=12,
                fontweight="bold",
            )

            if col == 0:
                ax.set_ylabel("Logical Error Rate", fontsize=FONTSIZE)
            ax.set_xlabel("Backend Size", fontsize=FONTSIZE)
            ax.text(
                1.0,
                1.12,
                "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="blue",
                va="top",
                ha="right",
            )

            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in unique_labels and l != "_nolegend_":
                    unique_labels[l] = h

        plt.subplots_adjust(left=0.08, right=0.88, bottom=0.2, top=0.88)
        fig.legend(
            handles=list(unique_labels.values()),
            labels=list(unique_labels.keys()),
            loc="lower center",
            bbox_to_anchor=(0.94, 0.2),
            frameon=False,
        )

        os.makedirs("data", exist_ok=True)
        plt.savefig(f"data/size_{error_prob}.pdf", format="pdf")
        plt.close(fig)


def generate_connectivity_topology_plot(connectivity_csv, topology_csv):
    def preprocess(df_path, backend_order=None, reverse_backends=False):
        df = pd.read_csv(df_path)
        df = df[df["backend"] != "real_infleqtion"]
        df["code"] = df["code"].apply(
            lambda x: code_rename_map.get(x.lower(), x.capitalize())
        )

        if backend_order is not None:
            df["backend"] = pd.Categorical(
                df["backend"], categories=backend_order, ordered=True
            )
        else:
            df["backend"] = df["backend"].replace(backend_rename_map)

        if reverse_backends:
            backends = df["backend"].unique()[::-1]
        else:
            backends = df["backend"].unique()

        codes = sorted(df["code"].unique())
        return df, codes, backends

    df_conn, codes_conn, backends_conn = preprocess(
        connectivity_csv, backend_order=["custom_grid", "custom_cube", "custom_full"]
    )
    df_topo, codes_topo, backends_topo = preprocess(topology_csv, reverse_backends=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE * 0.9), sharey=True
    )

    bar_width = 0.2
    group_spacing = 0.25
    palette = sns.color_palette(
        "pastel", n_colors=max(len(codes_conn), len(codes_topo))
    )
    hatches = code_hatches

    def plot_subplot(ax, df, codes, backends, title):
        x = np.arange(len(backends)) * (bar_width * len(codes) + group_spacing)

        for i, code in enumerate(codes):
            subset = df[df["code"] == code]
            means, stds = [], []

            for backend in backends:
                row = subset[subset["backend"] == backend]

                if not row.empty:
                    val = row["logical_error_rate"].values[0]
                    n = row["num_samples"].values[0]

                    if val == 0:
                        val = 1 / n

                    std = np.sqrt(val * (1 - val) / n)

                    means.append(val)
                    stds.append(std)
                else:
                    means.append(np.nan)
                    stds.append(0)

            ax.bar(
                x + i * bar_width,
                means,
                yerr=stds,
                width=bar_width,
                color=palette[i % len(palette)],
                hatch=hatches[i % len(hatches)],
                edgecolor="black",
                label=code,
            )

        ax.set_xticks(x + bar_width * (len(codes) - 1) / 2)
        ax.set_xticklabels(
            [str(b).replace("custom_", "").capitalize() for b in backends],
            fontsize=FONTSIZE - 2,
        )

        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)

        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")

        ax.text(
            1.0,
            1.14,
            "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="blue",
            va="top",
            ha="right",
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

    plot_subplot(axes[0], df_conn, codes_conn, backends_conn, "a) Artificial Topology")
    plot_subplot(axes[1], df_topo, codes_topo, backends_topo, "b) Real Topology")

    axes[0].set_ylabel("Logical Err. Rate (log)", fontsize=FONTSIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        fontsize=FONTSIZE,
        frameon=False,
    )

    plt.subplots_adjust(left=0.08, right=0.85, top=0.85, bottom=0.12, wspace=0.1)

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/connectivity.pdf", format="pdf")
    plt.close(fig)


def generate_technology_plot(path):
    technologies = ["Willow", "Apollo", "Infleqtion"]
    raw_dfs = [
        pd.read_csv(os.path.join(path, tech, "results.csv")) for tech in technologies
    ]
    df = pd.concat(raw_dfs, ignore_index=True)
    df["backend"] = df["backend"].replace(backend_rename_map)
    df["code"] = df["code"].apply(
        lambda x: code_rename_map.get(x.lower(), x.capitalize())
    )
    backends = sorted(df["backend"].unique())
    codes = sorted(df["code"].unique())
    n_backends = len(backends)
    n_codes = len(codes)
    x = np.arange(n_backends) * (BAR_WIDTH * n_codes + group_spacing)
    fig, ax = plt.subplots(1, 1, figsize=(1.2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE))
    for j, code in enumerate(codes):
        code_subset = df[df["code"] == code]
        means = []
        for backend in backends:
            row = code_subset[code_subset["backend"] == backend]
            means.append(row["logical_error_rate"].values[0] if not row.empty else 0)
        ax.bar(
            x + j * BAR_WIDTH,
            means,
            width=BAR_WIDTH,
            color=code_palette[j % len(code_palette)],
            hatch=code_hatches[j % len(code_hatches)],
            edgecolor="black",
            label=code,
        )
    ax.set_xticks(x + BAR_WIDTH * (n_codes - 1) / 2)
    ax.set_xticklabels(backends, fontsize=FONTSIZE - 2)
    ax.set_ylabel("Logical Err. Rate (log)", fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Technologies", loc="left", fontsize=12, fontweight="bold")
    ax.text(
        1.0,
        1.14,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="blue",
        va="top",
        ha="right",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.01, 0.52),
        fontsize=FONTSIZE - 2,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.84, 1])
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/technologies.pdf", format="pdf")
    plt.close(fig)


def generate_dqc_plot(dqc_files):
    fig, axes = plt.subplots(
        1, 4, figsize=(2.05 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True
    )

    if len(dqc_files) == 1:
        axes = np.array(axes)

    for file_idx, csv_path in enumerate(dqc_files):
        df = pd.read_csv(csv_path)
        df = df[df["routing_method"] != "basic"]
        df = df[df["code"] != "gross"]

        df["backend"] = df["backend"].replace(backend_rename_map)
        df["code"] = df["code"].apply(
            lambda x: code_rename_map.get(x.lower(), x.capitalize())
        )

        device = csv_path.split("/")[1].split("_")[1]

        for i, err_label in enumerate(["normal", "downscaled"]):
            ax_idx = file_idx * 2 + i
            ax = axes[ax_idx]

            if err_label == "normal":
                sub_df = df[~df["error_type"].str.contains("downscaled", na=False)]
            else:
                sub_df = df[df["error_type"].str.contains("downscaled", na=False)]

            codes = sorted(sub_df["code"].unique())
            backends = sorted(sub_df["backend"].unique())
            n_codes = len(codes)
            n_backends = len(backends)

            x = np.arange(n_codes)
            total_bar_width = BAR_WIDTH * n_backends
            offsets = np.linspace(
                -total_bar_width / 2 + BAR_WIDTH / 2,
                total_bar_width / 2 - BAR_WIDTH / 2,
                n_backends,
            )

            for j, backend in enumerate(backends):
                means, stds = [], []
                for code in codes:
                    row = sub_df[
                        (sub_df["code"] == code) & (sub_df["backend"] == backend)
                    ]
                    if not row.empty:
                        mean_val = row["logical_error_rate"].mean()
                        means.append(mean_val)
                        stds.append(
                            np.sqrt(
                                mean_val * (1 - mean_val) / row["num_samples"].sum()
                            )
                        )
                    else:
                        means.append(0)
                        stds.append(0)

                ax.bar(
                    x + offsets[j],
                    means,
                    yerr=stds,
                    width=BAR_WIDTH,
                    color=code_palette[j % len(code_palette)],
                    hatch=code_hatches[j % len(code_hatches)],
                    edgecolor="black",
                    label=backend,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")
            if ax_idx == 0:
                ax.set_ylabel("Logical Err. Rate (log)", fontsize=FONTSIZE)
            ax.set_yscale("log")
            ax.grid(axis="y")
            ax.set_axisbelow(True)

            ax.spines["top"].set_color("black")
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_color("black")
            ax.spines["right"].set_color("black")

            plot_label = f"{chr(97 + 2*file_idx + i)})"
            device_label = "Fl" if device == "Flamingo" else "NH"
            error_pct = "" if err_label == "normal" else "10%"
            ax.set_title(
                f"{plot_label} {device_label}. {error_pct}",
                loc="left",
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                1.02,
                1.16,
                "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="blue",
                va="top",
                ha="right",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.904, 0.82),
        fontsize=FONTSIZE,
        frameon=False,
        ncol=2,
        columnspacing=0.7,
    )

    plt.subplots_adjust(left=0.08, bottom=0.27, right=0.995, wspace=0.05)
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/dqc.pdf", format="pdf")
    plt.close(fig)


def generate_swap_overhead_plot(df_path, backend_label, total_columns=3):
    df = pd.read_csv(df_path)

    def format_code(code):
        code = code.lower()
        return {
            "hh": "Heavy-Hex",
            "surface": "Surface",
            "color": "Color",
            "bacon": "Bacon-Shor",
        }.get(code, code.capitalize())

    df["code"] = df["code"].apply(format_code)

    routing_methods = df["routing_method"].dropna().unique()
    layout_methods = df["layout_method"].dropna().unique()
    codes = sorted(df["code"].unique())

    n_routing = len(routing_methods)
    n_cols = total_columns
    n_rows = int(np.ceil(n_routing / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(WIDE_FIGSIZE * 2, HEIGHT_FIGSIZE),
        sharey=True,
        gridspec_kw={"wspace": 0.1},
    )
    axes = np.array(axes).reshape(-1)

    bar_width = 0.2
    hatches = ["/", "\\", "//", "o", "-"]
    layout_styles = {
        layout: (code_palette[i % len(code_palette)], hatches[i % len(hatches)])
        for i, layout in enumerate(layout_methods)
    }

    letters = [f"({chr(97+i)})" for i in range(len(routing_methods))]

    def sci_formatter(y, _pos):
        if y == 0:
            return "0"
        s = f"{y:.2e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mant}e{exp_int}"

    func_formatter = FuncFormatter(sci_formatter)

    for i, routing in enumerate(routing_methods):
        ax = axes[i]
        subset = df[df["routing_method"] == routing]

        x = np.arange(len(codes))
        for j, layout in enumerate(layout_methods):
            values = []
            errors = []
            for code in codes:
                entry = subset[
                    (subset["layout_method"] == layout) & (subset["code"] == code)
                ]
                if not entry.empty:
                    mean = entry["swap_overhead_mean"].values[0]
                    var = entry["swap_overhead_var"].values[0]
                else:
                    mean, var = 0, 0
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

        ax.set_title(
            f"{chr(97 + i)}) {routing.capitalize()}",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")

        if i % n_cols == 0:
            ax.set_ylabel("SWAP Overhead", fontsize=FONTSIZE)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        ax.yaxis.set_major_formatter(func_formatter)
        try:
            ax.yaxis.get_offset_text().set_visible(False)
        except Exception:
            pass

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    for ax in axes:
        if ax is not None and ax.has_data():
            ax.text(
                1.0,
                1.14,
                "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="blue",
                va="top",
                ha="right",
            )

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.3)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=layout_styles[layout][0],
            edgecolor="black",
            hatch=layout_styles[layout][1],
        )
        for layout in layout_methods
    ]
    labels = [layout.capitalize() for layout in layout_methods]

    axes[0].legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.18, 0.31),
        ncol=1,
        fontsize=FONTSIZE,
        frameon=True,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig(f"data/{backend_label}_swap_overhead.pdf", format="pdf")
    plt.close(fig)


def generate_swap_overhead_norm_plot(df_path, backend_label, total_columns=3):

    def format_code(code):
        code = code.lower()
        return {
            "hh": "Heavy-Hex",
            "surface": "Surface",
            "color": "Color",
            "bacon": "Bacon-Shor",
        }.get(code, code.capitalize())

    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(format_code)

    df_og_gates = pd.read_csv("experiment_results/Translation/results.csv")
    df_og_gates["code"] = df_og_gates["code"].apply(format_code)

    og_gate_dict = dict(zip(df_og_gates["code"], df_og_gates["original_2q_gates"]))

    routing_methods = df["routing_method"].dropna().unique()
    layout_methods = df["layout_method"].dropna().unique()
    codes = sorted(df["code"].unique())

    n_routing = len(routing_methods)
    n_cols = total_columns
    n_rows = int(np.ceil(n_routing / n_cols))

    bar_width = 0.2
    hatches = ["/", "\\", "//", "o", "-"]
    layout_styles = {
        layout: (code_palette[i % len(code_palette)], hatches[i % len(hatches)])
        for i, layout in enumerate(layout_methods)
    }

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE),
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.flatten()

    for i, routing in enumerate(routing_methods):
        ax = axes[i]
        subset = df[df["routing_method"] == routing]

        x = np.arange(len(codes))
        for j, layout in enumerate(layout_methods):
            values = []
            errors = []
            for code in codes:
                entry = subset[
                    (subset["layout_method"] == layout) & (subset["code"] == code)
                ]
                mean = (
                    entry["tq_gate_overhead_mean"].values[0] if not entry.empty else 0
                )
                var = entry["tq_gate_overhead_var"].values[0] if not entry.empty else 0

                og_val = og_gate_dict.get(code, np.nan)
                if not np.isnan(og_val) and og_val > 0:
                    mean = (mean / og_val) * 100
                    var = (var / (og_val**2)) * (100**2)
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

        ax.set_title(
            f"{chr(100 + i)}) {routing.capitalize()}",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )

        ax.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")
        axes[0].tick_params(axis="y", labelsize=FONTSIZE - 2)

        if i % n_cols == 0:
            ax.set_ylabel("% TQ Gates", fontsize=FONTSIZE)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        ax.text(
            1.0,
            1.14,
            "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="blue",
            va="top",
            ha="right",
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=layout_styles[layout][0],
            edgecolor="black",
            hatch=layout_styles[layout][1],
        )
        for layout in layout_methods
    ]
    labels = [layout.capitalize() for layout in layout_methods]
    axes[0].legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.17, 0.33),
        ncol=1,
        fontsize=FONTSIZE,
        frameon=True,
    )

    plt.savefig(f"data/{backend_label}_swap_overhead_norm.pdf", format="pdf")
    plt.close()


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def generate_compact_swap_plots(df_path, backend_label):
    def format_code(code):
        code = code.lower()
        return {"hh": "Heavy-Hex", "surface": "Surface", "color": "Color"}.get(
            code, code.capitalize()
        )

    df = pd.read_csv(df_path)
    df["code"] = df["code"].apply(format_code)

    df_og_gates = pd.read_csv("experiment_results/Translation/results.csv")
    df_og_gates["code"] = df_og_gates["code"].apply(format_code)
    og_gate_dict = dict(zip(df_og_gates["code"], df_og_gates["original_2q_gates"]))

    routing_methods = df["routing_method"].dropna().unique()
    layout_methods = df["layout_method"].dropna().unique()
    codes = sorted(df["code"].unique())

    n_routing = len(routing_methods)
    bar_width = 0.2
    hatches = ["/", "\\", "//", "o", "-"]
    layout_styles = {
        layout: (code_palette[i % len(code_palette)], hatches[i % len(hatches)])
        for i, layout in enumerate(layout_methods)
    }

    fig, axes = plt.subplots(
        2,
        n_routing,
        figsize=(2 * WIDE_FIGSIZE, 1.7 * HEIGHT_FIGSIZE),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(2, n_routing)

    def sci_formatter(y, _pos):
        if y == 0:
            return "0"
        s = f"{y:.2e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mant}e{exp_int}"

    func_formatter = FuncFormatter(sci_formatter)

    for col, routing in enumerate(routing_methods):
        subset = df[df["routing_method"] == routing]
        x = np.arange(len(codes))

        ax_abs = axes[0, col]
        for j, layout in enumerate(layout_methods):
            values, errors = [], []
            for code in codes:
                entry = subset[
                    (subset["layout_method"] == layout) & (subset["code"] == code)
                ]
                if not entry.empty:
                    mean = entry["swap_overhead_mean"].values[0]
                    var = entry["swap_overhead_var"].values[0]
                else:
                    mean, var = 0, 0
                values.append(mean)
                errors.append(np.sqrt(var))

            ax_abs.bar(
                x + j * bar_width,
                values,
                width=bar_width,
                color=layout_styles[layout][0],
                edgecolor="black",
                hatch=layout_styles[layout][1],
                yerr=errors,
                capsize=3,
                label=layout.capitalize(),
            )

        ax_abs.set_title(
            f"{chr(97+col)}) {routing.capitalize()}",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax_abs.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax_abs.set_xticklabels(codes, rotation=30, ha="right", fontsize=FONTSIZE - 2)
        if col == 0:
            ax_abs.set_ylabel("SWAP Overhead", fontsize=FONTSIZE)
        ax_abs.grid(axis="y")
        ax_abs.set_axisbelow(True)
        ax_abs.yaxis.set_major_formatter(func_formatter)
        ax_abs.text(
            1.0,
            1.16,
            "Lower is better ↓",
            transform=ax_abs.transAxes,
            fontsize=12,
            fontweight="bold",
            color="blue",
            va="top",
            ha="right",
        )

        ax_norm = axes[1, col]
        for j, layout in enumerate(layout_methods):
            values, errors = [], []
            for code in codes:
                entry = subset[
                    (subset["layout_method"] == layout) & (subset["code"] == code)
                ]
                mean = (
                    entry["tq_gate_overhead_mean"].values[0] if not entry.empty else 0
                )
                var = entry["tq_gate_overhead_var"].values[0] if not entry.empty else 0
                og_val = og_gate_dict.get(code, np.nan)
                if not np.isnan(og_val) and og_val > 0:
                    mean = (mean / og_val) * 100
                    var = (var / (og_val**2)) * (100**2)
                values.append(mean)
                errors.append(np.sqrt(var))

            ax_norm.bar(
                x + j * bar_width,
                values,
                width=bar_width,
                color=layout_styles[layout][0],
                edgecolor="black",
                hatch=layout_styles[layout][1],
                yerr=errors,
                capsize=3,
            )

        ax_norm.set_title(
            f"{chr(100+col)}) {routing.capitalize()}",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax_norm.set_xticks(x + (bar_width * (len(layout_methods) - 1)) / 2)
        ax_norm.set_xticklabels(codes, rotation=30, ha="right", fontsize=FONTSIZE - 2)
        if col == 0:
            ax_norm.set_ylabel("% TQ Gates", fontsize=FONTSIZE)
        ax_norm.grid(axis="y")
        ax_norm.set_axisbelow(True)
        ax_norm.text(
            1.0,
            1.16,
            "Lower is better ↓",
            transform=ax_norm.transAxes,
            fontsize=12,
            fontweight="bold",
            color="blue",
            va="top",
            ha="right",
        )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=layout_styles[l][0],
            edgecolor="black",
            hatch=layout_styles[l][1],
        )
        for l in layout_methods
    ]
    labels = [l.capitalize() for l in layout_methods]
    axes[0, 0].legend(
        handles,
        labels,
        loc="upper left",
        fontsize=FONTSIZE,
        frameon=True,
        ncols=3,
        columnspacing=0.6,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig(f"data/{backend_label}_routing.pdf", format="pdf")
    plt.close(fig)


def generate_normalized_gate_overhead(df_path):

    df = pd.read_csv(df_path)
    method_label_map = {
        "tket": "tket",
        "tket_optimized": "tket_optimized",
        "qiskit": "qiskit",
        "qiskit_optimized": "qiskit_optimized",
    }
    df["translating_method"] = df["translating_method"].map(method_label_map)

    gate_sets = ["ibm_heron", "h2"]
    df = df[
        df["gate_set"].isin(gate_sets) & df["translating_method"].isin(method_label_map)
    ]
    df["code"] = df["code"].apply(
        lambda x: code_rename_map.get(x.lower(), x.capitalize())
    )

    qiskit_base, tket_base = code_palette[0], code_palette[1]
    qiskit_opt = tuple(min(1, c + 0.25) for c in qiskit_base)
    tket_opt = tuple(min(1, c + 0.25) for c in tket_base)

    color_map = {
        "tket": tket_base,
        "tket_optimized": tket_opt,
        "qiskit": qiskit_base,
        "qiskit_optimized": qiskit_opt,
    }
    hatches = ["\\", "//", "/", "\\\\"]

    layout_styles = {
        method: (color_map[method], hatches[i % len(hatches)])
        for i, method in enumerate(method_label_map)
    }
    method_labels = {
        "tket": "TKET",
        "tket_optimized": "TKET Opt.",
        "qiskit": "Qiskit",
        "qiskit_optimized": "Qiskit Opt.",
    }

    methods = ["qiskit", "qiskit_optimized", "tket", "tket_optimized"]

    fig, axes = plt.subplots(
        1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True
    )

    for i, gate_set in enumerate(gate_sets):
        ax = axes[i]
        subset = df[df["gate_set"] == gate_set]
        if subset.empty:
            print(f"No data for {gate_set}, skipping.")
            continue

        codes = sorted(subset["code"].unique())
        x = np.arange(len(codes))

        pivot = subset.pivot(
            index="code", columns="translating_method", values="gate_overhead_mean"
        )
        total_gates = subset.pivot(
            index="code", columns="translating_method", values="original_total_gates"
        )

        for method in methods:
            if method in pivot.columns and method in total_gates.columns:
                pivot[method] = pivot[method] / total_gates[method]

        for j, method in enumerate(methods):
            if method not in pivot.columns:
                continue
            values = pivot.loc[codes, method].values
            ax.bar(
                x + j * BAR_WIDTH,
                values,
                width=BAR_WIDTH,
                label=method_labels[method],
                color=layout_styles[method][0],
                hatch=layout_styles[method][1],
                edgecolor="black",
            )

        device_label = "IBM Heron" if gate_set == "ibm_heron" else "H2"
        ax.set_title(
            f"{chr(97 + i)}) {device_label}", fontsize=12, fontweight="bold", loc="left"
        )

        ax.set_xticks(x + (BAR_WIDTH * (len(methods) - 1)) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=FONTSIZE - 2)
        if i == 0:
            ax.set_ylabel("Norm. Overhead", fontsize=FONTSIZE)

        ax.grid(axis="y")
        ax.set_ylim(0, 12)
        ax.set_axisbelow(True)

        ax.text(
            1.0,
            1.14,
            "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="blue",
            va="top",
            ha="right",
        )

    plt.subplots_adjust(left=0.08, bottom=0.3, right=0.86, wspace=0.15)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=layout_styles[m][0],
            edgecolor="black",
            hatch=layout_styles[m][1],
        )
        for m in methods
    ]
    labels = [method_labels[m] for m in methods]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.86, 0.55),
        fontsize=FONTSIZE,
        frameon=False,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/gate_overhead_normalized.pdf", format="pdf")
    plt.close(fig)


def generate_gate_overhead(df_path):

    df = pd.read_csv(df_path)
    method_label_map = {
        "tket": "tket",
        "tket_optimized": "tket_optimized",
        "qiskit": "qiskit",
        "qiskit_optimized": "qiskit_optimized",
    }
    df["translating_method"] = df["translating_method"].map(method_label_map)

    gate_sets = ["ibm_heron", "h2"]
    df = df[
        df["gate_set"].isin(gate_sets) & df["translating_method"].isin(method_label_map)
    ]
    df["code"] = df["code"].apply(
        lambda x: code_rename_map.get(x.lower(), x.capitalize())
    )

    qiskit_base, tket_base = code_palette[0], code_palette[1]
    qiskit_opt = tuple(min(1, c + 0.25) for c in qiskit_base)
    tket_opt = tuple(min(1, c + 0.25) for c in tket_base)

    color_map = {
        "tket": tket_base,
        "tket_optimized": tket_opt,
        "qiskit": qiskit_base,
        "qiskit_optimized": qiskit_opt,
    }
    hatches = ["\\", "//", "/", "\\\\"]

    layout_styles = {
        method: (color_map[method], hatches[i % len(hatches)])
        for i, method in enumerate(method_label_map)
    }
    method_labels = {
        "tket": "TKET",
        "tket_optimized": "TKET Opt.",
        "qiskit": "Qiskit",
        "qiskit_optimized": "Qiskit Opt.",
    }

    methods = ["qiskit", "tket", "qiskit_optimized", "tket_optimized"]

    fig, axes = plt.subplots(
        1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True
    )

    for i, gate_set in enumerate(gate_sets):
        ax = axes[i]
        subset = df[df["gate_set"] == gate_set]
        if subset.empty:
            print(f"No data for {gate_set}, skipping.")
            continue

        codes = sorted(subset["code"].unique())
        x = np.arange(len(codes))
        pivot = subset.pivot(
            index="code", columns="translating_method", values="gate_overhead_mean"
        )

        for j, method in enumerate(methods):
            if method not in pivot.columns:
                continue
            values = pivot.loc[codes, method].values
            ax.bar(
                x + j * BAR_WIDTH,
                values,
                width=BAR_WIDTH,
                label=method_labels[method],
                color=layout_styles[method][0],
                hatch=layout_styles[method][1],
                edgecolor="black",
            )

        device_label = "IBM Heron" if gate_set == "ibm_heron" else "H2"
        ax.set_title(
            f"{chr(97 + i)}) {device_label}", fontsize=12, fontweight="bold", loc="left"
        )

        ax.set_xticks(x + (BAR_WIDTH * (len(methods) - 1)) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=FONTSIZE - 2)
        if i == 0:
            ax.set_ylabel("Gate Overhead", fontsize=FONTSIZE)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        ax.text(
            1.0,
            1.14,
            "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="blue",
            va="top",
            ha="right",
        )

    plt.subplots_adjust(left=0.08, bottom=0.3, right=0.86, wspace=0.15)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=layout_styles[m][0],
            edgecolor="black",
            hatch=layout_styles[m][1],
        )
        for m in methods
    ]
    labels = [method_labels[m] for m in methods]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.87, 0.55),
        fontsize=FONTSIZE,
        frameon=False,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/gate_overhead.pdf", format="pdf")
    plt.close(fig)


def generate_overhead_2x2(df_path):
    df = pd.read_csv(df_path)
    method_label_map = {
        "tket": "tket",
        "tket_optimized": "tket_optimized",
        "qiskit": "qiskit",
        "qiskit_optimized": "qiskit_optimized",
    }
    df["translating_method"] = df["translating_method"].map(method_label_map)

    gate_sets = ["ibm_heron", "h2"]
    df = df[
        df["gate_set"].isin(gate_sets) & df["translating_method"].isin(method_label_map)
    ]
    df["code"] = df["code"].apply(
        lambda x: code_rename_map.get(x.lower(), x.capitalize())
    )

    qiskit_base, tket_base = code_palette[0], code_palette[1]
    qiskit_opt = tuple(min(1, c + 0.25) for c in qiskit_base)
    tket_opt = tuple(min(1, c + 0.25) for c in tket_base)

    color_map = {
        "tket": tket_base,
        "tket_optimized": tket_opt,
        "qiskit": qiskit_base,
        "qiskit_optimized": qiskit_opt,
    }
    hatches = ["\\", "//", "/", "\\\\"]

    layout_styles = {
        method: (color_map[method], hatches[i % len(hatches)])
        for i, method in enumerate(method_label_map)
    }
    method_labels = {
        "tket": "TKET",
        "tket_optimized": "TKET Opt.",
        "qiskit": "Qiskit",
        "qiskit_optimized": "Qiskit Opt.",
    }

    methods = ["qiskit", "qiskit_optimized", "tket", "tket_optimized"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(2 * WIDE_FIGSIZE, 1.7 * HEIGHT_FIGSIZE),
        sharex="col",
        sharey="row",
    )

    letters = ["a)", "b)", "c)", "d)"]

    for row, mode in enumerate(["overhead", "normalized"]):
        for col, gate_set in enumerate(gate_sets):
            ax = axes[row, col]
            subset = df[df["gate_set"] == gate_set]
            if subset.empty:
                print(f"no data for {gate_set}, skipping.")
                continue

            codes = sorted(subset["code"].unique())
            x = np.arange(len(codes))

            pivot = subset.pivot(
                index="code", columns="translating_method", values="gate_overhead_mean"
            )

            if mode == "normalized":
                total_gates = subset.pivot(
                    index="code",
                    columns="translating_method",
                    values="original_total_gates",
                )
                for method in methods:
                    if method in pivot.columns and method in total_gates.columns:
                        pivot[method] = pivot[method] / total_gates[method]

            for j, method in enumerate(methods):
                if method not in pivot.columns:
                    continue
                values = pivot.loc[codes, method].values
                ax.bar(
                    x + j * BAR_WIDTH,
                    values,
                    width=BAR_WIDTH,
                    label=method_labels[method],
                    color=layout_styles[method][0],
                    hatch=layout_styles[method][1],
                    edgecolor="black",
                )

            device_label = "IBM Heron" if gate_set == "ibm_heron" else "H2"
            ylabel = "Gate Overhead" if mode == "overhead" else "Norm. Overhead"
            letter = letters[row * 2 + col]

            ax.set_title(
                f"{letter} {device_label}", fontsize=12, fontweight="bold", loc="left"
            )

            if row == 1:
                ax.set_xticks(x + (BAR_WIDTH * (len(methods) - 1)) / 2)
                ax.set_xticklabels(
                    codes, fontsize=FONTSIZE - 2, rotation=30, ha="right"
                )
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=FONTSIZE - 2)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=FONTSIZE)

            ax.grid(axis="y")
            ax.set_axisbelow(True)

            if mode == "normalized":
                ax.set_ylim(0, 12)

            ax.text(
                1.0,
                1.14,
                "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="blue",
                va="top",
                ha="right",
            )

    plt.subplots_adjust(
        left=0.08, bottom=0.22, right=0.98, top=0.92, wspace=0.15, hspace=0.25
    )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=layout_styles[m][0],
            edgecolor="black",
            hatch=layout_styles[m][1],
        )
        for m in methods
    ]
    labels = [method_labels[m] for m in methods]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.56, 0.87),
        fontsize=FONTSIZE,
        frameon=False,
        ncol=4,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.3,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/gate_overhead.pdf", format="pdf")
    plt.close(fig)


def generate_variance_two(decoherence_csv, readout_csv):
    bar_width = 0.2
    group_spacing = 0.25
    hatches = ["/", "\\", "//", "o"]

    def preprocess_deco(df):
        df = df[df["error_type"] == "variance"].copy()
        df["code"] = df["code"].apply(
            lambda x: code_rename_map.get(x.lower(), x.capitalize())
        )
        df["std"] = np.sqrt(
            df["logical_error_rate"]
            * (1 - df["logical_error_rate"])
            / df["num_samples"]
        )
        df["backend_plot"] = df["backend"].str.replace(r"_\d+$", "", regex=True)
        return df

    def preprocess_readout(df):
        df = df.dropna(subset=["error_type", "logical_error_rate"]).copy()
        df["backend_plot"] = df["error_type"]
        df["code"] = df["code"].apply(
            lambda x: code_rename_map.get(x.lower(), x.capitalize())
        )
        df["std"] = np.sqrt(
            df["logical_error_rate"]
            * (1 - df["logical_error_rate"])
            / df["num_samples"]
        )
        return df

    df_deco = preprocess_deco(pd.read_csv(decoherence_csv))
    df_read = preprocess_readout(pd.read_csv(readout_csv))

    df_high = df_deco[df_deco["backend"].str.endswith("_1", na=False)]
    df_low = df_deco[df_deco["backend"].str.endswith("_10", na=False)]

    codes = sorted(set(df_deco["code"]).union(df_read["code"]))

    baseline_rows = df_deco[df_deco["backend_plot"] == "variance_none"]
    if baseline_rows.empty:
        raise ValueError("No variance_none baseline found in decoherence CSV")

    baseline_mean = baseline_rows.groupby("code")["logical_error_rate"].mean()
    baseline_std = baseline_rows.groupby("code")["std"].apply(
        lambda x: np.sqrt((x**2).sum())
    )

    rows = []
    for c in codes:
        if c not in baseline_mean:
            raise ValueError(f"No baseline measured for code '{c}'")
        rows.append(
            {
                "code": c,
                "backend_plot": "variance_none",
                "logical_error_rate": baseline_mean[c],
                "std": baseline_std[c],
            }
        )
    df_read = pd.concat([pd.DataFrame(rows), df_read], ignore_index=True)

    backend_order = ["variance_none", "variance_low", "variance_mid", "variance_high"]
    backend_labels = {
        "variance_none": "None",
        "variance_low": "Low",
        "variance_mid": "Mid",
        "variance_high": "High",
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE),
        sharey=True,
        gridspec_kw={"wspace": 0.1},
    )

    def plot_subplot(ax, df, title, show_labels=False):
        n_b = len(backend_order)
        x = np.arange(len(codes)) * (bar_width * n_b + group_spacing)
        for i, backend in enumerate(backend_order):
            sub = df[df["backend_plot"] == backend]
            means, stds = [], []
            for code in codes:
                r = sub[sub["code"] == code]
                if r.empty:
                    raise ValueError(
                        f"Missing value for code '{code}' and backend '{backend}'"
                    )
                mean_val = r["logical_error_rate"].mean()
                std_val = np.sqrt((r["std"] ** 2).sum())
                means.append(mean_val)
                stds.append(std_val)
            ax.bar(
                x + i * bar_width,
                means,
                yerr=stds,
                width=bar_width,
                color=code_palette[i % len(code_palette)],
                hatch=hatches[i % len(hatches)],
                edgecolor="black",
                label=backend_labels[backend] if show_labels else None,
            )
        ax.set_xticks(x + bar_width * (n_b - 1) / 2)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", y=0.97)
        ax.text(
            1.0,
            1.16,
            "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="blue",
            ha="right",
            va="top",
        )

    plot_subplot(axes[0], df_high, "a) Deco.", show_labels=True)
    axes[0].set_ylabel("Logical Error Rate (log)", fontsize=FONTSIZE, labelpad=2, y=0.3)
    plot_subplot(axes[1], df_low, "b) Deco. (10×)")
    plot_subplot(axes[2], df_read, "c) Readout")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=FONTSIZE - 2,
        frameon=False,
    )


    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.4)
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/variance.pdf", format="pdf")
    plt.close(fig)


def generate_decoder_plot(df_path):
    for err_name in ["modsi1000", "constant"]:
        df = pd.read_csv(df_path)
        if "error_type" in df.columns:
            df = df[df["error_type"] == err_name]

        if "code_rename_map" in globals():
            df["code"] = df["code"].apply(
                lambda x: code_rename_map.get(x.lower(), x.capitalize())
            )
        else:
            df["code"] = df["code"].apply(lambda x: x.capitalize())

        if "decoder_map" in globals():
            df["decoder"] = df["decoder"].apply(lambda x: decoder_map.get(x, x))

        df["std"] = np.sqrt(
            df["logical_error_rate"]
            * (1 - df["logical_error_rate"])
            / df["num_samples"]
        )

        codes = sorted(df["code"].unique())
        decoders = sorted(df["decoder"].unique())
        n_codes = len(codes)
        n_decoders = len(decoders)

        colors = code_palette
        x = np.arange(n_codes)
        offsets = np.linspace(-BAR_WIDTH, BAR_WIDTH, n_decoders)

        panels = [
            ("logical_error_rate", "Logical Err. Rate (log)", True),
            ("time_decode", "Time [s] (log)", False),
            ("mem_decode_full_MB", "Memory [MB]", False),
        ]

        fig, axes = plt.subplots(
            1, 3, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharex=True
        )

        for ax, (column, ylabel, use_errorbars) in zip(axes, panels):

            for i, decoder in enumerate(decoders):
                means, stds = [], []

                for code in codes:
                    subset = df[(df["code"] == code) & (df["decoder"] == decoder)]

                    if not subset.empty:
                        means.append(subset[column].values[0])
                        stds.append(subset["std"].values[0] if use_errorbars else 0)
                    else:
                        means.append(0)
                        stds.append(0)

                        xpos = x[codes.index(code)] + offsets[i]
                        ax.text(
                            xpos,
                            0.08,
                            "×",
                            transform=ax.get_xaxis_transform(),
                            color="red",
                            fontsize=18,
                            ha="center",
                            va="top",
                            fontweight="bold",
                            clip_on=False,
                        )

                ax.bar(
                    x + offsets[i],
                    means,
                    yerr=stds if use_errorbars else None,
                    width=BAR_WIDTH,
                    label=decoder,
                    color=colors[i % len(colors)],
                    edgecolor="black",
                )

            ax.set_ylabel(ylabel, fontsize=FONTSIZE, labelpad=2)
            ax.grid(axis="y")

            ax.text(
                1.0,
                1.15,
                "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=FONTSIZE,
                fontweight="bold",
                color="blue",
                va="top",
                ha="right",
            )

        axes[0].set_yscale("log")
        axes[0].set_title(
            "a) Effectiveness", loc="left", fontsize=FONTSIZE, fontweight="bold"
        )
        axes[1].set_title("b) Time", loc="left", fontsize=FONTSIZE, fontweight="bold")
        axes[2].set_title("c) Memory", loc="left", fontsize=FONTSIZE, fontweight="bold")

        axes[1].set_yscale("log")

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=20, ha="right")

        axes[-1].legend(loc="upper right", frameon=True)


        plt.subplots_adjust(left=0.06, right=0.99, bottom=0.25, wspace=0.25)
        if err_name == "modsi1000":
            plt.savefig("data/decoder_general.pdf", format="pdf")
        else:
            plt.savefig("data/decoder_general_constant.pdf", format="pdf")
        plt.close(fig)


def generate_decoder_error_barplot(df_path):
    for err_name in ["phenomenological", "modsi1000"]:
        df = pd.read_csv(df_path)
        if "error_type" in df.columns:
            df = df[df["error_type"] == err_name]

        if "decoder_map" in globals():
            df["decoder"] = df["decoder"].apply(lambda x: decoder_map.get(x, x))
        if "code_rename_map" in globals():
            df["code"] = df["code"].apply(
                lambda x: code_rename_map.get(x.lower(), x.capitalize())
            )
        else:
            df["code"] = df["code"].apply(lambda x: x.capitalize())

        code_name = df["code"].iloc[0]

        decoders = sorted(df["decoder"].unique())
        error_probs = sorted(df["error_probability"].unique())
        distances = sorted(df["distance"].unique())

        n_decoders = len(decoders)
        n_probs = len(error_probs)
        n_dist = len(distances)

        colors = code_palette if "code_palette" in globals() else plt.cm.Set2.colors

        fig, axes = plt.subplots(
            1, n_decoders, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True
        )

        if n_decoders == 1:
            axes = [axes]

        for col, decoder in enumerate(decoders):
            ax = axes[col]
            subset = df[df["decoder"] == decoder]

            x = np.arange(n_probs)
            group_width = BAR_WIDTH * n_dist
            offsets = np.linspace(
                -group_width / 2 + BAR_WIDTH / 2,
                group_width / 2 - BAR_WIDTH / 2,
                n_dist,
            )

            for i, dist in enumerate(distances):
                values = []
                for p in error_probs:
                    sub = subset[
                        (subset["distance"] == dist)
                        & (subset["error_probability"] == p)
                    ]
                    if not sub.empty:
                        values.append(sub["logical_error_rate"].values[0])
                    else:
                        values.append(0)
                        ax.text(
                            x[error_probs.index(p)] + offsets[i],
                            1e-3,
                            "×",
                            color="red",
                            fontsize=14,
                            fontweight="bold",
                            ha="center",
                            va="bottom",
                        )

                ax.bar(
                    x + offsets[i],
                    values,
                    width=BAR_WIDTH,
                    label=f"d={dist}",
                    color=colors[i % len(colors)],
                    edgecolor="black",
                )

            ax.set_yscale("log")
            ax.grid(axis="y")
            ax.set_xticks(x)
            ax.set_xticklabels([str(p) for p in error_probs])
            ax.set_xlabel("Physical Error Probability", fontsize=FONTSIZE)
            if col == 0:
                ax.set_ylabel("Logical Err. Rate (log)", fontsize=FONTSIZE)

            ax.set_title(
                f"{chr(97 + col)}) {decoder}",
                fontsize=FONTSIZE,
                fontweight="bold",
                loc="left",
            )

            ax.text(
                0.85,
                1,
                "Lower is better ↓",
                transform=ax.transAxes,
                fontsize=FONTSIZE,
                fontweight="bold",
                color="blue",
                ha="center",
                va="bottom",
            )

            if col == 1:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(-0.01, 1.08),
                    frameon=False,
                )

        plt.tight_layout(rect=[0, 0, 1, 1])
        os.makedirs("data", exist_ok=True)
        plt.savefig(f"data/decoder_special_{err_name}.pdf", format="pdf")
        plt.close(fig)


def generate_decoder_plot_time(df_path):

    df = pd.read_csv(df_path)

    if "code_rename_map" in globals():
        df["code"] = df["code"].apply(
            lambda x: code_rename_map.get(x.lower(), x.capitalize())
        )
    else:
        df["code"] = df["code"].apply(lambda x: x.capitalize())

    if "decoder_map" in globals():
        df["decoder"] = df["decoder"].apply(lambda x: decoder_map.get(x, x))

    if {"logical_error_rate", "num_samples"} <= set(df.columns):
        df["std"] = np.sqrt(
            df["logical_error_rate"]
            * (1 - df["logical_error_rate"])
            / df["num_samples"]
        )
    else:
        df["std"] = 0

    error_types = ["modsi1000", "constant"]
    colors = code_palette
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True
    )

    for ax, etype in zip(axes, error_types):
        subset = df[df["error_type"].str.lower() == etype.lower()]
        if subset.empty:
            print(f"⚠️ No data for error_type={etype}")
            continue

        codes = sorted(subset["code"].unique())
        decoders = sorted(subset["decoder"].unique())
        n_codes = len(codes)
        n_decoders = len(decoders)
        x = np.arange(n_codes)
        offsets = np.linspace(-BAR_WIDTH, BAR_WIDTH, n_decoders)

        for i, decoder in enumerate(decoders):
            means, stds, times = [], [], []
            for code in codes:
                entry = subset[
                    (subset["code"] == code) & (subset["decoder"] == decoder)
                ]
                if not entry.empty:
                    means.append(entry["logical_error_rate"].values[0])
                    stds.append(entry["std"].values[0])
                    times.append(
                        entry["exec_time"].values[0] if "exec_time" in entry else np.nan
                    )
                else:
                    means.append(0)
                    stds.append(0)
                    times.append(np.nan)
                    xpos = x[codes.index(code)] + offsets[i]
                    ax.text(
                        xpos,
                        0.04,
                        "×",
                        color="red",
                        fontsize=16,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            bars = ax.bar(
                x + offsets[i],
                times,
                yerr=stds,
                width=BAR_WIDTH,
                label=decoder,
                color=colors[i % len(colors)],
                edgecolor="black",
            )

        if etype == "modsi1000":
            ax.set_title(
                f"SI1000 Noise", loc="left", fontsize=FONTSIZE, fontweight="bold"
            )
        elif etype == "constant":
            ax.set_title(
                f"Constant Noise", loc="left", fontsize=FONTSIZE, fontweight="bold"
            )
        ax.set_xticks(x)
        ax.set_xticklabels(codes, fontsize=FONTSIZE - 2, rotation=30, ha="right")

        ax.set_yscale("log")
        ax.grid(axis="y")

        ax.text(
            1.0,
            1.14,
            "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=FONTSIZE - 1,
            fontweight="bold",
            color="blue",
            va="top",
            ha="right",
        )
    axes[0].set_ylabel("Exec. Time [s]", fontsize=FONTSIZE)
    axes[1].legend(
        fontsize=FONTSIZE - 2,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.4, 0.8),
    )
    plt.subplots_adjust(left=0.08, bottom=0.3, right=0.85, top=0.88, wspace=0.15)
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/decoder_time.pdf", format="pdf")
    plt.close(fig)


def plot_time_and_memory_stacked(csv_common, csv_big, csv_more_shots):
    def sci_formatter(y, _pos):
        if y == 0:
            return "0"
        s = f"{y:.2e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mant}e{exp_int}"

    df_common = pd.read_csv(csv_common)
    df_big = pd.read_csv(csv_big)
    df_more = pd.read_csv(csv_more_shots)

    row_baseline = df_common[
        df_common["backend"].str.contains("full", case=False)
    ].iloc[0]
    row_grid = df_common[df_common["backend"].str.contains("grid", case=False)].iloc[0]
    row_big = df_big.iloc[0]
    row_more = df_more.iloc[0]

    cases = [
        ("Baseline", row_baseline),
        ("Bigger\nBackend", row_big),
        ("Sparser\nBackend", row_grid),
        ("More Shots", row_more),
    ]

    labels = [name for name, _ in cases]
    rows = [row for _, row in cases]
    x = np.arange(len(labels))
    width = 0.65

    time_cols = [
        "time_backend",
        "time_circuit",
        "time_transpile",
        "time_repr",
        "time_noise",
        "time_decode",
    ]
    time_cols_no_decode = [
        "time_backend",
        "time_circuit",
        "time_transpile",
        "time_repr",
        "time_noise",
    ]
    mem_cols = [
        "mem_backend_full_MB",
        "mem_circuit_full_MB",
        "mem_transpile_full_MB",
        "mem_repr_full_MB",
        "mem_noise_full_MB",
        "mem_decode_full_MB",
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(WIDE_FIGSIZE * 0.8, HEIGHT_FIGSIZE * 2), sharex=True
    )

    color_map = code_palette

    bottom = np.zeros(len(labels))
    axes[0].grid(axis="y")
    for i, col in enumerate(time_cols):
        label = col.replace("time_", "").capitalize()
        if label == "Repr":
            label = "Repr. Change"
        elif label == "Circuit":
            label = "Circuit Gen."
        elif label == "Backend":
            label = "Backend Gen."

        values = [row[col] for row in rows]
        axes[0].bar(
            x,
            values,
            width,
            bottom=bottom,
            label=label,
            color=color_map[i % len(color_map)],
            edgecolor="black",
            zorder=1,
        )
        bottom += values

    axes[0].set_title("a) Runtime", loc="left", fontsize=FONTSIZE, fontweight="bold")
    axes[0].set_ylabel("Time [s]", fontsize=FONTSIZE)
    axes[0].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    axes[0].text(
        1.01,
        1.18,
        "Lower is better ↓",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=FONTSIZE,
        color="blue",
        fontweight="bold",
    )
    axes[0].legend(
        loc="upper left",
        fontsize=FONTSIZE - 1,
        frameon=True,
        ncol=2,
        labelspacing=0.3,
        handletextpad=0.4,
        handlelength=1.2,
        borderpad=0.18,
    )

    bottom = np.zeros(len(labels))
    axes[1].grid(axis="y")
    for i, col in enumerate(time_cols_no_decode):
        label = col.replace("time_", "").capitalize()
        if label == "Repr":
            label = "Repr. Change"
        elif label == "Circuit":
            label = "Circuit Gen."
        elif label == "Backend":
            label = "Backend Gen."

        values = [row[col] for row in rows]
        axes[1].bar(
            x,
            values,
            width,
            bottom=bottom,
            label=label,
            color=color_map[i % len(color_map)],
            edgecolor="black",
            zorder=1,
        )
        bottom += values

    axes[1].set_title(
        "b) Runtime (w/o Decoding)", loc="left", fontsize=FONTSIZE, fontweight="bold"
    )
    axes[1].set_ylabel("Time [s]", fontsize=FONTSIZE)
    axes[1].text(
        1.01,
        1.18,
        "Lower is better ↓",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=FONTSIZE,
        color="blue",
        fontweight="bold",
    )

    axes[2].grid(axis="y")

    mem_values = [row["peak_memory_full_MB"] for row in rows]

    axes[2].bar(x, mem_values, width, color=color_map[7], edgecolor="black", zorder=1)

    axes[2].set_title(
        "c) Peak Memory Usage", loc="left", fontsize=FONTSIZE, fontweight="bold"
    )
    axes[2].set_ylabel("Memory [MB]", fontsize=FONTSIZE)

    axes[2].text(
        1.01,
        1.18,
        "Lower is better ↓",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=FONTSIZE,
        color="blue",
        fontweight="bold",
    )

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=FONTSIZE - 2, rotation=0, ha="center")

    plt.subplots_adjust(left=0.13, right=0.99, top=0.94, bottom=0.1, hspace=0.35)

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/program_stats.pdf", format="pdf")
    plt.close(fig)


def plot_time_and_memory(csv_common, csv_big, csv_more_shots):
    def sci_formatter(y, _pos):
        if y == 0:
            return "0"
        s = f"{y:.2e}"
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mant}e{exp_int}"

    df_common = pd.read_csv(csv_common)
    df_big = pd.read_csv(csv_big)
    df_more = pd.read_csv(csv_more_shots)

    row_baseline = df_common[
        df_common["backend"].str.contains("full", case=False)
    ].iloc[0]
    row_grid = df_common[df_common["backend"].str.contains("grid", case=False)].iloc[0]
    row_big = df_big.iloc[0]
    row_more = df_more.iloc[0]

    cases = [
        ("Baseline", row_baseline),
        ("Bigger\nBackend", row_big),
        ("Sparser\nBackend", row_grid),
        ("More Shots", row_more),
    ]

    labels = [name for name, _ in cases]
    rows = [row for _, row in cases]
    x = np.arange(len(labels))
    width = 0.65

    time_cols = [
        "time_backend",
        "time_circuit",
        "time_transpile",
        "time_repr",
        "time_noise",
        "time_decode",
    ]
    time_cols_no_decode = [
        "time_backend",
        "time_circuit",
        "time_transpile",
        "time_repr",
        "time_noise",
    ]
    mem_cols = [
        "mem_backend_full_MB",
        "mem_circuit_full_MB",
        "mem_transpile_full_MB",
        "mem_repr_full_MB",
        "mem_noise_full_MB",
        "mem_decode_full_MB",
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(WIDE_FIGSIZE * 2.2, HEIGHT_FIGSIZE), sharex=True
    )

    color_map = code_palette

    bottom = np.zeros(len(labels))
    axes[0].grid(axis="y")
    for i, col in enumerate(time_cols):
        label = col.replace("time_", "").capitalize()
        if label == "Repr":
            label = "Repr. Change"
        elif label == "Circuit":
            label = "Circuit Gen."
        elif label == "Backend":
            label = "Backend Gen."

        values = [row[col] for row in rows]
        axes[0].bar(
            x,
            values,
            width,
            bottom=bottom,
            label=label,
            color=color_map[i % len(color_map)],
            edgecolor="black",
            zorder=1,
        )
        bottom += values

    axes[0].set_title("a) Runtime", loc="left", fontsize=FONTSIZE, fontweight="bold")
    axes[0].set_ylabel("Time [s]", fontsize=FONTSIZE)
    axes[0].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    axes[0].text(
        1.01,
        1.14,
        "Lower is better ↓",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=FONTSIZE,
        color="blue",
        fontweight="bold",
    )
    axes[0].legend(
        loc="upper left",
        fontsize=FONTSIZE - 2,
        frameon=True,
        ncol=1,
        labelspacing=0.3,
        handletextpad=0.4,
        handlelength=1.2,
        borderpad=0.18,
    )

    bottom = np.zeros(len(labels))
    axes[1].grid(axis="y")
    for i, col in enumerate(time_cols_no_decode):
        label = col.replace("time_", "").capitalize()
        if label == "Repr":
            label = "Repr. Change"
        elif label == "Circuit":
            label = "Circuit Gen."
        elif label == "Backend":
            label = "Backend Gen."

        values = [row[col] for row in rows]
        axes[1].bar(
            x,
            values,
            width,
            bottom=bottom,
            label=label,
            color=color_map[i % len(color_map)],
            edgecolor="black",
            zorder=1,
        )
        bottom += values

    axes[1].set_title(
        "b) Runtime (w/o Dec.)", loc="left", fontsize=FONTSIZE, fontweight="bold"
    )
    axes[1].set_ylabel("Time [s]", fontsize=FONTSIZE)
    axes[1].text(
        1.01,
        1.14,
        "Lower is better ↓",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=FONTSIZE,
        color="blue",
        fontweight="bold",
    )

    axes[2].grid(axis="y")

    mem_values = [row["peak_memory_full_MB"] for row in rows]

    axes[2].bar(x, mem_values, width, color=color_map[7], edgecolor="black", zorder=1)

    axes[2].set_title(
        "c) Peak Mem. Usage", loc="left", fontsize=FONTSIZE, fontweight="bold"
    )
    axes[2].set_ylabel("Memory [MB]", fontsize=FONTSIZE)

    axes[2].text(
        1.01,
        1.14,
        "Lower is better ↓",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=FONTSIZE,
        color="blue",
        fontweight="bold",
    )

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=FONTSIZE - 2, rotation=0, ha="center")

    plt.subplots_adjust(left=0.05, right=0.99, top=0.89, bottom=0.2, hspace=0.35)

    os.makedirs("data", exist_ok=True)
    plt.savefig("data/program_stats.pdf", format="pdf")
    plt.close(fig)


def plot_threshold_per_code(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["error_probability", "corrected_error_rate", "code"])
    df = df.sort_values("error_probability")

    codes = sorted(df["code"].unique())
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    code_color = {
        c.lower(): default_colors[i % len(default_colors)] for i, c in enumerate(codes)
    }

    fig, ax = plt.subplots(figsize=(WIDE_FIGSIZE * 0.8, HEIGHT_FIGSIZE * 1.1))

    for code in codes:
        code_key = code.lower()
        sub = df[df["code"] == code]

        x = sub["error_probability"].values
        y = sub["corrected_error_rate"].values
        display_name = code_rename_map.get(code_key, code.capitalize())
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label=display_name,
            color=code_color[code_key],
            markeredgecolor="none",
        )

    ax.set_xscale("log")

    ax.set_xlabel("Physical Error Probability", fontsize=12, labelpad=0.01)
    ax.set_ylabel("Logical Error Rate", fontsize=12)
    ax.text(
        1.0,
        1.12,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="blue",
        va="top",
        ha="right",
    )

    ax.grid(True, which="both")

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h

    ax.legend(
        handles=list(unique_labels.values()),
        labels=list(unique_labels.keys()),
        loc="upper left",
        ncol=2,
        frameon=True,
        handletextpad=0.2,
        columnspacing=0.5,
    )
    ax.set_title(
        "Effectiveness of Codes", loc="left", fontsize=FONTSIZE, fontweight="bold"
    )
    plt.subplots_adjust(left=0.15, right=0.99, top=0.85, bottom=0.2)
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/thresholds.pdf", format="pdf")
    plt.close(fig)


def plot_gross_shot_comparison(csv_1000, csv_more):
    df_1000 = pd.read_csv(csv_1000)
    df_more = pd.read_csv(csv_more)

    df_1000 = df_1000[df_1000["code"] == "gross"]
    df_more = df_more[df_more["code"] == "gross"]

    common_probs = np.intersect1d(
        df_1000["error_probability"].unique(), df_more["error_probability"].unique()
    )

    df_1000 = df_1000[df_1000["error_probability"].isin(common_probs)]
    df_more = df_more[df_more["error_probability"].isin(common_probs)]

    df_1000 = df_1000.sort_values("error_probability")
    df_more = df_more.sort_values("error_probability")

    fig, ax = plt.subplots(figsize=(0.7 * WIDE_FIGSIZE, HEIGHT_FIGSIZE))

    ax.plot(
        df_1000["error_probability"],
        df_1000["corrected_error_rate"],
        marker="o",
        linewidth=2,
        markersize=6,
        label="Gross (1000 shots)",
    )

    ax.plot(
        df_more["error_probability"],
        df_more["corrected_error_rate"],
        marker="s",
        linewidth=2,
        markersize=6,
        label=f"Gross ({df_more['num_samples'].iloc[0]} shots)",
    )

    ax.set_xscale("log")

    ax.set_xlabel("Physical Error Probability", fontsize=FONTSIZE, labelpad=0.01)
    ax.set_ylabel("Logical Error Rate", fontsize=FONTSIZE)
    ax.set_title(
        "Precision of #shots", loc="left", fontsize=FONTSIZE, fontweight="bold"
    )

    ax.grid(True, which="both")

    ax.legend(fontsize=FONTSIZE, frameon=True)

    ax.text(
        1.0,
        1.145,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="blue",
        va="top",
        ha="right",
    )

    plt.subplots_adjust(left=0.15, right=0.99, top=0.85, bottom=0.2)
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/shot_comparison.pdf", format="pdf")
    plt.close(fig)


def generate_size_plot_separate(df_path):
    df = pd.read_csv(df_path)
    err_prob = 0.004
    backend = "custom_full"
    error_types = ["SI1000", "Constant"]

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_codes = sorted([c.lower() for c in df["code"].unique()])
    code_color = {
        c: default_colors[i % len(default_colors)] for i, c in enumerate(all_codes)
    }

    marker_styles_local = (
        marker_styles if "marker_styles" in globals() else {"other": "o"}
    )

    for et in error_types:
        original_et = error_type_map.get(et, et.lower())

        all_xticks = sorted(
            df.loc[
                (df["backend"] == backend) & (df["error_type"] == original_et),
                "backend_size",
            ].unique()
        )

        subset = df[
            (df["backend"] == backend)
            & (df["error_type"] == original_et)
            & (df["error_probability"] == err_prob)
        ]

        fig, ax = plt.subplots(1, 1, figsize=(WIDE_FIGSIZE, HEIGHT_FIGSIZE))

        for code, group in subset.groupby("code"):
            code_key = code.lower()
            code_display = code_rename_map.get(code_key, code.capitalize())
            marker = marker_styles_local.get(code_key, marker_styles_local["other"])
            group_sorted = group.sort_values("backend_size")

            xs = group_sorted["backend_size"].to_numpy()
            ys = group_sorted["logical_error_rate"].to_numpy()

            line = ax.plot(
                xs,
                ys,
                label=code_display,
                marker=marker,
                color=code_color[code_key],
                markeredgecolor="none",
            )[0]

            highlight_x = HIGHLIGHT.get((code_key, et), [])
            if highlight_x:
                sel = np.isin(xs, highlight_x)
                ax.plot(
                    xs[sel],
                    ys[sel],
                    linestyle="None",
                    marker=marker,
                    markersize=line.get_markersize() * 1.4,
                    markerfacecolor=code_color[code_key],
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    color=code_color[code_key],
                    zorder=line.get_zorder() + 2,
                    label="_nolegend_",
                )

        ax.set_xticks(all_xticks)
        ax.set_xticklabels(all_xticks, fontsize=FONTSIZE - 2)
        ax.set_ylim(0, 0.65)
        ax.grid(True)
        ax.set_ylabel("Logical Error Rate", fontsize=FONTSIZE)
        ax.set_xlabel("Backend Size", fontsize=FONTSIZE)
        ax.set_title(f"Size", fontsize=12, fontweight="bold", loc="left")

        ax.text(
            1.0,
            1.12,
            "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="blue",
            va="top",
            ha="right",
        )

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))

        fig.legend(
            handles=list(unique_labels.values()),
            labels=list(unique_labels.keys()),
            loc="center right",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )

        plt.subplots_adjust(left=0.15, right=0.78, top=0.85, bottom=0.2)

        os.makedirs("data", exist_ok=True)
        plt.savefig(f"data/size_{et}_{err_prob}.pdf", format="pdf")
        plt.close(fig)


if __name__ == "__main__":
    path = "experiment_results"
    size = "experiment_results/Size/results.csv"
    connectivity = "experiment_results/Connectivity/results.csv"
    topology = "experiment_results/Real_Topology/results.csv"
    df_grid = "experiment_results/Routing_grid/results.csv"
    df_hh = "experiment_results/Routing_hh/results.csv"
    variance_decoherence = "experiment_results/Variance_Decoherence/results.csv"
    variance_readout = "experiment_results/Variance_Readout/results.csv"
    gate_overhead = "experiment_results/Translation/results.csv"
    dqc_flamingo = "experiment_results/DQC_Flamingo/results.csv"
    dqc_nighthawk = "experiment_results/DQC_Nighthawk/results.csv"
    dqc = [dqc_flamingo, dqc_nighthawk]
    decoder_general = "experiment_results/Decoder/results.csv"
    decoder_special = "experiment_results/Decoder_Chromobius/results.csv"
    program_topology = "experiment_results/Program_stats_topology/results.csv"
    program_size = "experiment_results/Program_stats_size/results.csv"
    program_shots = "experiment_results/Program_stats_shots/results.csv"
    threshold = "experiment_results/Accumulated_Small/results.csv"
    threshold_more = "experiment_results/Accumulated_bck/results.csv"
    threshold_shots = "experiment_results/Accumulated/results.csv"
    generate_size_plot_two(size)
    generate_connectivity_topology_plot(connectivity, topology)
    generate_technology_plot(path)
    generate_dqc_plot(dqc)
    generate_variance_two(variance_decoherence, variance_readout)
    generate_compact_swap_plots(df_grid, "Grid")
    generate_compact_swap_plots(df_hh, "Heavy-Hex")
    generate_overhead_2x2(gate_overhead)
    generate_decoder_plot(decoder_general)
    generate_decoder_error_barplot(decoder_special)
    plot_time_and_memory(program_topology, program_size, program_shots)
    plot_threshold_per_code(threshold)
    plot_gross_shot_comparison(threshold_more, threshold_shots)
