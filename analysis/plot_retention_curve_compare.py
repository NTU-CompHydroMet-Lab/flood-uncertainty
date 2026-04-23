import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from path_defaults import DEFAULT_ANALYSIS_ROOT, DEFAULT_FIGURE_ROOT

DEFAULT_OUTPUT_DIR = DEFAULT_FIGURE_ROOT

REQUIRED_COLUMNS = {"retention_rate", "iou", "precision", "recall"}
METRIC_ORDER = ["iou", "precision", "recall"]
METRIC_LABELS = {
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
}
GRID_ALPHA = 0.15
MODEL_COLORS = {
    "EDL": "#005A8D",
    "ensemble": "#B35A00",
    "mcdropout": "#006B5B",
}
MODEL_LABELS = {
    "EDL": "EDL",
    "ensemble": "ensemble",
    "mcdropout": "mcdropout",
}
MODEL_MARKERS = {
    "EDL": "o",
    "ensemble": "s",
    "mcdropout": "^",
}
BASE_GROUP_NAMES = ["aleatoric", "epistemic", "combined"]
IOU_SUMMARY_OUTPUT_NAME = "retention_curve_compare_IoU_by_Uncertainty_Type.png"
IOU_SUMMARY_TITLE = "Retention Curve Comparison by Uncertainty Type"
IOU_SUMMARY_PANELS = [
    {
        "title": "TOTAL",
        "series": [
            {"model": "EDL", "variant": "dst", "label": "EDL DST-u", "marker": "D"},
            {"model": "EDL", "variant": "a+e", "label": "EDL"},
            {"model": "ensemble", "variant": "a+e", "label": "ensemble"},
            {"model": "mcdropout", "variant": "a+e", "label": "mcdropout"},
        ],
    },
    {
        "title": "Aleatoric",
        "series": [
            {"model": "EDL", "variant": "aleatoric", "label": "EDL"},
            {"model": "ensemble", "variant": "aleatoric", "label": "ensemble"},
            {"model": "mcdropout", "variant": "aleatoric", "label": "mcdropout"},
        ],
    },
    {
        "title": "Epistemic",
        "series": [
            {"model": "EDL", "variant": "epistemic", "label": "EDL"},
            {"model": "ensemble", "variant": "epistemic", "label": "ensemble"},
            {"model": "mcdropout", "variant": "epistemic", "label": "mcdropout"},
        ],
    },
]
UNCERTAINTY_TYPE_COLORS = {
    "dst": "#005A8D",
    "a+e": "#7A5195",
    "aleatoric": "#B35A00",
    "epistemic": "#006B5B",
}
UNCERTAINTY_TYPE_MARKERS = {
    "dst": "D",
    "a+e": "s",
    "aleatoric": "o",
    "epistemic": "^",
}
METHOD_IOU_OUTPUT_NAMES = {
    "EDL": "retention_curve_compare_EDL_IoU_by_Uncertainty_Type.png",
    "ensemble": "retention_curve_compare_ensemble_IoU_by_Uncertainty_Type.png",
    "mcdropout": "retention_curve_compare_mcdropout_IoU_by_Uncertainty_Type.png",
}
METHOD_IOU_CONFIGS = {
    "EDL": {
        "title": "EDL",
        "series": [
            {"model": "EDL", "variant": "dst", "label": "DST-u"},
            {"model": "EDL", "variant": "a+e", "label": "Total variance"},
            {"model": "EDL", "variant": "aleatoric", "label": "Aleatoric"},
            {"model": "EDL", "variant": "epistemic", "label": "Epistemic"},
        ],
    },
    "ensemble": {
        "title": "ensemble",
        "series": [
            {"model": "ensemble", "variant": "a+e", "label": "Total variance"},
            {"model": "ensemble", "variant": "aleatoric", "label": "Aleatoric"},
            {"model": "ensemble", "variant": "epistemic", "label": "Epistemic"},
        ],
    },
    "mcdropout": {
        "title": "mcdropout",
        "series": [
            {"model": "mcdropout", "variant": "a+e", "label": "Total variance"},
            {"model": "mcdropout", "variant": "aleatoric", "label": "Aleatoric"},
            {"model": "mcdropout", "variant": "epistemic", "label": "Epistemic"},
        ],
    },
}
GROUP_CONFIGS = {
    "aleatoric": {
        "title": "Retention Curve Comparison by Aleatoric Uncertainty",
        "output_name": "retention_curve_compare_Aleatoric_Uncertainty.png",
        "figsize": (15.5, 4.8),
        "series": [
            {"model": "EDL", "variant": "aleatoric", "label": "EDL"},
            {"model": "ensemble", "variant": "aleatoric", "label": "ensemble"},
            {"model": "mcdropout", "variant": "aleatoric", "label": "mcdropout"},
        ],
    },
    "epistemic": {
        "title": "Retention Curve Comparison by Epistemic Uncertainty",
        "output_name": "retention_curve_compare_Epistemic_Uncertainty.png",
        "figsize": (15.5, 4.8),
        "series": [
            {"model": "EDL", "variant": "epistemic", "label": "EDL"},
            {"model": "ensemble", "variant": "epistemic", "label": "ensemble"},
            {"model": "mcdropout", "variant": "epistemic", "label": "mcdropout"},
        ],
    },
    "combined": {
        "title": "Retention Curve Comparison: DST-u vs Total variance",
        "output_name": "retention_curve_compare_Total_Variance_vs_DST_Uncertainty.png",
        "figsize": (15.8, 5.0),
        "series": [
            {"model": "EDL", "variant": "dst", "label": "EDL DST-u"},
            {
                "model": "EDL",
                "variant": "a+e",
                "label": "EDL Total variance",
                "marker": "s",
            },
            {
                "model": "ensemble",
                "variant": "a+e",
                "label": "ensemble Total variance",
            },
            {
                "model": "mcdropout",
                "variant": "a+e",
                "label": "mcdropout Total variance",
            },
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot merged retention curve comparison figures from existing CSV files."
    )
    parser.add_argument(
        "--analysis-root",
        default=str(DEFAULT_ANALYSIS_ROOT),
        help=f"Root directory containing analysis CSV files. Default: {DEFAULT_ANALYSIS_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for merged retention plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--group",
        choices=["aleatoric", "epistemic", "combined", "iou-summary", "method-iou", "all"],
        default="all",
        help="Which merged plot group to generate. Default: all",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI. Default: 300")
    return parser.parse_args()


def build_csv_paths(analysis_root):
    analysis_root = Path(analysis_root)
    csv_paths = {}

    for model_name in MODEL_LABELS:
        model_dir = analysis_root / model_name
        csv_paths[model_name] = {
            "aleatoric": model_dir / "retention_Water_Aleatoric.csv",
            "epistemic": model_dir / "retention_Water_Epistemic.csv",
            "a+e": model_dir / "retention_Water_Aleatoric_Plus_Epistemic.csv",
        }

    csv_paths["EDL"]["dst"] = analysis_root / "EDL" / "retention_Water_DST_Uncertainty.csv"
    return csv_paths


def load_retention_csv(csv_path):
    return pd.read_csv(csv_path)


def validate_retention_df(df, path):
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {path}: {missing_str}")

    if df.empty:
        raise ValueError(f"Retention CSV is empty: {path}")


def get_series_style(model_name, use_dst_marker=False):
    style = {
        "color": MODEL_COLORS[model_name],
        "linewidth": 1.9,
        "linestyle": "-",
        "marker": MODEL_MARKERS[model_name],
        "markevery": 4,
        "markersize": 5.8,
        "markeredgewidth": 1.2,
    }

    if MODEL_MARKERS[model_name] in {"o", "s", "^"}:
        style["markerfacecolor"] = "white"

    if use_dst_marker:
        style.update(
            {
                "markevery": 5,
                "markersize": 6.2,
                "markerfacecolor": "white",
                "markeredgewidth": 1.3,
            }
        )

    return style


def resolve_series_marker(series_config):
    return series_config.get("marker", MODEL_MARKERS[series_config["model"]])


def build_series_legend_handles(group_name):
    group_config = GROUP_CONFIGS[group_name]
    handles = []
    for series_config in group_config["series"]:
        handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[series_config["model"]],
                linewidth=2.2,
                linestyle="-",
                marker=resolve_series_marker(series_config),
                markersize=6.2,
                markerfacecolor="white",
                markeredgewidth=1.3,
                label=series_config["label"],
            )
        )
    return handles


def build_iou_summary_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["EDL"],
            linewidth=2.2,
            linestyle="-",
            marker=MODEL_MARKERS["EDL"],
            markersize=6.2,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label="EDL",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["ensemble"],
            linewidth=2.2,
            linestyle="-",
            marker=MODEL_MARKERS["ensemble"],
            markersize=6.2,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label="ensemble",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["mcdropout"],
            linewidth=2.2,
            linestyle="-",
            marker=MODEL_MARKERS["mcdropout"],
            markersize=6.2,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label="mcdropout",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["EDL"],
            linewidth=2.2,
            linestyle="-",
            marker="D",
            markersize=6.2,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label="EDL DST-u",
        ),
    ]


def get_uncertainty_type_style(variant):
    return {
        "color": UNCERTAINTY_TYPE_COLORS[variant],
        "linewidth": 1.9,
        "linestyle": "-",
        "marker": UNCERTAINTY_TYPE_MARKERS[variant],
        "markevery": 4,
        "markersize": 5.8,
        "markerfacecolor": "white",
        "markeredgewidth": 1.2,
    }


def build_uncertainty_type_legend_handles(series_configs):
    handles = []
    for series_config in series_configs:
        variant = series_config["variant"]
        handles.append(
            Line2D(
                [0],
                [0],
                color=UNCERTAINTY_TYPE_COLORS[variant],
                linewidth=2.2,
                linestyle="-",
                marker=UNCERTAINTY_TYPE_MARKERS[variant],
                markersize=6.2,
                markerfacecolor="white",
                markeredgewidth=1.3,
                label=series_config["label"],
            )
        )
    return handles


def save_figure(fig, output_path, dpi):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def prepare_series_dataframe(df, path):
    validate_retention_df(df, path)
    prepared_df = df.loc[:, ["retention_rate", "iou", "precision", "recall"]].copy()
    prepared_df = prepared_df.sort_values("retention_rate").reset_index(drop=True)
    return prepared_df


def load_group_series_data(group_config, csv_paths):
    series_data = []

    for series_config in group_config["series"]:
        model_name = series_config["model"]
        variant = series_config["variant"]
        csv_path = csv_paths[model_name][variant]
        df = prepare_series_dataframe(load_retention_csv(csv_path), csv_path)
        series_data.append(
            {
                "config": series_config,
                "df": df,
                "x_values": df["retention_rate"] * 100,
            }
        )

    return series_data


def plot_metric_panel(ax, metric_name, series_data):
    for entry in series_data:
        series_config = entry["config"]
        model_name = series_config["model"]
        variant = series_config["variant"]
        style = get_series_style(
            model_name=model_name,
            use_dst_marker=(variant == "dst"),
        )
        style["marker"] = resolve_series_marker(series_config)
        ax.plot(entry["x_values"], entry["df"][metric_name], **style)

    ax.set_title(METRIC_LABELS[metric_name])
    ax.set_xlabel("Retention Rate (%)")
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
    ax.invert_xaxis()


def plot_compare_group(group_name, csv_paths, output_path, dpi):
    group_config = GROUP_CONFIGS[group_name]
    fig, axes = plt.subplots(1, 3, figsize=group_config["figsize"], sharex=True, sharey=False)
    series_data = load_group_series_data(group_config, csv_paths)

    for ax, metric_name in zip(axes, METRIC_ORDER):
        plot_metric_panel(ax, metric_name, series_data)

    axes[0].set_ylabel("Score")
    axes[-1].legend(
        handles=build_series_legend_handles(group_name),
        title="Series",
        loc="lower right",
        frameon=True,
    )

    fig.suptitle(group_config["title"])
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.84, wspace=0.20)

    save_figure(fig, output_path, dpi)


def build_metric_output_path(output_path, metric_name):
    metric_label = METRIC_LABELS[metric_name].replace(" ", "_")
    return output_path.with_name(f"{output_path.stem}_{metric_label}{output_path.suffix}")


def plot_single_metric_group(group_name, metric_name, csv_paths, output_path, dpi):
    group_config = GROUP_CONFIGS[group_name]
    series_data = load_group_series_data(group_config, csv_paths)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.8))

    plot_metric_panel(ax, metric_name, series_data)
    ax.set_ylabel("Score")
    ax.legend(
        handles=build_series_legend_handles(group_name),
        title="Series",
        loc="lower right",
        frameon=True,
    )
    fig.suptitle(f"{group_config['title']} - {METRIC_LABELS[metric_name]}")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.82)

    save_figure(fig, build_metric_output_path(output_path, metric_name), dpi)


def plot_iou_summary(csv_paths, output_path, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=False)

    for ax, panel_config in zip(axes, IOU_SUMMARY_PANELS):
        series_data = load_group_series_data(panel_config, csv_paths)
        plot_metric_panel(ax, "iou", series_data)
        ax.set_title(panel_config["title"])
        ax.set_ylabel("IoU")

    axes[-1].legend(
        handles=build_iou_summary_legend_handles(),
        title="Series",
        loc="lower right",
        frameon=True,
    )
    fig.suptitle(IOU_SUMMARY_TITLE)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.84, wspace=0.20)
    save_figure(fig, output_path, dpi)


def plot_method_iou_figure(method_name, csv_paths, output_dir, dpi):
    method_config = METHOD_IOU_CONFIGS[method_name]
    series_data = load_group_series_data(method_config, csv_paths)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

    for entry in series_data:
        series_config = entry["config"]
        style = get_uncertainty_type_style(series_config["variant"])
        ax.plot(entry["x_values"], entry["df"]["iou"], **style)

    ax.set_title(method_config["title"])
    ax.set_xlabel("Retention Rate (%)")
    ax.set_ylabel("IoU")
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
    ax.invert_xaxis()
    ax.legend(
        handles=build_uncertainty_type_legend_handles(method_config["series"]),
        title="Uncertainty type",
        loc="lower right",
        frameon=True,
    )
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.88)

    output_path = output_dir / METHOD_IOU_OUTPUT_NAMES[method_name]
    save_figure(fig, output_path, dpi)
    return output_path


def plot_method_iou_figures(csv_paths, output_dir, dpi):
    output_paths = []
    for method_name in METHOD_IOU_CONFIGS:
        output_paths.append(plot_method_iou_figure(method_name, csv_paths, output_dir, dpi))
    return output_paths


def main():
    args = parse_args()
    csv_paths = build_csv_paths(args.analysis_root)
    output_dir = Path(args.output_dir)

    selected_groups = BASE_GROUP_NAMES if args.group == "all" else []
    if args.group in BASE_GROUP_NAMES:
        selected_groups = [args.group]

    for group_name in selected_groups:
        output_path = output_dir / GROUP_CONFIGS[group_name]["output_name"]
        plot_compare_group(group_name, csv_paths, output_path, args.dpi)
        print(f"Saved merged retention plot to: {output_path}")
        for metric_name in METRIC_ORDER:
            metric_output_path = build_metric_output_path(output_path, metric_name)
            plot_single_metric_group(group_name, metric_name, csv_paths, output_path, args.dpi)
            print(f"Saved single-metric retention plot to: {metric_output_path}")

    if args.group in {"iou-summary", "all"}:
        output_path = output_dir / IOU_SUMMARY_OUTPUT_NAME
        plot_iou_summary(csv_paths, output_path, args.dpi)
        print(f"Saved IoU summary retention plot to: {output_path}")

    if args.group in {"method-iou", "all"}:
        for output_path in plot_method_iou_figures(csv_paths, output_dir, args.dpi):
            print(f"Saved method IoU retention plot to: {output_path}")


if __name__ == "__main__":
    main()
