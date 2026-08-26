import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.warp import transform as transform_coords


PLOT_FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
]

COORD_DECIMALS = 2

CONFUSION_N_COORD_TICKS = 2
CONFUSION_TITLE_FONTSIZE = 18
CONFUSION_TICK_FONTSIZE = 14
CONFUSION_LEGEND_FONTSIZE = 14
CONFUSION_LEGEND_TITLE_FONTSIZE = 16
CONFUSION_COORD_PAD = 7
CONFUSION_TITLE_PAD = 10
CONFUSION_METRIC_FONTSIZE = 12
CONFUSION_FIGURE_SIZE = (19.8, 12.4)
CONFUSION_SUBPLOTS = {
    "left": 0.04,
    "right": 0.995,
    "top": 0.965,
    "bottom": 0.15,
    "metrics_bottom": 0.19,
    "wspace": 0.03,
    "hspace": 0.13,
}

CONFUSION_MODEL_TITLES = {
    "v2": "UNet++",
    "EDL": "UNet++ EDL",
    "ensemble": "UNet++ ensemble",
    "mcdropout": "UNet++ mcdropout",
}

CONFUSION_GT_COLORS = [
    "#808080",
    "#694F2A",
    "#00A6D6",
]
CONFUSION_GT_LABELS = ["Invalid", "Land", "Flood"]

CONFUSION_CM_COLORS = [
    "#808080",
    "#3F3830",
    "#E1B423",
    "#D34B57",
    "#3CA1AB",
]
CONFUSION_CM_LABELS = ["Invalid", "TN", "FP", "FN", "TP"]

UNCERTAINTY_TITLE_FONTSIZE = 18
UNCERTAINTY_TICK_FONTSIZE = 14
UNCERTAINTY_COLORBAR_FONTSIZE = 14
UNCERTAINTY_INFO_TITLE_FONTSIZE = 18
UNCERTAINTY_INFO_LABEL_FONTSIZE = 18
UNCERTAINTY_INFO_BODY_FONTSIZE = 18
UNCERTAINTY_COORD_PAD = 7
UNCERTAINTY_COLORBAR_FORMAT = "%.2f"
UNCERTAINTY_ROW_COLORBAR_SIZE = "3.5%"
UNCERTAINTY_ROW_COLORBAR_PAD = 0.04
UNCERTAINTY_FIGURE_SIZE = (14, 14.6)
UNCERTAINTY_SUBPLOTS = {
    "left": 0.035,
    "right": 0.985,
    "top": 0.975,
    "bottom": 0.045,
    "wspace": 0.3,
    "hspace": 0.18,
}
UNCERTAINTY_CMAP = "Blues"
UNCERTAINTY_INVALID_COLOR = "#d9d9d9"
UNCERTAINTY_FP_COLOR = "#E1B423"
UNCERTAINTY_FN_COLOR = "#D34B57"
UNCERTAINTY_FP_FILL_ALPHA = 0.6
UNCERTAINTY_FN_FILL_ALPHA = 0.6
UNCERTAINTY_ROW_LABELS = [
    "DST-u",
    "Aleatoric + Epistemic",
    "Aleatoric",
    "Epistemic",
]


def apply_plot_rcparams():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = PLOT_FONT_SERIF


def make_tick_positions(size, n_ticks):
    if size <= 1:
        return np.array([0], dtype=int)

    positions = np.linspace(0, size - 1, num=min(n_ticks, size))
    return np.unique(positions.round().astype(int))


def format_decimal_degree(value):
    return f"{value:.{COORD_DECIMALS}f}"


def _compute_lon_ticks(spatial_meta, width, row_ref, stride=1, n_ticks=CONFUSION_N_COORD_TICKS):
    tick_positions = make_tick_positions(width, n_ticks)
    cols = np.clip(tick_positions * stride, 0, spatial_meta["width"] - 1)
    row = min(row_ref * stride, spatial_meta["height"] - 1)
    rows = [row] * len(cols)

    xs, ys = rasterio.transform.xy(spatial_meta["transform"], rows, cols.tolist(), offset="center")
    longitudes, _ = transform_coords(spatial_meta["crs"], "EPSG:4326", xs, ys)
    labels = [format_decimal_degree(value) for value in longitudes]
    return tick_positions, labels


def _compute_lat_ticks(spatial_meta, height, col_ref, stride=1, n_ticks=CONFUSION_N_COORD_TICKS):
    tick_positions = make_tick_positions(height, n_ticks)
    rows = np.clip(tick_positions * stride, 0, spatial_meta["height"] - 1)
    col = min(col_ref * stride, spatial_meta["width"] - 1)
    cols = [col] * len(rows)

    xs, ys = rasterio.transform.xy(spatial_meta["transform"], rows.tolist(), cols, offset="center")
    _, latitudes = transform_coords(spatial_meta["crs"], "EPSG:4326", xs, ys)
    labels = [format_decimal_degree(value) for value in latitudes]
    return tick_positions, labels


def apply_decimal_ticks_confusion(ax, spatial_meta, width, height, show_x=False, show_y=False):
    if show_x:
        xticks, xlabels = _compute_lon_ticks(
            spatial_meta=spatial_meta,
            width=width,
            row_ref=height // 2,
            n_ticks=CONFUSION_N_COORD_TICKS,
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=CONFUSION_TICK_FONTSIZE)
        ax.tick_params(axis="x", bottom=True, labelbottom=True, length=2, pad=CONFUSION_COORD_PAD)
    else:
        ax.set_xticks([])

    if show_y:
        yticks, ylabels = _compute_lat_ticks(
            spatial_meta=spatial_meta,
            height=height,
            col_ref=width // 2,
            n_ticks=CONFUSION_N_COORD_TICKS,
        )
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=CONFUSION_TICK_FONTSIZE)
        ax.tick_params(axis="y", left=True, labelleft=True, length=2, pad=CONFUSION_COORD_PAD)
    else:
        ax.set_yticks([])


def apply_decimal_ticks_uncertainty(ax, spatial_meta, stride, width_ds, height_ds, show_x=False, show_y=False):
    if show_x:
        xticks, xlabels = _compute_lon_ticks(
            spatial_meta=spatial_meta,
            width=width_ds,
            row_ref=height_ds // 2,
            stride=stride,
            n_ticks=CONFUSION_N_COORD_TICKS,
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=UNCERTAINTY_TICK_FONTSIZE)
        ax.tick_params(axis="x", bottom=True, labelbottom=True, length=2, pad=UNCERTAINTY_COORD_PAD)
    else:
        ax.set_xticks([])

    if show_y:
        yticks, ylabels = _compute_lat_ticks(
            spatial_meta=spatial_meta,
            height=height_ds,
            col_ref=width_ds // 2,
            stride=stride,
            n_ticks=CONFUSION_N_COORD_TICKS,
        )
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=UNCERTAINTY_TICK_FONTSIZE)
        ax.tick_params(axis="y", left=True, labelleft=True, length=2, pad=UNCERTAINTY_COORD_PAD)
    else:
        ax.set_yticks([])


def get_confusion_gt_cmap_and_labels():
    return ListedColormap(CONFUSION_GT_COLORS), CONFUSION_GT_LABELS, CONFUSION_GT_COLORS


def get_confusion_cm_cmap_and_labels():
    return ListedColormap(CONFUSION_CM_COLORS), CONFUSION_CM_LABELS, CONFUSION_CM_COLORS


def add_discrete_legend(fig, labels, colors, title, anchor_x):
    handles = [Patch(facecolor=color, edgecolor="black", label=label) for label, color in zip(labels, colors)]
    return fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(anchor_x, 0.03),
        ncol=len(labels),
        frameon=True,
        title=title,
        fontsize=CONFUSION_LEGEND_FONTSIZE,
        title_fontsize=CONFUSION_LEGEND_TITLE_FONTSIZE,
    )


def compute_confusion_metrics(cm_map):
    tp = int(np.sum(cm_map == 4))
    fp = int(np.sum(cm_map == 2))
    fn = int(np.sum(cm_map == 3))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def format_confusion_metrics(metrics):
    return (
        f"IoU {metrics['iou']:.3f} | "
        f"Prec {metrics['precision']:.3f} | "
        f"Rec {metrics['recall']:.3f}"
    )


def draw_confusion_compare_figure(
    rgb,
    gt_display,
    cm_maps,
    spatial_meta,
    figure_size=CONFUSION_FIGURE_SIZE,
    show_metrics=False,
):
    apply_plot_rcparams()
    gt_cmap, gt_labels, gt_colors = get_confusion_gt_cmap_and_labels()
    cm_cmap, cm_labels, cm_colors = get_confusion_cm_cmap_and_labels()

    fig, axes = plt.subplots(2, 3, figsize=figure_size, constrained_layout=False)
    axes = axes.flatten()
    height, width = gt_display.shape

    panels = [
        ("Sentinel-2 RGB", "rgb"),
        ("Ground Truth", "gt"),
        (CONFUSION_MODEL_TITLES["v2"], "v2"),
        (CONFUSION_MODEL_TITLES["EDL"], "EDL"),
        (CONFUSION_MODEL_TITLES["ensemble"], "ensemble"),
        (CONFUSION_MODEL_TITLES["mcdropout"], "mcdropout"),
    ]

    for index, (ax, (title, panel_key)) in enumerate(zip(axes, panels)):
        row_idx = index // 3
        col_idx = index % 3
        ax.set_title(title, fontsize=CONFUSION_TITLE_FONTSIZE, pad=CONFUSION_TITLE_PAD)
        apply_decimal_ticks_confusion(
            ax=ax,
            spatial_meta=spatial_meta,
            width=width,
            height=height,
            show_x=row_idx == 1,
            show_y=col_idx == 0,
        )

        if panel_key == "rgb":
            ax.imshow((rgb / 3500.0).clip(0, 1), interpolation="nearest")
            continue

        if panel_key == "gt":
            ax.imshow(gt_display, cmap=gt_cmap, vmin=0, vmax=len(gt_labels) - 1, interpolation="nearest")
            continue

        ax.imshow(cm_maps[panel_key], cmap=cm_cmap, vmin=0, vmax=len(cm_labels) - 1, interpolation="nearest")
        if show_metrics:
            ax.text(
                0.5,
                -0.095,
                format_confusion_metrics(compute_confusion_metrics(cm_maps[panel_key])),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=CONFUSION_METRIC_FONTSIZE,
                clip_on=False,
            )

    bottom_margin = CONFUSION_SUBPLOTS["metrics_bottom"] if show_metrics else CONFUSION_SUBPLOTS["bottom"]
    fig.subplots_adjust(
        left=CONFUSION_SUBPLOTS["left"],
        right=CONFUSION_SUBPLOTS["right"],
        top=CONFUSION_SUBPLOTS["top"],
        bottom=bottom_margin,
        wspace=CONFUSION_SUBPLOTS["wspace"],
        hspace=CONFUSION_SUBPLOTS["hspace"],
    )
    add_discrete_legend(fig, gt_labels, gt_colors, "Ground Truth", 0.27)
    add_discrete_legend(fig, cm_labels, cm_colors, "Confusion Map", 0.73)
    return fig


def build_fill_overlay(mask, color_hex, alpha):
    if alpha <= 0:
        return np.zeros(mask.shape + (4,), dtype=np.float32)

    color = np.array(plt.matplotlib.colors.to_rgba(color_hex, alpha=alpha))
    overlay = np.zeros(mask.shape + (4,), dtype=np.float32)
    overlay[mask] = color
    return overlay


def build_outline_overlay(mask, color_hex):
    color = np.array(plt.matplotlib.colors.to_rgba(color_hex))
    overlay = np.zeros(mask.shape + (4,), dtype=np.float32)
    overlay[mask] = color
    return overlay


def compute_limits_from_valid_values(valid_values):
    if valid_values.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanpercentile(valid_values, 2))
    vmax = float(np.nanpercentile(valid_values, 98))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return vmin, vmax


def compute_uncertainty_row_limits(event_bundle, row_panels):
    valid_values = []

    for panel_config in row_panels:
        if panel_config["kind"] != "map":
            continue

        masked_map = event_bundle["models"][panel_config["model"]]["precomputed_maps"][panel_config["source"]][
            "masked_map"
        ]
        compressed = masked_map.compressed()
        if compressed.size:
            valid_values.append(compressed)

    if not valid_values:
        return 0.0, 1.0

    return compute_limits_from_valid_values(np.concatenate(valid_values))


def build_uncertainty_layout():
    return [
        [
            {"kind": "rgb"},
            {"kind": "map", "model": "EDL", "source": "dstu", "title": "EDL DST-u"},
            {"kind": "info"},
        ],
        [
            {
                "kind": "map",
                "model": "EDL",
                "source": "epistemic_plus_aleatoric",
                "title": "EDL",
            },
            {
                "kind": "map",
                "model": "ensemble",
                "source": "epistemic_plus_aleatoric",
                "title": "Ensemble",
            },
            {
                "kind": "map",
                "model": "mcdropout",
                "source": "epistemic_plus_aleatoric",
                "title": "MC Dropout",
            },
        ],
        [
            {"kind": "map", "model": "EDL", "source": "aleatoric", "title": "EDL"},
            {"kind": "map", "model": "ensemble", "source": "aleatoric", "title": "Ensemble"},
            {"kind": "map", "model": "mcdropout", "source": "aleatoric", "title": "MC Dropout"},
        ],
        [
            {"kind": "map", "model": "EDL", "source": "epistemic", "title": "EDL"},
            {"kind": "map", "model": "ensemble", "source": "epistemic", "title": "Ensemble"},
            {"kind": "map", "model": "mcdropout", "source": "epistemic", "title": "MC Dropout"},
        ],
    ]


def add_uncertainty_info_panel(ax):
    ax.axis("off")
    title_size = UNCERTAINTY_INFO_TITLE_FONTSIZE
    label_size = UNCERTAINTY_INFO_LABEL_FONTSIZE
    box_x = 0.15
    box_width = 0.11
    box_height = 0.07
    fp_y = 0.71
    fn_y = 0.57
    invalid_y = 0.43
    text_x = 0.3

    ax.text(
        0.10,
        0.94,
        "Error Overlay",
        transform=ax.transAxes,
        fontsize=title_size,
        va="top",
        fontweight="bold",
    )

    ax.add_patch(
        Rectangle(
            (box_x, fp_y - box_height / 2),
            box_width,
            box_height,
            transform=ax.transAxes,
            facecolor=plt.matplotlib.colors.to_rgba(UNCERTAINTY_FP_COLOR, alpha=UNCERTAINTY_FP_FILL_ALPHA),
            edgecolor=UNCERTAINTY_FP_COLOR,
            linewidth=1.0,
            clip_on=False,
        )
    )
    ax.text(text_x, fp_y, "False positive", transform=ax.transAxes, fontsize=label_size, va="center")

    ax.add_patch(
        Rectangle(
            (box_x, fn_y - box_height / 2),
            box_width,
            box_height,
            transform=ax.transAxes,
            facecolor=plt.matplotlib.colors.to_rgba(UNCERTAINTY_FN_COLOR, alpha=UNCERTAINTY_FN_FILL_ALPHA),
            edgecolor=UNCERTAINTY_FN_COLOR,
            linewidth=1.0,
            clip_on=False,
        )
    )
    ax.text(text_x, fn_y, "False negative", transform=ax.transAxes, fontsize=label_size, va="center")

    ax.add_patch(
        Rectangle(
            (box_x, invalid_y - box_height / 2),
            box_width,
            box_height,
            transform=ax.transAxes,
            facecolor=UNCERTAINTY_INVALID_COLOR,
            edgecolor="black",
            linewidth=1.0,
            clip_on=False,
        )
    )
    ax.text(text_x, invalid_y, "Gray: invalid / no data", transform=ax.transAxes, fontsize=label_size, va="center")
def add_uncertainty_row_colorbar(fig, cax, reference_ax, image, label):
    ref_bbox = reference_ax.get_position()
    cbar_bbox = cax.get_position()
    cax.set_position([cbar_bbox.x0, ref_bbox.y0, cbar_bbox.width, ref_bbox.height])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=UNCERTAINTY_COLORBAR_FONTSIZE)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter(UNCERTAINTY_COLORBAR_FORMAT))
    colorbar.set_label(label, fontsize=UNCERTAINTY_TITLE_FONTSIZE, rotation=270, labelpad=24)


def draw_uncertainty_compare_figure(event_bundle, figure_size=UNCERTAINTY_FIGURE_SIZE):
    apply_plot_rcparams()
    layout = build_uncertainty_layout()
    fig = plt.figure(figsize=figure_size, constrained_layout=False)
    outer_grid = fig.add_gridspec(
        4,
        1,
        left=UNCERTAINTY_SUBPLOTS["left"],
        right=UNCERTAINTY_SUBPLOTS["right"],
        top=UNCERTAINTY_SUBPLOTS["top"],
        bottom=UNCERTAINTY_SUBPLOTS["bottom"],
        hspace=UNCERTAINTY_SUBPLOTS["hspace"],
    )
    cmap = plt.get_cmap(UNCERTAINTY_CMAP).copy()
    cmap.set_bad(color=UNCERTAINTY_INVALID_COLOR)
    spatial_meta = event_bundle["spatial_meta"]
    stride = event_bundle["stride"]
    width_ds = event_bundle["width_ds"]
    height_ds = event_bundle["height_ds"]

    for row_idx, row in enumerate(layout):
        if row_idx == 0:
            row_grid = outer_grid[row_idx, 0].subgridspec(
                1,
                4,
                width_ratios=[1.0, 1.0, 0.06, 1.0],
                wspace=UNCERTAINTY_SUBPLOTS["wspace"],
            )
            row_axes = [
                fig.add_subplot(row_grid[0, 0]),
                fig.add_subplot(row_grid[0, 1]),
                fig.add_subplot(row_grid[0, 3]),
            ]
            cax = fig.add_subplot(row_grid[0, 2])
        else:
            row_grid = outer_grid[row_idx, 0].subgridspec(
                1,
                4,
                width_ratios=[1.0, 1.0, 1.0, 0.06],
                wspace=UNCERTAINTY_SUBPLOTS["wspace"],
            )
            row_axes = [fig.add_subplot(row_grid[0, col_idx]) for col_idx in range(3)]
            cax = fig.add_subplot(row_grid[0, 3])
        row_limits = compute_uncertainty_row_limits(event_bundle, row)
        row_map_axes = []
        row_image = None

        for col_idx, panel_config in enumerate(row):
            ax = row_axes[col_idx]
            show_x = row_idx == len(layout) - 1
            show_y = col_idx == 0

            if panel_config["kind"] == "rgb":
                ax.imshow(event_bundle["rgb_plot"], interpolation="nearest")
                ax.set_title("Sentinel-2 RGB", fontsize=UNCERTAINTY_TITLE_FONTSIZE, pad=10)
                apply_decimal_ticks_uncertainty(
                    ax=ax,
                    spatial_meta=spatial_meta,
                    stride=stride,
                    width_ds=width_ds,
                    height_ds=height_ds,
                    show_x=show_x,
                    show_y=show_y,
                )
                ax.tick_params(axis="x", labelsize=UNCERTAINTY_TICK_FONTSIZE)
                ax.tick_params(axis="y", labelsize=UNCERTAINTY_TICK_FONTSIZE)
                continue

            if panel_config["kind"] == "info":
                add_uncertainty_info_panel(ax)
                continue

            model_entry = event_bundle["models"][panel_config["model"]]
            map_data = model_entry["precomputed_maps"][panel_config["source"]]["masked_map"]
            image = ax.imshow(
                map_data,
                cmap=cmap,
                vmin=row_limits[0],
                vmax=row_limits[1],
                interpolation="nearest",
            )
            ax.imshow(
                build_fill_overlay(model_entry["fp_fill"], UNCERTAINTY_FP_COLOR, UNCERTAINTY_FP_FILL_ALPHA),
                interpolation="nearest",
            )
            ax.imshow(
                build_fill_overlay(model_entry["fn_fill"], UNCERTAINTY_FN_COLOR, UNCERTAINTY_FN_FILL_ALPHA),
                interpolation="nearest",
            )
            ax.imshow(build_outline_overlay(model_entry["fp_outline"], UNCERTAINTY_FP_COLOR), interpolation="nearest")
            ax.imshow(build_outline_overlay(model_entry["fn_outline"], UNCERTAINTY_FN_COLOR), interpolation="nearest")
            ax.set_title(panel_config["title"], fontsize=UNCERTAINTY_TITLE_FONTSIZE, pad=10)
            apply_decimal_ticks_uncertainty(
                ax=ax,
                spatial_meta=spatial_meta,
                stride=stride,
                width_ds=width_ds,
                height_ds=height_ds,
                show_x=show_x,
                show_y=show_y,
            )
            ax.tick_params(axis="x", labelsize=UNCERTAINTY_TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=UNCERTAINTY_TICK_FONTSIZE)
            row_map_axes.append(ax)
            if row_image is None:
                row_image = image

        if row_image is not None and row_map_axes:
            add_uncertainty_row_colorbar(fig, cax, row_map_axes[-1], row_image, UNCERTAINTY_ROW_LABELS[row_idx])
        else:
            cax.axis("off")

    return fig
