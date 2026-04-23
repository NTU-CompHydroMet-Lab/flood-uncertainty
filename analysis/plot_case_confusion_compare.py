import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

import case_compare_shared as shared
from path_defaults import DEFAULT_DATA_ROOT, DEFAULT_FIGURE_ROOT, DEFAULT_PRED_ROOT


DEFAULT_EVENT_ID = "EMSR466_AOI01_DEL_PRODUCT"

DEFAULT_OUTPUT_DIR = DEFAULT_FIGURE_ROOT

CONFUSION_TITLE_FONTSIZE = shared.CONFUSION_TITLE_FONTSIZE
CONFUSION_TICK_FONTSIZE = shared.CONFUSION_TICK_FONTSIZE
CONFUSION_LEGEND_FONTSIZE = shared.CONFUSION_LEGEND_FONTSIZE
CONFUSION_LEGEND_TITLE_FONTSIZE = shared.CONFUSION_LEGEND_TITLE_FONTSIZE
CONFUSION_COORD_PAD = shared.CONFUSION_COORD_PAD
CONFUSION_COORD_DECIMALS = shared.COORD_DECIMALS
N_COORD_TICKS = shared.CONFUSION_N_COORD_TICKS
CM_COLORS = shared.CONFUSION_CM_COLORS
CM_LABELS = shared.CONFUSION_CM_LABELS
GT_COLORS = shared.CONFUSION_GT_COLORS
GT_LABELS = shared.CONFUSION_GT_LABELS
MODEL_TITLES = shared.CONFUSION_MODEL_TITLES

MODEL_CM_SUFFIXES = {
    "v2": "_prediction_v2",
    "EDL": "_output_EDL",
    "ensemble": "_output_ensemble",
    "mcdropout": "_mcdropout_output_ensemble",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a 2x3 case-study compare figure from S2, GT, and confusion maps."
    )
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID, help="Event ID, e.g. EMSR466_AOI01_DEL_PRODUCT")
    parser.add_argument("--subset", default="test", help="Dataset subset. Default: test")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help=f"WorldFloods data root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--pred-root",
        default=str(DEFAULT_PRED_ROOT),
        help=f"Prediction root. Default: {DEFAULT_PRED_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename. Default: {event_id}_confusion_compare.png",
    )
    parser.add_argument("--dpi", default=300, type=int, help="Figure DPI. Default: 300")
    return parser.parse_args()


def build_paths(event_id, subset, data_root, pred_root):
    data_root = Path(data_root)
    pred_root = Path(pred_root)

    s2_path = data_root / subset / "S2" / f"{event_id}.tif"
    gt_path = data_root / subset / "gt" / f"{event_id}.tif"
    cm_paths = {
        model_name: pred_root / subset / model_name / f"cm_{event_id}{suffix}.tif"
        for model_name, suffix in MODEL_CM_SUFFIXES.items()
    }

    return {
        "s2": s2_path,
        "gt": gt_path,
        "cm": cm_paths,
    }


def validate_required_paths(path_dict):
    missing_paths = []

    for key in ("s2", "gt"):
        if not path_dict[key].exists():
            missing_paths.append(path_dict[key])

    for cm_path in path_dict["cm"].values():
        if not cm_path.exists():
            missing_paths.append(cm_path)

    if missing_paths:
        missing_str = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required input file(s):\n{missing_str}")


def load_s2_rgb(s2_path):
    with rasterio.open(s2_path) as src:
        rgb = src.read([4, 3, 2]).astype(np.float32)
    return np.transpose(rgb, (1, 2, 0))


def load_spatial_meta(s2_path):
    with rasterio.open(s2_path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS in raster: {s2_path}")

        return {
            "transform": src.transform,
            "crs": src.crs,
            "height": src.height,
            "width": src.width,
        }


def load_gt_map(gt_path):
    with rasterio.open(gt_path) as src:
        gt = src.read(2).astype(np.int16)
    return gt


def load_cm_map(cm_path):
    with rasterio.open(cm_path) as src:
        cm_map = src.read(1).astype(np.int16)
    return cm_map


def build_gt_display(gt_raw):
    gt_display = np.zeros_like(gt_raw, dtype=np.int16)
    gt_display[gt_raw == 0] = 0
    gt_display[gt_raw == 1] = 1
    gt_display[gt_raw == 2] = 2
    return gt_display


def get_gt_cmap_and_labels():
    return shared.get_confusion_gt_cmap_and_labels()


def get_cm_cmap_and_labels():
    return shared.get_confusion_cm_cmap_and_labels()


def make_tick_positions(size, n_ticks=N_COORD_TICKS):
    return shared.make_tick_positions(size, n_ticks)


def format_decimal_degree(value):
    return shared.format_decimal_degree(value)


def compute_lon_ticks(spatial_meta, width, row_ref):
    return shared._compute_lon_ticks(
        spatial_meta=spatial_meta,
        width=width,
        row_ref=row_ref,
        n_ticks=N_COORD_TICKS,
    )


def compute_lat_ticks(spatial_meta, height, col_ref):
    return shared._compute_lat_ticks(
        spatial_meta=spatial_meta,
        height=height,
        col_ref=col_ref,
        n_ticks=N_COORD_TICKS,
    )


def apply_lonlat_ticks(ax, spatial_meta, width, height, show_x=False, show_y=False):
    return shared.apply_decimal_ticks_confusion(
        ax=ax,
        spatial_meta=spatial_meta,
        width=width,
        height=height,
        show_x=show_x,
        show_y=show_y,
    )


def add_discrete_legend(fig, labels, colors, title, anchor_x):
    return shared.add_discrete_legend(fig, labels, colors, title, anchor_x)


def validate_shapes(rgb, gt_display, cm_maps):
    expected_shape = gt_display.shape
    rgb_shape = rgb.shape[:2]

    if rgb_shape != expected_shape:
        raise ValueError(f"S2 shape {rgb_shape} does not match GT shape {expected_shape}")

    for model_name, cm_map in cm_maps.items():
        if cm_map.shape != expected_shape:
            raise ValueError(
                f"Confusion map shape mismatch for {model_name}: {cm_map.shape} != {expected_shape}"
            )


def plot_compare_figure(event_id, rgb, gt_display, cm_maps, spatial_meta):
    return shared.draw_confusion_compare_figure(
        rgb=rgb,
        gt_display=gt_display,
        cm_maps=cm_maps,
        spatial_meta=spatial_meta,
    )


def save_figure(fig, output_path, dpi):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_name = args.output_name or f"{args.event_id}_confusion_compare.png"
    output_path = Path(args.output_dir) / output_name

    path_dict = build_paths(
        event_id=args.event_id,
        subset=args.subset,
        data_root=args.data_root,
        pred_root=args.pred_root,
    )
    validate_required_paths(path_dict)

    rgb = load_s2_rgb(path_dict["s2"])
    spatial_meta = load_spatial_meta(path_dict["s2"])
    gt_display = build_gt_display(load_gt_map(path_dict["gt"]))
    cm_maps = {
        model_name: load_cm_map(cm_path)
        for model_name, cm_path in path_dict["cm"].items()
    }

    validate_shapes(rgb, gt_display, cm_maps)
    figure = plot_compare_figure(args.event_id, rgb, gt_display, cm_maps, spatial_meta)
    save_figure(figure, output_path, args.dpi)
    print(f"Saved compare figure to: {output_path}")


if __name__ == "__main__":
    main()
