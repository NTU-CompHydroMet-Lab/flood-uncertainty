import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm

import case_compare_shared as shared
import plot_case_confusion_compare as confusion_plot
import plot_case_uncertainty_compare as uncertainty_plot


PLOT_DIR = Path(__file__).resolve().parent
EDA_DIR = PLOT_DIR.parent / "EDA"
if str(EDA_DIR) not in sys.path:
    sys.path.append(str(EDA_DIR))

from compute_pavpu import compute_pavpu_for_image, get_thresholds


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "report" / "image_of_md" / "case_tiles"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "result" / "analysis_S2"
DEFAULT_EVENT_IDS = [
    "EMSR466_AOI01_DEL_PRODUCT",
    "EMSR264_08VATOMANDRY_DEL_v2",
]

PAVPU_THRESHOLD_KEYS = {
    "EDL": {
        "dstu": "Water_DST_Uncertainty",
        "aleatoric": "Water_Aleatoric",
        "epistemic": "Water_Epistemic",
        "epistemic_plus_aleatoric": "Water_Aleatoric_Plus_Epistemic",
    },
    "ensemble": {
        "aleatoric": "Water_Aleatoric",
        "epistemic": "Water_Epistemic",
        "epistemic_plus_aleatoric": "Water_Aleatoric_Plus_Epistemic",
    },
    "mcdropout": {
        "aleatoric": "Water_Aleatoric",
        "epistemic": "Water_Epistemic",
        "epistemic_plus_aleatoric": "Water_Aleatoric_Plus_Epistemic",
    },
}

PAVPU_CSV_ITEMS = [
    ("edl_dstu_pavpu", "EDL", "dstu"),
    ("edl_aleatoric_plus_epistemic_pavpu", "EDL", "epistemic_plus_aleatoric"),
    ("ensemble_aleatoric_plus_epistemic_pavpu", "ensemble", "epistemic_plus_aleatoric"),
    ("mcdropout_aleatoric_plus_epistemic_pavpu", "mcdropout", "epistemic_plus_aleatoric"),
    ("edl_aleatoric_pavpu", "EDL", "aleatoric"),
    ("edl_epistemic_pavpu", "EDL", "epistemic"),
    ("ensemble_aleatoric_pavpu", "ensemble", "aleatoric"),
    ("ensemble_epistemic_pavpu", "ensemble", "epistemic"),
    ("mcdropout_aleatoric_pavpu", "mcdropout", "aleatoric"),
    ("mcdropout_epistemic_pavpu", "mcdropout", "epistemic"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export tiled case-study compare figures without overwriting the original full-image outputs."
    )
    parser.add_argument(
        "--event-id",
        dest="event_ids",
        action="append",
        default=None,
        help="Event ID to export. Repeat this flag to process multiple events. Default: current two case-study events.",
    )
    parser.add_argument("--subset", default="test", help="Dataset subset. Default: test")
    parser.add_argument(
        "--data-root",
        default=str(confusion_plot.DEFAULT_DATA_ROOT),
        help=f"WorldFloods data root. Default: {confusion_plot.DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--pred-root",
        default=str(confusion_plot.DEFAULT_PRED_ROOT),
        help=f"Prediction root. Default: {confusion_plot.DEFAULT_PRED_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output root for tiled figures. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--kind",
        choices=["confusion", "uncertainty", "both"],
        default="both",
        help="Which figure family to export. Default: both",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels. Default: 512")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI. Default: 300")
    parser.add_argument("--patch-size", type=int, default=3, help="PAVPU patch size. Default: 3")
    parser.add_argument(
        "--retention-target",
        type=float,
        default=0.9,
        help="Retention target used to load PAVPU thresholds. Default: 0.9",
    )
    parser.add_argument(
        "--show-confusion-metrics",
        action="store_true",
        help="Show IoU / Precision / Recall below confusion model panels.",
    )
    parser.add_argument(
        "--show-pavpu",
        action="store_true",
        default=True,
        help="Compute and export PAVPU values to a CSV file for uncertainty tiles.",
    )
    return parser.parse_args()


def resolve_event_ids(event_ids):
    if event_ids:
        return event_ids
    return list(DEFAULT_EVENT_IDS)


def iter_tile_windows(height, width, tile_size):
    row_pad = max(4, len(str(height)))
    col_pad = max(4, len(str(width)))

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            tile_height = min(tile_size, height - row)
            tile_width = min(tile_size, width - col)
            yield {
                "row": row,
                "col": col,
                "height": tile_height,
                "width": tile_width,
                "row_tag": f"{row:0{row_pad}d}",
                "col_tag": f"{col:0{col_pad}d}",
                "window": Window(col_off=col, row_off=row, width=tile_width, height=tile_height),
            }


def count_tiles(height, width, tile_size):
    rows = (height + tile_size - 1) // tile_size
    cols = (width + tile_size - 1) // tile_size
    return rows * cols


def crop_spatial_meta(spatial_meta, window):
    return {
        "transform": window_transform(window, spatial_meta["transform"]),
        "crs": spatial_meta["crs"],
        "height": int(window.height),
        "width": int(window.width),
    }


def build_output_dir(output_root, tile_size, event_id, kind):
    return Path(output_root) / f"{tile_size}x{tile_size}" / event_id / kind


def build_tile_filename(event_id, tile_info, suffix):
    return (
        f"{event_id}_r{tile_info['row_tag']}_c{tile_info['col_tag']}"
        f"_h{tile_info['height']:04d}_w{tile_info['width']:04d}_{suffix}.png"
    )


def load_confusion_source(event_id, args):
    path_dict = confusion_plot.build_paths(
        event_id=event_id,
        subset=args.subset,
        data_root=args.data_root,
        pred_root=args.pred_root,
    )
    confusion_plot.validate_required_paths(path_dict)

    rgb = confusion_plot.load_s2_rgb(path_dict["s2"])
    spatial_meta = confusion_plot.load_spatial_meta(path_dict["s2"])
    gt_display = confusion_plot.build_gt_display(confusion_plot.load_gt_map(path_dict["gt"]))
    cm_maps = {
        model_name: confusion_plot.load_cm_map(cm_path)
        for model_name, cm_path in path_dict["cm"].items()
    }
    confusion_plot.validate_shapes(rgb, gt_display, cm_maps)

    return {
        "event_id": event_id,
        "rgb": rgb,
        "gt_display": gt_display,
        "cm_maps": cm_maps,
        "spatial_meta": spatial_meta,
    }


def export_confusion_tiles(source, output_root, tile_size, dpi, show_metrics=False):
    event_id = source["event_id"]
    height, width = source["gt_display"].shape
    output_dir = build_output_dir(output_root, tile_size, event_id, "confusion")
    total_tiles = count_tiles(height, width, tile_size)

    count = 0
    tile_iterator = tqdm(
        iter_tile_windows(height, width, tile_size),
        total=total_tiles,
        desc=f"[confusion] {event_id}",
        unit="tile",
    )
    for tile_info in tile_iterator:
        row = tile_info["row"]
        col = tile_info["col"]
        tile_height = tile_info["height"]
        tile_width = tile_info["width"]

        rgb_tile = source["rgb"][row : row + tile_height, col : col + tile_width, :]
        gt_tile = source["gt_display"][row : row + tile_height, col : col + tile_width]
        cm_maps_tile = {
            model_name: cm_map[row : row + tile_height, col : col + tile_width]
            for model_name, cm_map in source["cm_maps"].items()
        }
        spatial_meta_tile = crop_spatial_meta(source["spatial_meta"], tile_info["window"])

        figure = shared.draw_confusion_compare_figure(
            rgb=rgb_tile,
            gt_display=gt_tile,
            cm_maps=cm_maps_tile,
            spatial_meta=spatial_meta_tile,
            figure_size=shared.CONFUSION_FIGURE_SIZE,
            show_metrics=show_metrics,
        )
        output_path = output_dir / build_tile_filename(event_id, tile_info, "confusion_compare")
        confusion_plot.save_figure(figure, output_path, dpi)
        count += 1

    print(f"[confusion] {event_id}: saved {count} tiled figures to {output_dir}")


def load_pavpu_thresholds(analysis_root, retention_target):
    thresholds = {}

    for model_name, source_map in PAVPU_THRESHOLD_KEYS.items():
        thresholds[model_name] = {}
        for source_name, criteria_name in source_map.items():
            unc_threshold, acc_threshold = get_thresholds(
                model_name,
                criteria_name,
                str(analysis_root),
                retention_target,
            )
            thresholds[model_name][source_name] = {
                "unc_threshold": unc_threshold,
                "acc_threshold": acc_threshold,
            }

    return thresholds


def compute_pavpu_value(cm_map_tile, unc_map_tile, patch_size, acc_threshold, unc_threshold):
    _, _, _, _, pavpu = compute_pavpu_for_image(
        cm_map_tile,
        unc_map_tile,
        patch_size,
        acc_threshold,
        unc_threshold,
    )
    return float(pavpu)


def serialize_pavpu_value(value):
    if np.isnan(value):
        return ""
    return f"{value:.6f}"


def build_pavpu_csv_row(event_id, tile_info, tile_filename, tile_models, pavpu_thresholds, patch_size):
    row = {
        "event_id": event_id,
        "row": tile_info["row"],
        "col": tile_info["col"],
        "height": tile_info["height"],
        "width": tile_info["width"],
        "tile_filename": tile_filename,
    }

    for field_name, model_name, source_name in PAVPU_CSV_ITEMS:
        threshold = pavpu_thresholds[model_name][source_name]
        value = compute_pavpu_value(
            tile_models[model_name]["cm_map"],
            tile_models[model_name]["criteria"][source_name],
            patch_size,
            threshold["acc_threshold"],
            threshold["unc_threshold"],
        )
        row[field_name] = serialize_pavpu_value(value)

    return row


def write_pavpu_csv(output_dir, event_id, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{event_id}_uncertainty_pavpu.csv"
    fieldnames = [
        "event_id",
        "row",
        "col",
        "height",
        "width",
        "tile_filename",
        *[field_name for field_name, _, _ in PAVPU_CSV_ITEMS],
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def load_uncertainty_source(event_id, args):
    path_dict = uncertainty_plot.build_paths(
        event_id=event_id,
        subset=args.subset,
        data_root=args.data_root,
        pred_root=args.pred_root,
    )
    uncertainty_plot.validate_required_paths(path_dict)

    rgb_image = uncertainty_plot.load_rgb_image(path_dict["s2"])
    spatial_meta = uncertainty_plot.load_spatial_meta(path_dict["s2"])
    model_entries = {
        model_name: uncertainty_plot.load_model_entry(model_name, path_dict["models"][model_name])
        for model_name in uncertainty_plot.MODEL_ORDER
    }
    uncertainty_plot.validate_shapes(rgb_image, model_entries)

    return {
        "event_id": event_id,
        "rgb_image": rgb_image,
        "spatial_meta": spatial_meta,
        "model_entries": model_entries,
    }


def build_uncertainty_tile_entry(full_entry, row, col, tile_height, tile_width, stride):
    tile_entry = {
        "cm_map": full_entry["cm_map"][row : row + tile_height, col : col + tile_width],
        "valid_mask": full_entry["valid_mask"][row : row + tile_height, col : col + tile_width],
        "fp_mask": full_entry["fp_mask"][row : row + tile_height, col : col + tile_width],
        "fn_mask": full_entry["fn_mask"][row : row + tile_height, col : col + tile_width],
        "criteria": {
            source_name: values[row : row + tile_height, col : col + tile_width]
            for source_name, values in full_entry["criteria"].items()
        },
    }

    fp_fill = uncertainty_plot.downsample_2d(tile_entry["fp_mask"], stride)
    fn_fill = uncertainty_plot.downsample_2d(tile_entry["fn_mask"], stride)
    fp_outline = uncertainty_plot.downsample_2d(tile_entry["fp_mask"], stride)
    fn_outline = uncertainty_plot.downsample_2d(tile_entry["fn_mask"], stride)

    fp_outline = uncertainty_plot.dilate_mask(
        uncertainty_plot.mask_to_outline(fp_outline),
        uncertainty_plot.FP_OUTLINE_THICKNESS,
    )
    fn_outline = uncertainty_plot.dilate_mask(
        uncertainty_plot.mask_to_outline(fn_outline),
        uncertainty_plot.FN_OUTLINE_THICKNESS,
    )

    precomputed_maps = {}
    for source_name in tile_entry["criteria"]:
        masked_map = uncertainty_plot.get_masked_map(tile_entry, source_name)
        masked_map = uncertainty_plot.downsample_2d(masked_map, stride)
        precomputed_maps[source_name] = {
            "masked_map": masked_map,
        }

    tile_entry["fp_fill"] = fp_fill
    tile_entry["fn_fill"] = fn_fill
    tile_entry["fp_outline"] = fp_outline
    tile_entry["fn_outline"] = fn_outline
    tile_entry["precomputed_maps"] = precomputed_maps
    return tile_entry


def export_uncertainty_tiles(source, output_root, tile_size, dpi, pavpu_thresholds, patch_size, show_pavpu=False):
    event_id = source["event_id"]
    _, height, width = source["rgb_image"].shape
    output_dir = build_output_dir(output_root, tile_size, event_id, "uncertainty")
    total_tiles = count_tiles(height, width, tile_size)

    count = 0
    pavpu_rows = []
    tile_iterator = tqdm(
        iter_tile_windows(height, width, tile_size),
        total=total_tiles,
        desc=f"[uncertainty] {event_id}",
        unit="tile",
    )
    for tile_info in tile_iterator:
        row = tile_info["row"]
        col = tile_info["col"]
        tile_height = tile_info["height"]
        tile_width = tile_info["width"]
        stride = uncertainty_plot.compute_stride((tile_height, tile_width))

        rgb_tile = source["rgb_image"][:, row : row + tile_height, col : col + tile_width]
        rgb_plot = (uncertainty_plot.downsample_rgb(rgb_tile, stride) / 3500.0).clip(0, 1)
        spatial_meta_tile = crop_spatial_meta(source["spatial_meta"], tile_info["window"])

        tile_models = {}
        for model_name, full_entry in source["model_entries"].items():
            tile_models[model_name] = build_uncertainty_tile_entry(
                full_entry=full_entry,
                row=row,
                col=col,
                tile_height=tile_height,
                tile_width=tile_width,
                stride=stride,
            )

        output_filename = build_tile_filename(event_id, tile_info, "uncertainty_compare")
        if show_pavpu:
            pavpu_rows.append(
                build_pavpu_csv_row(
                    event_id=event_id,
                    tile_info=tile_info,
                    tile_filename=output_filename,
                    tile_models=tile_models,
                    pavpu_thresholds=pavpu_thresholds,
                    patch_size=patch_size,
                )
            )

        event_bundle = {
            "rgb_plot": rgb_plot,
            "models": tile_models,
            "spatial_meta": spatial_meta_tile,
            "stride": stride,
            "height_ds": rgb_plot.shape[0],
            "width_ds": rgb_plot.shape[1],
        }

        figure = shared.draw_uncertainty_compare_figure(
            event_bundle=event_bundle,
            figure_size=shared.UNCERTAINTY_FIGURE_SIZE,
        )
        output_path = output_dir / output_filename
        uncertainty_plot.save_figure(figure, output_path, dpi)
        count += 1

    print(f"[uncertainty] {event_id}: saved {count} tiled figures to {output_dir}")
    if show_pavpu:
        csv_path = write_pavpu_csv(output_dir, event_id, pavpu_rows)
        print(f"[uncertainty] {event_id}: saved PAVPU CSV to {csv_path}")


def main():
    args = parse_args()
    event_ids = resolve_event_ids(args.event_ids)
    pavpu_thresholds = None

    if args.kind in {"uncertainty", "both"} and args.show_pavpu:
        pavpu_thresholds = load_pavpu_thresholds(DEFAULT_ANALYSIS_ROOT, args.retention_target)

    for event_id in event_ids:
        if args.kind in {"confusion", "both"}:
            confusion_source = load_confusion_source(event_id, args)
            export_confusion_tiles(
                confusion_source,
                args.output_root,
                args.tile_size,
                args.dpi,
                show_metrics=args.show_confusion_metrics,
            )

        if args.kind in {"uncertainty", "both"}:
            uncertainty_source = load_uncertainty_source(event_id, args)
            export_uncertainty_tiles(
                uncertainty_source,
                args.output_root,
                args.tile_size,
                args.dpi,
                pavpu_thresholds,
                args.patch_size,
                show_pavpu=args.show_pavpu,
            )


if __name__ == "__main__":
    main()
