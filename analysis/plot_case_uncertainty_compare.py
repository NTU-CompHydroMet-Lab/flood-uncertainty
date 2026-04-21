import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

import case_compare_shared as shared


DEFAULT_EVENT_ID = "EMSR264_08VATOMANDRY_DEL_v2"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data")
DEFAULT_PRED_ROOT = REPO_ROOT / "result" / "val_test_inference"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "report" / "image_of_md"

MODEL_ORDER = ["EDL", "ensemble", "mcdropout"]
MAX_PLOT_SIZE = 1600
FP_OUTLINE_THICKNESS = 1
FN_OUTLINE_THICKNESS = 1

FP_COLOR = shared.UNCERTAINTY_FP_COLOR
FN_COLOR = shared.UNCERTAINTY_FN_COLOR
INVALID_COLOR = shared.UNCERTAINTY_INVALID_COLOR
UNCERTAINTY_CMAP = shared.UNCERTAINTY_CMAP
FP_FILL_ALPHA = shared.UNCERTAINTY_FP_FILL_ALPHA
FN_FILL_ALPHA = shared.UNCERTAINTY_FN_FILL_ALPHA
N_COORD_TICKS = shared.CONFUSION_N_COORD_TICKS

MODEL_BANDS = {
    "EDL": {
        "dstu": 1,
        "aleatoric": 5,
        "epistemic": 6,
    },
    "ensemble": {
        "aleatoric": 4,
        "epistemic": 6,
    },
    "mcdropout": {
        "aleatoric": 4,
        "epistemic": 6,
    },
}

MODEL_TITLES = {
    "EDL": "EDL",
    "ensemble": "Ensemble",
    "mcdropout": "MC Dropout",
}

MODEL_SUFFIXES = {
    "EDL": "_output_EDL",
    "ensemble": "_output_ensemble",
    "mcdropout": "_mcdropout_output_ensemble",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a combined uncertainty compare figure from S2, predictions, and confusion maps."
    )
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID, help="Event ID, e.g. EMSR264_08VATOMANDRY_DEL_v2")
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
        help="Output filename. Default: {event_id}_uncertainty_compare.png",
    )
    parser.add_argument("--dpi", default=300, type=int, help="Figure DPI. Default: 300")
    return parser.parse_args()


def build_paths(event_id, subset, data_root, pred_root):
    data_root = Path(data_root)
    pred_root = Path(pred_root)

    model_paths = {
        model_name: {
            "pred_path": pred_root / subset / model_name / f"{event_id}{MODEL_SUFFIXES[model_name]}.tif",
            "cm_path": pred_root / subset / model_name / f"cm_{event_id}{MODEL_SUFFIXES[model_name]}.tif",
        }
        for model_name in MODEL_ORDER
    }

    return {
        "s2": data_root / subset / "S2" / f"{event_id}.tif",
        "models": model_paths,
    }


def validate_required_paths(path_dict):
    missing_paths = []

    if not path_dict["s2"].exists():
        missing_paths.append(path_dict["s2"])

    for model_name in MODEL_ORDER:
        for key in ("pred_path", "cm_path"):
            path = path_dict["models"][model_name][key]
            if not path.exists():
                missing_paths.append(path)

    if missing_paths:
        missing_str = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required input file(s):\n{missing_str}")


def load_rgb_image(s2_path):
    with rasterio.open(s2_path) as src:
        return src.read([4, 3, 2]).astype(np.float32)


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


def load_model_entry(model_name, model_path_dict):
    band_config = MODEL_BANDS[model_name]
    criteria_names = list(band_config.keys())
    rasterio_bands = [band_config[name] + 1 for name in criteria_names]

    with rasterio.open(model_path_dict["pred_path"]) as src:
        band_stack = src.read(rasterio_bands).astype(np.float32)

    criteria = {}
    for idx, criteria_name in enumerate(criteria_names):
        criteria[criteria_name] = band_stack[idx]

    criteria["epistemic_plus_aleatoric"] = criteria["epistemic"] + criteria["aleatoric"]

    with rasterio.open(model_path_dict["cm_path"]) as src:
        cm_map = src.read(1).astype(np.int16)

    return {
        "cm_map": cm_map,
        "valid_mask": cm_map != 0,
        "fp_mask": cm_map == 2,
        "fn_mask": cm_map == 3,
        "criteria": criteria,
    }


def compute_stride(shape):
    height, width = shape
    return max(1, int(np.ceil(max(height, width) / MAX_PLOT_SIZE)))


def downsample_rgb(rgb_image, stride):
    rgb = np.transpose(rgb_image, (1, 2, 0))
    return rgb[::stride, ::stride]


def downsample_2d(array, stride):
    return array[::stride, ::stride]


def compute_local_limits(masked_array):
    valid_values = masked_array.compressed()
    if not valid_values.size:
        raise ValueError("No valid uncertainty values found.")
    return shared.compute_limits_from_valid_values(valid_values)


def make_tick_positions(size, n_ticks=N_COORD_TICKS):
    return shared.make_tick_positions(size, n_ticks)


def format_decimal_degree(value):
    return shared.format_decimal_degree(value)


def apply_lonlat_ticks(ax, spatial_meta, stride, width_ds, height_ds, show_x=False, show_y=False):
    return shared.apply_decimal_ticks_uncertainty(
        ax=ax,
        spatial_meta=spatial_meta,
        stride=stride,
        width_ds=width_ds,
        height_ds=height_ds,
        show_x=show_x,
        show_y=show_y,
    )


def mask_to_outline(mask):
    up = np.zeros_like(mask, dtype=bool)
    down = np.zeros_like(mask, dtype=bool)
    left = np.zeros_like(mask, dtype=bool)
    right = np.zeros_like(mask, dtype=bool)

    up[1:] = mask[:-1]
    down[:-1] = mask[1:]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]

    interior = mask & up & down & left & right
    return mask & ~interior


def dilate_mask(mask, iterations):
    dilated = mask.copy()
    for _ in range(iterations):
        up = np.zeros_like(dilated, dtype=bool)
        down = np.zeros_like(dilated, dtype=bool)
        left = np.zeros_like(dilated, dtype=bool)
        right = np.zeros_like(dilated, dtype=bool)
        up_left = np.zeros_like(dilated, dtype=bool)
        up_right = np.zeros_like(dilated, dtype=bool)
        down_left = np.zeros_like(dilated, dtype=bool)
        down_right = np.zeros_like(dilated, dtype=bool)

        up[1:] = dilated[:-1]
        down[:-1] = dilated[1:]
        left[:, 1:] = dilated[:, :-1]
        right[:, :-1] = dilated[:, 1:]
        up_left[1:, 1:] = dilated[:-1, :-1]
        up_right[1:, :-1] = dilated[:-1, 1:]
        down_left[:-1, 1:] = dilated[1:, :-1]
        down_right[:-1, :-1] = dilated[1:, 1:]

        dilated = (
            dilated
            | up
            | down
            | left
            | right
            | up_left
            | up_right
            | down_left
            | down_right
        )
    return dilated


def build_fill_overlay(mask, color_hex, alpha):
    return shared.build_fill_overlay(mask, color_hex, alpha)


def build_outline_overlay(mask, color_hex):
    return shared.build_outline_overlay(mask, color_hex)


def get_masked_map(model_entry, source_name):
    base_map = model_entry["criteria"][source_name]
    valid_mask = model_entry["valid_mask"]
    return np.ma.masked_where(~valid_mask, base_map)


def precompute_model_visuals(model_entry, stride):
    fp_fill = downsample_2d(model_entry["fp_mask"], stride)
    fn_fill = downsample_2d(model_entry["fn_mask"], stride)
    fp_outline = downsample_2d(model_entry["fp_mask"], stride)
    fn_outline = downsample_2d(model_entry["fn_mask"], stride)

    fp_outline = dilate_mask(mask_to_outline(fp_outline), FP_OUTLINE_THICKNESS)
    fn_outline = dilate_mask(mask_to_outline(fn_outline), FN_OUTLINE_THICKNESS)

    precomputed_maps = {}
    for source_name in model_entry["criteria"]:
        masked_map = get_masked_map(model_entry, source_name)
        masked_map = downsample_2d(masked_map, stride)
        precomputed_maps[source_name] = {
            "masked_map": masked_map,
        }

    model_entry["fp_fill"] = fp_fill
    model_entry["fn_fill"] = fn_fill
    model_entry["fp_outline"] = fp_outline
    model_entry["fn_outline"] = fn_outline
    model_entry["precomputed_maps"] = precomputed_maps
    return model_entry


def validate_shapes(rgb_image, model_entries):
    expected_shape = model_entries["EDL"]["cm_map"].shape
    if rgb_image.shape[1:] != expected_shape:
        raise ValueError(f"S2 shape {rgb_image.shape[1:]} does not match confusion map shape {expected_shape}")

    for model_name in MODEL_ORDER:
        cm_shape = model_entries[model_name]["cm_map"].shape
        if cm_shape != expected_shape:
            raise ValueError(
                f"Confusion map shape mismatch for {model_name}: {cm_shape} != {expected_shape}"
            )

        for source_name, values in model_entries[model_name]["criteria"].items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"Criteria shape mismatch for {model_name}/{source_name}: {values.shape} != {expected_shape}"
                )


def load_event_bundle(path_dict):
    rgb_image = load_rgb_image(path_dict["s2"])
    spatial_meta = load_spatial_meta(path_dict["s2"])
    model_entries = {
        model_name: load_model_entry(model_name, path_dict["models"][model_name])
        for model_name in MODEL_ORDER
    }

    validate_shapes(rgb_image, model_entries)

    stride = compute_stride(model_entries["EDL"]["cm_map"].shape)
    rgb_plot = (downsample_rgb(rgb_image, stride) / 3500.0).clip(0, 1)

    for model_name in MODEL_ORDER:
        model_entries[model_name] = precompute_model_visuals(model_entries[model_name], stride)

    return {
        "rgb_plot": rgb_plot,
        "models": model_entries,
        "spatial_meta": spatial_meta,
        "stride": stride,
        "height_ds": rgb_plot.shape[0],
        "width_ds": rgb_plot.shape[1],
    }


def build_panel_layout():
    return shared.build_uncertainty_layout()


def plot_compare_figure(event_bundle):
    return shared.draw_uncertainty_compare_figure(event_bundle)


def save_figure(fig, output_path, dpi):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_name = args.output_name or f"{args.event_id}_uncertainty_compare.png"
    output_path = Path(args.output_dir) / output_name

    path_dict = build_paths(
        event_id=args.event_id,
        subset=args.subset,
        data_root=args.data_root,
        pred_root=args.pred_root,
    )
    validate_required_paths(path_dict)

    event_bundle = load_event_bundle(path_dict)
    figure = plot_compare_figure(event_bundle)
    save_figure(figure, output_path, args.dpi)
    print(f"Saved compare figure to: {output_path}")


if __name__ == "__main__":
    main()
