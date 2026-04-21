from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tqdm import tqdm


SUBSET = "test"
DATA_ROOT = Path("/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data")
PRED_ROOT = Path("/home/NAS/homes/cjchen-10025/ML4FloodsUncertainty/result/val_test_inference")
OUTPUT_ROOT = Path("/home/NAS/homes/cjchen-10025/ML4FloodsUncertainty/result/analysis_S2/compare")

MODEL_ORDER = ["EDL", "ensemble", "mcdropout"]
MAX_PLOT_SIZE = 1600
OUTLINE_THICKNESS = 1

FP_COLOR = "#d62828"
FN_COLOR = "#ffb703"
INVALID_COLOR = "#d9d9d9"
UNCERTAINTY_CMAP = "Blues"
FP_FILL_ALPHA = 0.5
FN_FILL_ALPHA = 0.5

MODEL_BANDS = {
    "EDL": {
        "dstu": 1,
        "water_probability": 2,
        "aleatoric": 5,
        "epistemic": 6,
    },
    "ensemble": {
        "water_probability": 2,
        "aleatoric": 4,
        "epistemic": 6,
    },
    "mcdropout": {
        "water_probability": 2,
        "aleatoric": 4,
        "epistemic": 6,
    },
}

MODE_LAYOUTS = {
    "epistemic_plus_aleatoric_with_dstu": [
        {"kind": "rgb", "title": "Original S2 RGB"},
        {"kind": "map", "model": "EDL", "source": "dstu", "title": "EDL DST-u"},
        {
            "kind": "map",
            "model": "EDL",
            "source": "epistemic_plus_aleatoric",
            "title": "EDL Epistemic + Aleatoric",
        },
        {
            "kind": "map",
            "model": "ensemble",
            "source": "epistemic_plus_aleatoric",
            "title": "ensemble Epistemic + Aleatoric",
        },
        {
            "kind": "map",
            "model": "mcdropout",
            "source": "epistemic_plus_aleatoric",
            "title": "mcdropout Epistemic + Aleatoric",
        },
    ],
    "aleatoric": [
        {"kind": "rgb", "title": "Original S2 RGB"},
        {"kind": "map", "model": "EDL", "source": "aleatoric", "title": "EDL Aleatoric"},
        {"kind": "map", "model": "ensemble", "source": "aleatoric", "title": "ensemble Aleatoric"},
        {"kind": "map", "model": "mcdropout", "source": "aleatoric", "title": "mcdropout Aleatoric"},
    ],
    "epistemic": [
        {"kind": "rgb", "title": "Original S2 RGB"},
        {"kind": "map", "model": "EDL", "source": "epistemic", "title": "EDL Epistemic"},
        {"kind": "map", "model": "ensemble", "source": "epistemic", "title": "ensemble Epistemic"},
        {"kind": "map", "model": "mcdropout", "source": "epistemic", "title": "mcdropout Epistemic"},
    ],
    "water_probability": [
        {"kind": "rgb", "title": "Original S2 RGB"},
        {
            "kind": "map",
            "model": "EDL",
            "source": "water_probability",
            "title": "EDL Water Probability",
        },
        {
            "kind": "map",
            "model": "ensemble",
            "source": "water_probability",
            "title": "ensemble Water Probability",
        },
        {
            "kind": "map",
            "model": "mcdropout",
            "source": "water_probability",
            "title": "mcdropout Water Probability",
        },
    ],
}


def build_model_paths(filename):
    return {
        "EDL": {
            "pred_path": PRED_ROOT / SUBSET / "EDL" / f"{filename}_output_EDL.tif",
            "cm_path": PRED_ROOT / SUBSET / "EDL" / f"cm_{filename}_output_EDL.tif",
        },
        "ensemble": {
            "pred_path": PRED_ROOT / SUBSET / "ensemble" / f"{filename}_output_ensemble.tif",
            "cm_path": PRED_ROOT / SUBSET / "ensemble" / f"cm_{filename}_output_ensemble.tif",
        },
        "mcdropout": {
            "pred_path": PRED_ROOT / SUBSET / "mcdropout" / f"{filename}_mcdropout_output_ensemble.tif",
            "cm_path": PRED_ROOT / SUBSET / "mcdropout" / f"cm_{filename}_mcdropout_output_ensemble.tif",
        },
    }


def list_test_filenames():
    s2_dir = DATA_ROOT / SUBSET / "S2"
    return sorted(path.stem for path in s2_dir.glob("*.tif"))


def collect_required_paths(filename):
    model_paths = build_model_paths(filename)
    required_paths = [DATA_ROOT / SUBSET / "S2" / f"{filename}.tif"]
    for model_name in MODEL_ORDER:
        required_paths.append(model_paths[model_name]["pred_path"])
        required_paths.append(model_paths[model_name]["cm_path"])
    return required_paths, model_paths


def has_all_required_paths(paths):
    return all(path.exists() for path in paths)


def load_rgb_image(filename):
    s2_path = DATA_ROOT / SUBSET / "S2" / f"{filename}.tif"
    with rasterio.open(s2_path) as src:
        return src.read([4, 3, 2]).astype(np.float32)


def load_model_entry(model_name, model_paths):
    pred_path = model_paths[model_name]["pred_path"]
    cm_path = model_paths[model_name]["cm_path"]
    band_config = MODEL_BANDS[model_name]

    criteria_names = list(band_config.keys())
    rasterio_bands = [band_config[name] + 1 for name in criteria_names]

    with rasterio.open(pred_path) as src:
        band_stack = src.read(rasterio_bands).astype(np.float32)

    criteria = {}
    for idx, criteria_name in enumerate(criteria_names):
        criteria[criteria_name] = band_stack[idx]

    criteria["epistemic_plus_aleatoric"] = (
        criteria["epistemic"] + criteria["aleatoric"]
    )

    with rasterio.open(cm_path) as src:
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


def downsample_2d(array, stride):
    return array[::stride, ::stride]


def downsample_rgb(rgb_image, stride):
    rgb = np.transpose(rgb_image, (1, 2, 0))
    return rgb[::stride, ::stride]


def compute_local_limits(masked_array):
    valid_values = masked_array.compressed()
    if not valid_values.size:
        raise ValueError("No valid uncertainty values found.")

    vmin = float(np.nanpercentile(valid_values, 2))
    vmax = float(np.nanpercentile(valid_values, 98))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return vmin, vmax


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


def build_overlay(outline_mask, color_hex):
    color = np.array(plt.matplotlib.colors.to_rgba(color_hex))
    overlay = np.zeros(outline_mask.shape + (4,), dtype=np.float32)
    overlay[outline_mask] = color
    return overlay


def build_fill_overlay(mask, color_hex, alpha):
    color = np.array(plt.matplotlib.colors.to_rgba(color_hex, alpha=alpha))
    overlay = np.zeros(mask.shape + (4,), dtype=np.float32)
    overlay[mask] = color
    return overlay


def get_masked_map(model_entry, source_name):
    base_map = model_entry["criteria"][source_name]
    valid_mask = model_entry["valid_mask"]
    return np.ma.masked_where(~valid_mask, base_map)


def precompute_model_visuals(model_entry, stride):
    fp_fill = downsample_2d(model_entry["fp_mask"], stride)
    fn_fill = downsample_2d(model_entry["fn_mask"], stride)
    fp_outline = downsample_2d(model_entry["fp_mask"], stride)
    fn_outline = downsample_2d(model_entry["fn_mask"], stride)
    fp_outline = dilate_mask(mask_to_outline(fp_outline), OUTLINE_THICKNESS)
    fn_outline = dilate_mask(mask_to_outline(fn_outline), OUTLINE_THICKNESS)

    precomputed_maps = {}
    for source_name in model_entry["criteria"]:
        masked_map = get_masked_map(model_entry, source_name)
        masked_map = downsample_2d(masked_map, stride)
        vmin, vmax = compute_local_limits(masked_map)
        precomputed_maps[source_name] = {
            "masked_map": masked_map,
            "vmin": vmin,
            "vmax": vmax,
        }

    model_entry["fp_fill"] = fp_fill
    model_entry["fn_fill"] = fn_fill
    model_entry["fp_outline"] = fp_outline
    model_entry["fn_outline"] = fn_outline
    model_entry["precomputed_maps"] = precomputed_maps
    return model_entry


def load_event_bundle(filename, model_paths):
    rgb_image = load_rgb_image(filename)
    model_data = {
        model_name: load_model_entry(model_name, model_paths)
        for model_name in MODEL_ORDER
    }

    sample_shape = model_data["EDL"]["cm_map"].shape
    stride = compute_stride(sample_shape)
    rgb_plot = (downsample_rgb(rgb_image, stride) / 3500).clip(0, 1)

    for model_name in MODEL_ORDER:
        model_data[model_name] = precompute_model_visuals(model_data[model_name], stride)

    return {
        "rgb_plot": rgb_plot,
        "stride": stride,
        "models": model_data,
    }


def add_map_panel(fig, ax, panel_config, model_entry, cmap):
    map_data = model_entry["precomputed_maps"][panel_config["source"]]
    image = ax.imshow(
        map_data["masked_map"],
        cmap=cmap,
        vmin=map_data["vmin"],
        vmax=map_data["vmax"],
        interpolation="nearest",
    )
    ax.imshow(build_fill_overlay(model_entry["fp_fill"], FP_COLOR, FP_FILL_ALPHA), interpolation="nearest")
    ax.imshow(build_fill_overlay(model_entry["fn_fill"], FN_COLOR, FN_FILL_ALPHA), interpolation="nearest")
    ax.imshow(build_overlay(model_entry["fp_outline"], FP_COLOR), interpolation="nearest")
    ax.imshow(build_overlay(model_entry["fn_outline"], FN_COLOR), interpolation="nearest")
    ax.set_title(panel_config["title"])
    ax.set_xticks([])
    ax.set_yticks([])

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(panel_config["title"])


def plot_mode_figure(filename, mode_name, event_bundle):
    layout = MODE_LAYOUTS[mode_name]
    figure_width = 6 * len(layout)
    fig, axes = plt.subplots(1, len(layout), figsize=(figure_width, 7), constrained_layout=True)

    axes[0].imshow(event_bundle["rgb_plot"], interpolation="nearest")
    axes[0].set_title(layout[0]["title"])
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    cmap = plt.get_cmap(UNCERTAINTY_CMAP).copy()
    cmap.set_bad(color=INVALID_COLOR)

    for ax, panel_config in zip(axes[1:], layout[1:]):
        model_entry = event_bundle["models"][panel_config["model"]]
        add_map_panel(fig, ax, panel_config, model_entry, cmap)

    legend_handles = [
        mlines.Line2D([], [], color=FP_COLOR, linewidth=2, linestyle="-", label="FP outline"),
        mlines.Line2D([], [], color=FN_COLOR, linewidth=2, linestyle="-", label="FN outline"),
    ]
    axes[-1].legend(handles=legend_handles, loc="lower right", frameon=True)
    fig.suptitle(f"{filename}: {mode_name}", fontsize=18)
    return fig


def save_mode_figure(filename, mode_name, event_bundle):
    mode_output_dir = OUTPUT_ROOT / mode_name
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    figure = plot_mode_figure(filename, mode_name, event_bundle)
    output_path = mode_output_dir / f"{filename}_{mode_name}_fp_fn_compare.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    filenames = list_test_filenames()
    print(f"Found {len(filenames)} test events")

    for filename in tqdm(filenames, desc="Generating compare figures"):
        required_paths, model_paths = collect_required_paths(filename)
        if not has_all_required_paths(required_paths):
            print(f"[SKIP] Missing files for {filename}")
            continue

        event_bundle = load_event_bundle(filename, model_paths)
        for mode_name in MODE_LAYOUTS:
            save_mode_figure(filename, mode_name, event_bundle)


if __name__ == "__main__":
    main()
