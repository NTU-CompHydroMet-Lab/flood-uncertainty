"""Check and plot the first N retained training patches of one WorldFloods event.

From the project root: uv run python -m scripts.preprocess.check_water_only_kl_dataloader_1stpatch --num-patches 16
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from rasterio.windows import Window
from torch.utils.data import DataLoader

from flood_uncertainty.data.worldfloods_kl import WorldFloodsKLDatasetTiled
from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.data.worldfloods.configs import BANDS_S2, CHANNELS_CONFIGURATIONS
from ml4floods.data.worldfloods.dataset import WorldFloodsDatasetTiled
from ml4floods.models.dataset_setup import get_transformations
from ml4floods.preprocess.tiling import Dict_to_WindowSlices, WindowSize, WindowSlices, get_window_tiles
from ml4floods.preprocess.worldfloods.normalize import get_normalisation
from ml4floods.preprocess.worldfloods.prepare_patches import filter_windows_v2
from scripts.preprocess.generate_water_only_kl_masks import validate_rasters
from scripts.preprocess.preview_water_only_kl_mask import GT_COLORS, KL_COLOR
from scripts.train.train_edl_water_kl import BASELINE_INDEX_SHA256, split_pair


@dataclass
class Patch:
    image: np.ndarray  # spectral bands, C x H x W
    gt: np.ndarray     # cloud and water, 2 x H x W
    kl: np.ndarray     # H x W


def train_pairs(data) -> list[tuple[str, str]]:
    """Train (S2, GT) paths exactly as training reads them from the local split file."""
    train = json.loads(Path(data.train_test_split_file).read_text())["train"]
    return list(zip(train[data.input_folder], train[data.target_folder], strict=True))


def training_pair(pairs, event, data) -> tuple[str, str]:
    matches = [(s2, gt) for s2, gt in pairs if Path(s2).stem == event]
    if len(matches) != 1:
        raise ValueError(f"Expected one training pair for {event}, found {len(matches)}")
    split_pair(*matches[0], data, "train")  # Same S2/GT layout check as training.
    return matches[0]


def event_training_windows(data, image_file: str, train_images: list[str]) -> list[Window]:
    """Return one event's training windows in training order, chosen as in train_edl_water_kl.snapshot_windows.

    The pinned window index is used only when training would use it. Otherwise this event is
    filtered again; filter_windows_v2 keeps each window independently, so the result matches
    training. Read-only: unlike ml4floods get_dataset, this never writes a window cache.
    """
    rules = data.get("filter_windows") or {}
    size = WindowSize(*data.window_size)
    with rasterio.open(image_file) as source:
        tiles = get_window_tiles(source, height=size.height, width=size.width)
    if not rules.get("apply", False):
        return tiles
    if rules["version"] != "v2":
        raise ValueError("Water-only KL training requires the WorldFloods v2 window filter")
    cache = Path(data.path_to_splits) / f"windows_{Path(data.train_test_split_file).name}"
    baseline_settings = rules["threshold_clouds"] == 0.8 and size == (256, 256)
    cached = cache.read_bytes() if baseline_settings and cache.is_file() else b""
    pinned = cached and hashlib.sha256(cached).hexdigest() == BASELINE_INDEX_SHA256
    rows = json.loads(cached)["slices"] if pinned else []
    # Training uses the index only when it covers every train image.
    if rows and set(train_images).issubset(row["file_name"] for row in rows):
        return [Dict_to_WindowSlices(row).window for row in rows if row["file_name"] == image_file]
    # Same recomputation as training: delimited prefixes, zero-padded edge labels, fraction strictly below.
    raw = WorldFloodsDatasetTiled([WindowSlices(image_file, tile) for tile in tiles],
                                  image_prefix=f"/{data.input_folder}/", gt_prefix=f"/{data.target_folder}/")
    return [tile.window for tile in filter_windows_v2(raw, threshold_clouds=rules["threshold_clouds"])]


def training_dataset(data, image_file: str, windows: list[Window], mask_root: Path, seed: int):
    """KL dataset built as in training: same bands, path prefixes and a seeded train transform."""
    if data.get("add_mndwi_input"):
        raise ValueError("Water-only KL training rejects add_mndwi_input, so this check does too")
    train_transform, _ = get_transformations(data)
    train_transform.set_random_seed(seed)
    return WorldFloodsKLDatasetTiled(
        [WindowSlices(image_file, window) for window in windows], kl_mask_root=mask_root,
        # Same as training: slashes keep the path replacement on the folder, not a filename.
        image_prefix=f"/{data.input_folder}/", gt_prefix=f"/{data.target_folder}/",
        bands=CHANNELS_CONFIGURATIONS[data.channel_configuration], transforms=train_transform,
    )


def unpack_patch(batch):
    image, gt, kl = (batch[key][0].numpy() for key in ("image", "mask", "kl_mask"))
    if image.ndim != 3 or gt.shape != (2, *image.shape[-2:]) or kl.shape != (1, *image.shape[-2:]):
        raise ValueError(f"Unexpected patch shapes: image={image.shape}, GT={gt.shape}, KL={kl.shape}")
    if not np.isin(kl, (0, 1)).all() or not np.isin(gt, (0, 1, 2)).all():
        raise ValueError("Unexpected values in DataLoader GT or KL labels")
    return Patch(image, gt, kl[0])


def orient(array, turns, flip):
    result = np.rot90(array, turns, axes=(-2, -1))
    return np.flip(result, axis=-1) if flip else result


def check_alignment(source, batch):
    """Infer geometry from all spectral bands in raw S2 units, then compare discrete labels exactly."""
    matches = []
    for turns in range(4):
        for flip in (False, True):
            candidate = orient(source.image, turns, flip)
            if candidate.shape == batch.image.shape and np.allclose(candidate, batch.image, rtol=1e-6, atol=1e-3):
                matches.append((turns, flip))
    result = dict(image_matching_orientations=len(matches), gt_mismatch_pixels=None, kl_mismatch_pixels=None)
    if len(matches) != 1:
        result["status"] = "image_mismatch" if not matches else "ambiguous_image_geometry"
        return result
    turns, flip = matches[0]
    gt_diff = np.any(orient(source.gt, turns, flip) != batch.gt, axis=0)
    kl_diff = orient(source.kl, turns, flip) != batch.kl
    result.update(
        rotation_ccw_degrees=90 * turns, horizontal_flip_after_rotation=flip,
        gt_mismatch_pixels=int(gt_diff.sum()), kl_mismatch_pixels=int(kl_diff.sum()),
        status="matched" if not (gt_diff.any() or kl_diff.any()) else "label_mismatch",
    )
    return result


def plot_comparison(path, title, source, batch, bands):
    rgb_indices = [bands.index(BANDS_S2.index(name)) for name in ("B4", "B3", "B2")]
    figure = Figure(figsize=(14, 7.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 4, squeeze=False)
    for row, (name, patch) in enumerate((("Source TIFF", source), ("DataLoader", batch))):
        rgb = np.clip(np.moveaxis(patch.image[rgb_indices], 0, -1) / 3500.0, 0, 1)
        axes[row, 0].imshow(rgb, interpolation="nearest")
        axes[row, 1].imshow(patch.gt[1], cmap=ListedColormap(GT_COLORS), vmin=0, vmax=2, interpolation="nearest")
        axes[row, 2].imshow(patch.kl, cmap=ListedColormap(["#161616", KL_COLOR]), vmin=0, vmax=1, interpolation="nearest")
        axes[row, 3].imshow(rgb, interpolation="nearest")
        axes[row, 3].imshow(np.ma.masked_equal(patch.kl, 0), cmap=ListedColormap([KL_COLOR]),
                            vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
        axes[row, 0].set_ylabel(name)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, heading in zip(axes[0], ("S2 RGB", "Water GT (blue = water)", "KL mask (pink = 1)", "RGB + KL")):
        axis.set_title(heading)
    figure.suptitle(title)
    figure.savefig(path, dpi=150)
    figure.clear()


def audit_patch(plot_path, event, window, source, batch, bands):
    alignment = check_alignment(source, batch)
    source_kl = int(np.count_nonzero(source.kl == 1))
    batch_kl = int(np.count_nonzero(batch.kl == 1))
    overlap = int(np.count_nonzero((batch.kl == 1) & (batch.gt[1] != 0)))
    report = dict(
        alignment=alignment, source_patch_kl_pixels=source_kl,
        counts_match=(source_kl == batch_kl), kl_on_valid_gt_pixels=overlap,
        example_window_col_row_width_height=[int(v) for v in window.flatten()],
        example_water_pixels=int(np.count_nonzero(batch.gt[1] == 2)),
        example_kl_pixels=batch_kl, plot=plot_path.name,
    )
    passed = report["counts_match"] and overlap == 0 and alignment["status"] == "matched"
    report["status"] = "passed" if passed else "failed"
    geometry = alignment["status"]
    if alignment["image_matching_orientations"] == 1:
        geometry = f"{alignment['rotation_ccw_degrees']}° CCW"
        if alignment["horizontal_flip_after_rotation"]:
            geometry += " + horizontal flip"
    title = (
        f"{event} | col={window.col_off}, row={window.row_off} | {report['status']}\n"
        f"{geometry} | water: {report['example_water_pixels']:,} | KL: {report['example_kl_pixels']:,} | "
        f"GT mismatches: {alignment['gt_mismatch_pixels']} | KL mismatches: {alignment['kl_mismatch_pixels']}"
    )
    plot_comparison(plot_path, title, source, batch, bands)
    return report


def audit_event(args, out_dir):
    config = load_mode_config(str(args.config), mode="train")
    data = config.data_params
    pairs = train_pairs(data)
    image_file, gt_file = training_pair(pairs, args.event, data)
    image_path, gt_path = Path(image_file), Path(gt_file)
    kl_path = args.mask_root / "train" / "kl_mask" / gt_path.name
    windows = event_training_windows(data, image_file, [s2 for s2, _ in pairs])[:args.num_patches]
    seed = config.seed if args.seed is None else args.seed
    report = dict(
        scope="first_retained_patch" if args.num_patches == 1 else "first_retained_patches",
        event=args.event, config=str(args.config), seed=seed,
        source_image=str(image_path), source_gt=str(gt_path), source_kl=str(kl_path),
        configured_filter=dict(data.get("filter_windows") or {}), window_size=list(data.window_size),
        requested_patches=args.num_patches, patches_seen=0, patches=[],
    )
    if not windows:
        report.update(status="no_retained_patches", plot=None, alignment=None)
        return report

    dataset = training_dataset(data, image_file, windows, args.mask_root, seed)
    bands = list(dataset.channels_read)
    raster_bands = [band + 1 for band in bands]
    normalization = None
    if data.train_transformation.normalize:
        normalization = get_normalisation(data.channel_configuration, channels_first=True)
    batches = iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    with rasterio.open(image_path) as image, rasterio.open(gt_path) as gt, rasterio.open(kl_path) as mask:
        validate_rasters(image, gt)
        if mask.count != 1 or mask.dtypes != ("uint8",) or mask.nodata is not None:
            raise ValueError(f"Expected a single-band uint8 KL mask with no nodata: {kl_path}")
        if (mask.shape, mask.crs, mask.transform) != (gt.shape, gt.crs, gt.transform):
            raise ValueError(f"KL mask/GT grid mismatch: {kl_path}")
        for index, window in enumerate(windows, start=1):
            if ((window.height, window.width) != tuple(data.window_size)
                    or not (0 <= window.col_off < gt.width and 0 <= window.row_off < gt.height)):
                raise ValueError(f"Training window does not match configured tile size or event bounds: {window}")
            batch = unpack_patch(next(batches))
            batch.image = batch.image[:len(bands)]
            if normalization is not None:
                mean, std = normalization
                batch.image = batch.image * std + mean
            read = dict(window=window, boundless=True, fill_value=0)
            source = Patch(
                np.nan_to_num(image.read(raster_bands, **read)).astype(np.float32),
                gt.read([1, 2], **read), mask.read(1, **read),
            )
            filename = "patch.png" if args.num_patches == 1 else f"patch_{index:03d}.png"
            plot_path = out_dir / filename
            patch_report = audit_patch(plot_path, args.event, window, source, batch, bands)
            report["patches"].append(patch_report)
            print(f"[{index}/{len(windows)}] {patch_report['status']}  {plot_path}", flush=True)

    report["patches_seen"] = len(report["patches"])
    report["status"] = "passed" if all(p["status"] == "passed" for p in report["patches"]) else "failed"
    if args.num_patches == 1:
        report.update(report["patches"][0])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="EMSR260_01CICOGNARA_DEL_MONIT01_v2")
    parser.add_argument(
        "--num-patches", type=int, default=16, metavar="N",
        help="Plot up to N retained patches from this event in training-list order (default: 16)",
    )
    parser.add_argument("--config", type=Path, default=Path("configurations/edl_water.json"))
    parser.add_argument("--mask-root", type=Path, default=Path("artifacts/kl_masks/worldfloods_v2"))
    parser.add_argument("--output-dir", type=Path, help="Parent directory for per-run results")
    parser.add_argument("--seed", type=int, help="Augmentation seed (default: training config seed)")
    args = parser.parse_args()
    if not args.event or Path(args.event).name != args.event or args.event in (".", ".."):
        parser.error("--event must be an event stem")
    if args.num_patches < 1:
        parser.error("--num-patches must be at least 1")
    parent = args.output_dir or args.mask_root / "train" / "dataloader_check"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = parent / f"{args.event}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"Checking up to {args.num_patches} training patches of {args.event}. Results: {out_dir}", flush=True)
    report = audit_event(args, out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Plotted {report['patches_seen']} of {args.num_patches} requested patches. Status: {report['status']}")
    print(f"Report: {out_dir / 'report.json'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
