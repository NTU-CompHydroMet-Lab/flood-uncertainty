"""Check one training event and plot an actual WorldFloods KL DataLoader patch.

From the project root: uv run python -m scripts.preprocess.check_water_only_kl_dataloader
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.data.worldfloods.configs import BANDS_S2
from ml4floods.preprocess.tiling import WindowSize, get_window_tiles
from ml4floods.preprocess.worldfloods.normalize import get_normalisation
from scripts.preprocess.check_water_only_kl_dataloader_1stpatch import (
    event_training_windows, train_pairs, training_dataset, training_pair,
)
from scripts.preprocess.generate_water_only_kl_masks import validate_rasters
from scripts.preprocess.preview_water_only_kl_mask import GT_COLORS, KL_COLOR


@dataclass
class Patch:
    image: np.ndarray  # spectral bands, C x H x W
    gt: np.ndarray     # cloud and water, 2 x H x W
    kl: np.ndarray     # H x W


def count_source_pixels(mask, window_size, windows):
    """Count the denominator from the full TIFF and the expected numerator from the actual split."""
    grid = get_window_tiles(mask, height=window_size.height, width=window_size.width)
    kept = {tile.window for tile in windows}
    if len(kept) != len(windows) or not kept.issubset(grid):
        raise ValueError("Training windows are duplicated or do not match the event's tiling grid")
    total = retained = 0
    for window in grid:
        values = mask.read(1, window=window)  # Out-of-image padding contributes zero.
        if np.any(values > 1):
            raise ValueError(f"Expected a binary KL mask in {mask.name}, {window}")
        count = int(np.count_nonzero(values))
        total += count
        if window in kept:
            retained += count
    return len(grid), total, retained


def inspect_batches(dataset):
    """Visit every retained patch and save the first with both water and KL, if available."""
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    counts = dict(patches_seen=0, patches_with_kl=0, loader_kl_pixels=0, kl_on_valid_gt_pixels=0)
    best_rank = None
    example = None
    for batch in loader:
        images, labels, masks = (batch[key].numpy() for key in ("image", "mask", "kl_mask"))
        expected_gt_shape = (len(images), 2, *images.shape[-2:])
        if images.ndim != 4 or labels.shape != expected_gt_shape or masks.shape != labels[:, :1].shape:
            raise ValueError(f"Unexpected batch shapes: image={images.shape}, GT={labels.shape}, KL={masks.shape}")
        if not np.isin(masks, (0, 1)).all() or not np.isin(labels, (0, 1, 2)).all():
            raise ValueError("Unexpected values in DataLoader GT or KL labels")
        for image, gt, kl in zip(images, labels, masks[:, 0], strict=True):
            tile = dataset.list_of_windows[counts["patches_seen"]]
            selected = kl == 1
            water = gt[1]
            n_kl = int(selected.sum())
            counts["patches_seen"] += 1
            counts["patches_with_kl"] += int(n_kl > 0)
            counts["loader_kl_pixels"] += n_kl
            counts["kl_on_valid_gt_pixels"] += int(np.count_nonzero(selected & (water != 0)))
            rank = (n_kl > 0, bool(np.any(water == 2)))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                example = (tile.window, Patch(image.copy(), gt.copy(), kl.copy()))
    return counts, example


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


def audit_event(args, out_dir):
    config = load_mode_config(str(args.config), mode="train")
    data = config.data_params
    pairs = train_pairs(data)
    image_file, gt_file = training_pair(pairs, args.event, data)
    image_path, gt_path = Path(image_file), Path(gt_file)
    kl_path = args.mask_root / "train" / "kl_mask" / gt_path.name
    window_size = WindowSize(*data.window_size)
    seed = config.seed if args.seed is None else args.seed
    # Same read-only window choice as the first-patch check; ml4floods get_dataset could write a window cache.
    event_windows = event_training_windows(data, image_file, [s2 for s2, _ in pairs])
    dataset = training_dataset(data, image_file, event_windows, args.mask_root, seed)
    windows = dataset.list_of_windows
    with rasterio.open(image_path) as image, rasterio.open(gt_path) as gt, rasterio.open(kl_path) as mask:
        validate_rasters(image, gt)
        if mask.count != 1 or mask.dtypes != ("uint8",) or mask.nodata is not None:
            raise ValueError(f"Expected a single-band uint8 KL mask with no nodata: {kl_path}")
        if (mask.shape, mask.crs, mask.transform) != (gt.shape, gt.crs, gt.transform):
            raise ValueError(f"KL mask/GT grid mismatch: {kl_path}")
        n_windows, full_kl, kept_kl = count_source_pixels(mask, window_size, windows)
        counts, example = inspect_batches(dataset)
        report = dict(
            event=args.event, config=str(args.config), seed=seed,
            source_image=str(image_path), source_gt=str(gt_path), source_kl=str(kl_path),
            configured_filter=dict(data.get("filter_windows") or {}), window_size=list(window_size),
            full_event_patches=n_windows, retained_patches=len(windows), full_event_kl_pixels=full_kl,
            retained_tiff_kl_pixels=kept_kl, **counts,
            kl_retention_percent=100 * counts["loader_kl_pixels"] / full_kl if full_kl else None,
            plot=None, alignment=None,
        )
        counts_match = counts["patches_seen"] == len(windows) and counts["loader_kl_pixels"] == kept_kl
        report["counts_match"] = counts_match
        if example is None:
            report["status"] = "no_retained_patches"
            return report

        window, batch = example
        read = dict(window=window, boundless=True, fill_value=0)
        bands = list(dataset.channels_read)
        source = Patch(
            np.nan_to_num(image.read([index + 1 for index in bands], **read)).astype(np.float32),
            gt.read([1, 2], **read), mask.read(1, **read),
        )
        batch.image = batch.image[:len(bands)]
        if data.train_transformation.normalize:
            mean, std = get_normalisation(data.channel_configuration, channels_first=True)
            batch.image = batch.image * std + mean

    alignment = check_alignment(source, batch)
    report.update(
        alignment=alignment,
        example_window_col_row_width_height=[int(v) for v in window.flatten()],
        example_water_pixels=int(np.count_nonzero(batch.gt[1] == 2)),
        example_kl_pixels=int(np.count_nonzero(batch.kl)), plot="patch.png",
    )
    passed = counts_match and counts["kl_on_valid_gt_pixels"] == 0 and alignment["status"] == "matched"
    report["status"] = "passed" if passed else "failed"
    if report["status"] == "passed" and kept_kl == 0:
        report["status"] = "no_retained_kl"
    geometry = alignment["status"]
    if alignment["image_matching_orientations"] == 1:
        geometry = f"{alignment['rotation_ccw_degrees']}° CCW"
        if alignment["horizontal_flip_after_rotation"]:
            geometry += " + horizontal flip"
    title = (
        f"{args.event} | col={window.col_off}, row={window.row_off} | {report['status']}\n"
        f"{geometry} | water: {report['example_water_pixels']:,} | KL: {report['example_kl_pixels']:,} | "
        f"GT mismatches: {alignment['gt_mismatch_pixels']} | KL mismatches: {alignment['kl_mismatch_pixels']}"
    )
    plot_comparison(out_dir / report["plot"], title, source, batch, bands)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="EMSR260_01CICOGNARA_DEL_MONIT01_v2")
    parser.add_argument("--config", type=Path, default=Path("configurations/edl_water.json"))
    parser.add_argument("--mask-root", type=Path, default=Path("artifacts/kl_masks/worldfloods_v2"))
    parser.add_argument("--output-dir", type=Path, help="Parent directory for per-run results")
    parser.add_argument("--seed", type=int, help="Augmentation seed (default: training config seed)")
    args = parser.parse_args()
    if not args.event or Path(args.event).name != args.event or args.event in (".", ".."):
        parser.error("--event must be an event stem")
    parent = args.output_dir or args.mask_root / "train" / "dataloader_check"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = parent / f"{args.event}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"Checking training windows for {args.event}. Results: {out_dir}", flush=True)
    report = audit_event(args, out_dir)
    summary = json.dumps(report, indent=2)
    (out_dir / "report.json").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"Report: {out_dir / 'report.json'}")
    if report["plot"]:
        print(f"Plot: {out_dir / report['plot']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
