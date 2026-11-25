"""Whole-event maps of uncertainty, P(water) and errors for the water-only invalid-KL ablation arms.

From the project root, after scripts/analysis/run_kl_ablation_eval.sh:
    uv run python -m scripts.eval.plot_kl_event_maps --split test
    uv run python -m scripts.eval.plot_kl_event_maps --events EMSR347_07ZOMBA_DEL_v2 --max-side 2000

Same 4x4 layout as the case studies, one figure per event, from the saved full-image predictions (the same ones as
the metrics). Panel titles and event_stats.csv are computed at full resolution on the evaluator's scored pixels
(water GT 1/2 and valid in every arm). Only the display is subsampled: every s-th pixel, s = ceil(longest side /
--max-side), so features thinner than s pixels can disappear from the figure but not from the numbers.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import binary_erosion

from scripts.eval.plot_val_kl_maps import (ARMS, CONTROL, ROOT, event_arrays, patch_stats, plot_featured,
                                           read_saved_window, saved_meta, saved_prediction_path)

PANEL_INCHES = 4.0  # Longest side of one map panel; dpi is set so that it spans --max-side pixels.


def edges(mask):
    return mask & ~binary_erosion(mask)


def full_event(image_file, gt_file, args, data_root, metas):
    """Full-resolution GT, KL and predictions of all arms, plus the per-arm statistics."""
    event = Path(image_file).stem
    arrays = event_arrays(args.split, image_file, gt_file, args.mask_root, data_root)
    height, width = arrays["water"].shape
    results = {}
    for arm, _ in ARMS:
        path = saved_prediction_path(args.pred_root, args.checkpoint, arm, args.split, event)
        u, p, valid = read_saved_window(path, Window(0, 0, width, height), arrays["grid"])
        results[arm] = dict(u=u, p=p, valid=valid, meta=metas[arm])
    valid_all = np.logical_and.reduce([result["valid"] for result in results.values()])
    scored = ((arrays["water"] == 1) | (arrays["water"] == 2)) & valid_all
    stats = {arm: patch_stats(arrays, results[arm], scored, valid_all) for arm, _ in ARMS}
    return arrays, results, scored, stats


def subsample(image_file, arrays, results, scored, step):
    """Every step-th pixel for display; boundaries are recomputed so they stay one displayed pixel wide."""
    with rasterio.open(image_file) as image:  # B4, B3, B2 raw reflectance, same stretch as the case studies
        rgb = image.read([4, 3, 2])[:, ::step, ::step].astype(np.float32)
    kl, water = arrays["kl"][::step, ::step], arrays["water"][::step, ::step]
    shown = dict(rgb=np.clip(np.moveaxis(np.nan_to_num(rgb), 0, -1) / 3500.0, 0, 1), water=water, kl=kl,
                 kl_edge=edges(kl), water_edge=edges(water == 2))
    shown_results = {arm: dict(result, u=result["u"][::step, ::step], p=result["p"][::step, ::step],
                               valid=result["valid"][::step, ::step]) for arm, result in results.items()}
    return shown, shown_results, scored[::step, ::step]


def figure_size(shape):
    """Panels keep the event's aspect ratio; the width leaves room for titles and colorbars."""
    rows, cols = shape
    panel_w = PANEL_INCHES * min(1.0, cols / rows)
    panel_h = PANEL_INCHES * min(1.0, rows / cols)
    return max(4 * panel_w, 12.0) + 1.0, 4 * (panel_h + 0.75) + 0.6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", default="best", help="Subdirectory written by run_kl_ablation_eval.sh")
    parser.add_argument("--pred-root", type=Path, default=ROOT / "artifacts/results/edl_water_kl_ablation")
    parser.add_argument("--mask-root", type=Path, default=ROOT / "artifacts/kl_masks/worldfloods_v2")
    parser.add_argument("--events", nargs="+", help="Plot only these events (default: every event in the split)")
    parser.add_argument("--max-side", type=int, default=800, help="Longest displayed side of a panel, in pixels")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/eval/edl_water_kl_ablation/event_maps")
    args = parser.parse_args()
    if args.max_side < 100:
        parser.error("--max-side must be at least 100")

    config = json.loads((ROOT / f"configurations/edl_water_kl_ablation/{CONTROL}.json").read_text())["shared"]
    data_root = Path(config["data_params"]["path_to_splits"])
    split = json.loads(Path(config["data_params"]["train_test_split_file"]).read_text())[args.split]
    pairs = sorted(zip(split["S2"], split["gt"], strict=True))
    if args.events:
        unknown = set(args.events) - {Path(image).stem for image, _ in pairs}
        if unknown:
            parser.error(f"Not in the {args.split} split: {sorted(unknown)}")
        pairs = [pair for pair in pairs if Path(pair[0]).stem in args.events]
    metas = {arm: saved_meta(args.pred_root, args.checkpoint, arm) for arm, _ in ARMS}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / f"{args.split}_{args.checkpoint}_{stamp}"  # fresh: no stale figures
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for image_file, gt_file in pairs:
        event = Path(image_file).stem
        arrays, results, scored, stats = full_event(image_file, gt_file, args, data_root, metas)
        rows += [dict(event=event, arm=arm, **metas[arm], **stats[arm]) for arm, _ in ARMS]
        step = max(1, math.ceil(max(arrays["water"].shape) / args.max_side))
        shown, shown_results, shown_scored = subsample(image_file, arrays, results, scored, step)
        heading = (f"{event} | whole event {arrays['water'].shape[1]} x {arrays['water'].shape[0]} px, "
                   f"every {step} px shown | {args.split} | {args.checkpoint} | saved predictions")
        path = out_dir / f"{event}.png"
        plot_featured(path, shown, shown_results, shown_scored, stats, heading,
                      figsize=figure_size(shown["water"].shape), dpi=args.max_side / PANEL_INCHES)
        print(path, flush=True)
    with (out_dir / "event_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
