"""Case-study maps of uncertainty, P(water) and errors for the water-only invalid-KL ablation arms.

From the project root:
    # Test case studies from the saved full-image predictions (the same predictions as the metrics):
    uv run python -m scripts.eval.plot_val_kl_maps --split test --pred-root artifacts/results/edl_water_kl_ablation
    # Validation maps with on-the-fly inference from the training checkpoints:
    uv run python -m scripts.eval.plot_val_kl_maps --split val --checkpoint best

Window selection (from GT and KL masks only; frozen in docs/water_only_kl_review_checklist.md, item 13 B): for each
event and rule, the 256-pixel grid window with the largest positive count. Rules: kl (KL pixels), water_edge (water
within 3 px of KL, computed on the whole event), clear_control (water count in a window with no KL and no cloud GT),
clear_kl (KL pixels whose cloud GT is clear). Each rule's top events are featured; every selected window goes to the
contact sheet. With --pred-root, windows with >= 500 near-KL water pixels are also searched for the largest and
smallest (C - kl_off) near-KL water recall, labelled outcome-selected.
"""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rasterio.windows import Window
from scipy.ndimage import binary_dilation, binary_erosion

from flood_uncertainty.data.worldfloods_kl import WorldFloodsKLDatasetTiled
from flood_uncertainty.models.edl_water_kl import WaterOnlyInvalidKLEDL
from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS, COLORS_WORLDFLOODS_INVLANDWATER
from ml4floods.models.dataset_setup import get_transformations
from ml4floods.models.utils.configuration import AttrDict
from ml4floods.models.worldfloods_model import METRIC_MODE
from ml4floods.preprocess.tiling import WindowSlices, get_window_tiles
from scripts.analysis.evaluate_kl_ablation import PREDICTION_BANDS, kl_mask_for

ROOT = Path(__file__).resolve().parents[2]
ARMS = (  # (directory, figure label)
    ("kl_off", "kl_off (lambda=0)"),
    ("kl_shared_beta", "A: beta(t)"),
    ("kl_002_beta", "B: 0.02*beta(t)"),
    ("kl_002_const", "C: 0.02"),
)
CONTROL, METHOD = "kl_off", "kl_002_const"
SIZE = 256
HALO = 3  # "Near KL" = within 3 pixels (Chebyshev) of a KL pixel, computed on the whole event.
RULES = ("kl", "water_edge", "clear_control", "clear_kl")
OUTCOME_MIN_NEAR_WATER = 500
KL_COLOR, WATER_EDGE_COLOR = "#ff2591", "#00e5ff"
# Error map classes: 0 ignored, 1 TN, 2 TP, 3 FP, 4 FN (correct pixels muted).
ERROR_COLORS = ("#8c8c8c", "#f2f2f2", "#c6dbef", "#2171b5", "#fd8d3c")
ERROR_LABELS = ("ignored", "TN", "TP", "FP", "FN")


def latest_run(ablation_root: Path, arm: str) -> Path:
    runs = sorted(p for p in (ablation_root / arm).iterdir() if (p / "train_manifest.json").is_file())
    if not runs:
        raise FileNotFoundError(f"No run with a manifest under {ablation_root / arm}")
    return runs[-1]


def best_epoch(run: Path, manifest: dict) -> int:
    """Epoch with the best monitored validation metric, as ModelCheckpoint(best) selects it."""
    metric = manifest["resolved_config"]["model_params"]["hyperparameters"]["metric_monitor"]
    with (run / "logs" / "version_0" / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get(metric)]
    if not rows:
        raise ValueError(f"No {metric} values in {run}")
    pick = min if METRIC_MODE[metric] == "min" else max
    return int(pick(rows, key=lambda row: float(row[metric]))["epoch"])


def checkpoint_path(run: Path, manifest: dict, choice: str) -> Path:
    if choice == "last":
        return run / "checkpoint" / "last.ckpt"
    # Per-epoch weight files are written once, so they are safe to read while an arm still trains.
    epoch = best_epoch(run, manifest) if choice == "best" else int(choice)
    matches = sorted((run / "checkpoint" / "epochs").glob(f"epoch={epoch}-step=*.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one epoch {epoch} checkpoint in {run / 'checkpoint/epochs'}, found {len(matches)}")
    return matches[0]


def check_same_inputs(manifests: dict) -> None:
    """All arms must share the split file and data settings, since inputs are prepared once."""
    reference = manifests[CONTROL]
    for arm, manifest in manifests.items():
        if (manifest["split_file_sha256"] != reference["split_file_sha256"]
                or manifest["resolved_config"]["data_params"] != reference["resolved_config"]["data_params"]):
            raise ValueError(f"{arm} uses a different split file or data settings than {CONTROL}")


def event_arrays(split, image_file, gt_file, mask_root: Path, data_root: Path) -> dict:
    """Full-event GT, KL, KL halo and boundaries after grid checks. Events without an AOI get an empty KL mask."""
    with rasterio.open(image_file) as image, rasterio.open(gt_file) as gt:
        if (image.shape, image.crs, image.transform) != (gt.shape, gt.crs, gt.transform):
            raise ValueError(f"S2 and GT grids differ for {Path(gt_file).stem}")
        cloud, water = gt.read(1), gt.read(2)
        kl = kl_mask_for(split, image_file, gt, mask_root, data_root)
        grid = (gt.shape, gt.crs, gt.transform)
    has_aoi = kl is not None
    kl = kl if has_aoi else np.zeros(water.shape, dtype=bool)
    is_water = water == 2
    return dict(
        water=water, cloud=cloud, kl=kl, has_aoi=has_aoi, grid=grid,
        inside=np.ones(water.shape, dtype=bool),  # Cropped with zero padding: marks the in-bounds footprint.
        halo=binary_dilation(kl, structure=np.ones((2 * HALO + 1, 2 * HALO + 1), dtype=bool)) & ~kl,
        kl_edge=kl & ~binary_erosion(kl), water_edge=is_water & ~binary_erosion(is_water),
    )


def crop(array, window: Window):
    """Crop a full-event array to a grid window, zero-padded past the raster edge like boundless reads."""
    out = np.zeros((SIZE, SIZE), dtype=array.dtype)
    rows = slice(window.row_off, min(window.row_off + SIZE, array.shape[0]))
    cols = slice(window.col_off, min(window.col_off + SIZE, array.shape[1]))
    out[:rows.stop - rows.start, :cols.stop - cols.start] = array[rows, cols]
    return out


def grid_windows(gt_file: str) -> list[Window]:
    with rasterio.open(gt_file) as gt:
        return get_window_tiles(gt, height=SIZE, width=SIZE)


def rule_counts(arrays, window):
    water, cloud, kl, halo, inside = (crop(arrays[key], window) for key in ("water", "cloud", "kl", "halo", "inside"))
    is_water = water == 2
    # Clear control: no KL and an affirmative clear cloud label on every in-bounds pixel (no cloud, no unknown).
    clear_window = not kl.any() and bool((cloud[inside] == 1).all())
    return dict(
        kl=int(kl.sum()), water_edge=int((halo & is_water).sum()),
        clear_control=int(is_water.sum()) if clear_window else 0, clear_kl=int((kl & (cloud == 1)).sum()),
    )


def select_windows(pairs, split, mask_root, data_root, featured_per_rule):
    """One window per event and rule; deduplicated selections plus the featured windows of each rule."""
    best = {}
    for image_file, gt_file in pairs:
        event = Path(image_file).stem
        arrays = event_arrays(split, image_file, gt_file, mask_root, data_root)
        for window in grid_windows(gt_file):
            counts = rule_counts(arrays, window)
            for rule in RULES:
                key = (rule, event)
                rank = (-counts[rule], window.row_off, window.col_off)
                if counts[rule] > 0 and (key not in best or rank < best[key][0]):
                    best[key] = (rank, dict(image=image_file, gt=gt_file, event=event, window=window,
                                            count=counts[rule]))
    selections = {}
    for (rule, event), (_, entry) in sorted(best.items()):
        key = (event, entry["window"].col_off, entry["window"].row_off)
        selections.setdefault(key, {**{k: v for k, v in entry.items() if k != "count"}, "rules": {}})
        selections[key]["rules"][rule] = entry["count"]
    featured = []
    for rule in RULES:
        ranked = sorted(((entry["count"], event, entry) for (r, event), (_, entry) in best.items() if r == rule),
                        key=lambda item: (-item[0], item[1], item[2]["window"].row_off, item[2]["window"].col_off))
        for count, event, entry in ranked[:featured_per_rule]:
            featured.append({**entry, "reason": rule})
        print(f"Rule {rule}: {len(ranked)} eligible events, featured {min(len(ranked), featured_per_rule)}", flush=True)
    return list(selections.values()), featured


def merge_featured(featured):
    """One figure per window: a window chosen by several rules (or the outcome search) keeps every reason."""
    merged = {}
    for entry in featured:
        key = (entry["event"], entry["window"].col_off, entry["window"].row_off)
        if key in merged:
            merged[key]["reason"] += "+" + entry["reason"]
            continue
        merged[key] = dict(entry)
    return list(merged.values())


def manual_patches(pairs, requests):
    by_event = {Path(image).stem: (image, gt) for image, gt in pairs}
    patches = []
    for event, col, row in requests:
        if event not in by_event:
            raise ValueError(f"{event} is not in the selected split")
        image, gt = by_event[event]
        window = Window(int(col), int(row), SIZE, SIZE)
        if window not in grid_windows(gt):
            raise ValueError(f"col={col}, row={row} is not on {event}'s {SIZE}-pixel grid")
        patches.append(dict(image=image, gt=gt, event=event, window=window, reason="manual"))
    return patches


def saved_prediction_path(pred_root, checkpoint, arm, split, event):
    return pred_root / checkpoint / arm / "predictions" / split / "EDL" / f"{event}_output_EDL.tif"


def read_saved_window(path, window, grid):
    """u, P and prediction validity for one window of a saved prediction (outside the raster is invalid)."""
    with rasterio.open(path) as src:
        if (src.shape, src.crs, src.transform) != grid:
            raise ValueError(f"Prediction grid differs from GT: {path}")
        if tuple(src.descriptions[1:3]) != PREDICTION_BANDS[:2]:
            raise ValueError(f"Unexpected prediction bands {src.descriptions}: {path}")
        cls, u, p = (src.read(band, window=window, boundless=True, fill_value=0) for band in (1, 2, 3))
    valid = (cls != 0) & np.isfinite(u) & np.isfinite(p) & (u >= 0) & (u <= 1) & (p >= 0) & (p <= 1)
    return u, p, valid


def saved_meta(pred_root, checkpoint, arm):
    """Provenance of saved predictions: the checkpoint recorded by run_kl_ablation_eval.sh."""
    ckpt = (pred_root / checkpoint / arm / "checkpoint.txt").read_text().strip()
    parts = Path(ckpt).parts  # .../<arm>/<run stamp>/checkpoint/[epochs/]<file>
    epoch = re.search(r"epoch=(\d+)", parts[-1])
    return dict(run_stamp=parts[parts.index("checkpoint") - 1] if "checkpoint" in parts else "", checkpoint=ckpt,
                ckpt_epoch=int(epoch.group(1)) if epoch else Path(ckpt).stem, ckpt_step="")


def outcome_windows(pairs, split, mask_root, data_root, pred_root, checkpoint):
    """Largest and smallest (C - kl_off) near-KL water recall among windows with enough near-KL water."""
    found = []
    for image_file, gt_file in pairs:
        event = Path(image_file).stem
        arrays = event_arrays(split, image_file, gt_file, mask_root, data_root)
        if not arrays["has_aoi"]:
            continue
        for window in grid_windows(gt_file):
            water, halo = crop(arrays["water"], window), crop(arrays["halo"], window)
            if np.count_nonzero(halo & (water == 2)) < OUTCOME_MIN_NEAR_WATER:
                continue
            recall = {}
            valid_all = np.ones((SIZE, SIZE), dtype=bool)
            preds = {}
            for arm, _ in ARMS:  # Same support as the evaluator and the figures: valid in every arm.
                _, p, valid = read_saved_window(saved_prediction_path(pred_root, checkpoint, arm, split, event),
                                                window, arrays["grid"])
                preds[arm], valid_all = p, valid_all & valid
            near_water = halo & (water == 2) & valid_all
            if np.count_nonzero(near_water) < OUTCOME_MIN_NEAR_WATER:
                continue
            for arm in (CONTROL, METHOD):
                recall[arm] = np.count_nonzero(near_water & (preds[arm] > 0.5)) / np.count_nonzero(near_water)
            found.append((recall[METHOD] - recall[CONTROL], event, window, image_file, gt_file))
    if not found:
        return []
    found.sort(key=lambda item: (item[0], item[1], item[2].row_off, item[2].col_off))
    print(f"Outcome search: {len(found)} windows with >= {OUTCOME_MIN_NEAR_WATER} near-KL water pixels", flush=True)
    picks = [("outcome_worst", found[0]), ("outcome_best", found[-1])]
    return [dict(image=image, gt=gt, event=event, window=window, reason=reason, recall_diff=diff)
            for reason, (diff, event, window, image, gt) in picks]


def load_patch_arrays(patches, split, mask_root, data_root):
    cache = {}
    for patch in patches:
        if patch["event"] not in cache:
            cache = {patch["event"]: event_arrays(split, patch["image"], patch["gt"], mask_root, data_root)}
        arrays = cache[patch["event"]]
        for key in ("water", "cloud", "kl", "halo", "kl_edge", "water_edge"):
            patch[key] = crop(arrays[key], patch["window"])
        patch["has_aoi"], patch["grid"] = arrays["has_aoi"], arrays["grid"]
        with rasterio.open(patch["image"]) as image:  # B4, B3, B2 in raw reflectance
            rgb = image.read([4, 3, 2], window=patch["window"], boundless=True, fill_value=0).astype(np.float32)
        patch["rgb"] = np.clip(np.moveaxis(np.nan_to_num(rgb), 0, -1) / 3500.0, 0, 1)


def predict_saved(patches, pred_root, checkpoint, split):
    results = {id(patch): {} for patch in patches}
    for arm, _ in ARMS:
        meta = saved_meta(pred_root, checkpoint, arm)
        for patch in patches:
            u, p, valid = read_saved_window(saved_prediction_path(pred_root, checkpoint, arm, split, patch["event"]),
                                            patch["window"], patch["grid"])
            results[id(patch)][arm] = dict(u=u, p=p, valid=valid, meta=meta)
    return results


def predict_on_the_fly(patches, data, ablation_root, checkpoint, mask_root, device):
    runs = {arm: latest_run(ablation_root, arm) for arm, _ in ARMS}
    manifests = {arm: json.loads((run / "train_manifest.json").read_text()) for arm, run in runs.items()}
    check_same_inputs(manifests)
    _, val_transform = get_transformations(data)
    dataset = WorldFloodsKLDatasetTiled(
        [WindowSlices(p["image"], p["window"]) for p in patches], kl_mask_root=mask_root,
        image_prefix=f"/{data.input_folder}/", gt_prefix=f"/{data.target_folder}/",
        bands=CHANNELS_CONFIGURATIONS[data.channel_configuration], transforms=val_transform,
        events_without_kl={p["event"] for p in patches if not p["has_aoi"]},
    )
    images = torch.stack([dataset[index]["image"] for index in range(len(patches))])  # deterministic transform
    results = {id(patch): {} for patch in patches}
    for arm, _ in ARMS:  # One model in memory at a time.
        ckpt = checkpoint_path(runs[arm], manifests[arm], checkpoint)
        config = AttrDict.from_nested_dicts(manifests[arm]["resolved_config"])
        model = WaterOnlyInvalidKLEDL(config.model_params, normalized_data=config.data_params.test_transformation.normalize)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
        model.to(device).eval()
        u_all, p_all = [], []
        with torch.inference_mode():
            for start in range(0, len(images), 8):  # Small batches keep CPU activation memory bounded.
                output = model.edl_logits_to_output(model(images[start:start + 8].to(device)).float())
                u_all.append(output["dst_u"][:, 0].cpu().numpy())
                p_all.append(output["prob"][:, 0].cpu().numpy())
        u_all, p_all = np.concatenate(u_all), np.concatenate(p_all)
        meta = dict(run_stamp=runs[arm].name, checkpoint=str(ckpt), ckpt_epoch=state.get("epoch"),
                    ckpt_step=state.get("global_step"))
        for index, patch in enumerate(patches):
            u, p = u_all[index], p_all[index]
            results[id(patch)][arm] = dict(u=u, p=p, valid=np.ones(u.shape, dtype=bool), meta=meta)
    return results


def mean_or_na(values, region):
    return float(values[region].mean()) if region.any() else "NA"


def ratio_or_na(numerator, denominator):
    return numerator / denominator if denominator else "NA"


def fmt(value):
    return value if value == "NA" else f"{value:.3f}"


def patch_stats(patch, arm_result, scored, valid_all):
    water, kl, halo = patch["water"], patch["kl"], patch["halo"]
    u, p = arm_result["u"], arm_result["p"]
    pred, truth = p > 0.5, water == 2
    tp, fp, fn = (np.count_nonzero(scored & mask) for mask in (pred & truth, pred & ~truth, ~pred & truth))
    kl_scored = kl & valid_all  # Same KL support as the evaluator: valid in every arm.
    stats = dict(kl_pixels=int(kl_scored.sum()), scored_pixels=int(scored.sum()), u_kl=mean_or_na(u, kl_scored),
                 water_iou=ratio_or_na(tp, tp + fp + fn))
    if not patch["has_aoi"]:  # Without an AOI, near/far are undefined rather than "all far".
        return {**stats, "near_scored_pixels": "NA", "far_scored_pixels": "NA", "u_near": "NA", "u_far": "NA",
                "near_water_pixels": "NA", "near_water_recall": "NA"}
    near, far = scored & halo, scored & ~halo
    near_water = near & truth
    return {**stats, "near_scored_pixels": int(near.sum()), "far_scored_pixels": int(far.sum()),
            "u_near": mean_or_na(u, near), "u_far": mean_or_na(u, far), "near_water_pixels": int(near_water.sum()),
            "near_water_recall": ratio_or_na(np.count_nonzero(near_water & pred), int(near_water.sum()))}


def error_map(scored, p, water):
    pred, truth = p > 0.5, water == 2
    classes = np.zeros(water.shape, dtype=np.uint8)
    classes[scored & ~pred & ~truth] = 1
    classes[scored & pred & truth] = 2
    classes[scored & pred & ~truth] = 3
    classes[scored & ~pred & truth] = 4
    return classes


def overlay(axis, mask, color, alpha=1.0):
    axis.imshow(np.ma.masked_equal(mask, False), cmap=ListedColormap([color]), vmin=0, vmax=1, alpha=alpha,
                interpolation="nearest")


def show_u(axis, u, visible, kl_edge):
    cmap = colormaps["viridis"].with_extremes(bad="#808080")  # gray = not scored and not KL
    artist = axis.imshow(np.ma.masked_where(~visible, u), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    overlay(axis, kl_edge, KL_COLOR)
    return artist


def show_errors(axis, classes, kl_edge):
    axis.imshow(classes, cmap=ListedColormap(ERROR_COLORS), vmin=-0.5, vmax=4.5, interpolation="nearest")
    overlay(axis, kl_edge, KL_COLOR)


def plot_featured(path, patch, results, scored, stats, heading, figsize=(16, 16.5), dpi=100):
    """4x4 figure: inputs, then u, P(water) and errors per arm. `heading` is the first title line."""
    water, kl = patch["water"], patch["kl"]
    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 4)
    axes[0, 0].imshow(patch["rgb"], interpolation="nearest")
    axes[0, 0].set_title("Sentinel-2 RGB")
    axes[0, 1].imshow(water, cmap=ListedColormap(COLORS_WORLDFLOODS_INVLANDWATER), vmin=-0.5, vmax=2.5,
                      interpolation="nearest")
    axes[0, 1].set_title("Water GT (black invalid, brown land, blue water)")
    axes[0, 2].imshow(kl, cmap=ListedColormap(["#161616", KL_COLOR]), vmin=0, vmax=1, interpolation="nearest")
    axes[0, 2].set_title(f"KL mask ({int(kl.sum()):,} px)")
    axes[0, 3].imshow(patch["rgb"], interpolation="nearest")
    overlay(axes[0, 3], kl, KL_COLOR, alpha=0.5)
    axes[0, 3].set_title("RGB + KL mask")
    cmap = colormaps["viridis"].with_extremes(bad="#808080")
    for column, (arm, arm_label) in enumerate(ARMS):
        result, s = results[arm], stats[arm]
        visible = scored | (kl & result["valid"])
        u_artist = show_u(axes[1, column], result["u"], visible, patch["kl_edge"])
        axes[1, column].set_title(f"{arm_label} | epoch {result['meta']['ckpt_epoch']}\n"
                                  f"u: KL {fmt(s['u_kl'])}, near {fmt(s['u_near'])}, far {fmt(s['u_far'])}")
        p_artist = axes[2, column].imshow(np.ma.masked_where(~visible, result["p"]), cmap=cmap, vmin=0, vmax=1,
                                          interpolation="nearest")
        overlay(axes[2, column], patch["water_edge"], WATER_EDGE_COLOR)
        axes[2, column].set_title(f"{arm_label}\nP(water) | water IoU {fmt(s['water_iou'])}")
        show_errors(axes[3, column], error_map(scored, result["p"], water), patch["kl_edge"])
        count = s["near_water_pixels"]
        axes[3, column].set_title(f"{arm_label} errors\nnear-KL water recall {fmt(s['near_water_recall'])} "
                                  f"(n={count if count == 'NA' else f'{count:,}'})")
    figure.colorbar(u_artist, ax=list(axes[1]), shrink=0.8, label="DST uncertainty u = 2/S")
    figure.colorbar(p_artist, ax=list(axes[2]), shrink=0.8, label="P(water)")
    figure.legend(handles=[Patch(color=c, label=l) for c, l in zip(ERROR_COLORS, ERROR_LABELS)],
                  loc="lower center", ncol=5, frameon=False)
    for axis in axes.flat:
        axis.set_axis_off()
    figure.suptitle(f"{heading}\n"
                    f"pink = KL boundary; cyan = water GT boundary; gray = not scored; "
                    f"near = scored pixels within {HALO} px of KL")
    figure.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))  # Leave room for the bottom legend.
    figure.savefig(path, dpi=dpi)
    figure.clear()


def plot_contact_sheets(out_dir, patches, results, scored_masks, rows_per_page):
    for page, start in enumerate(range(0, len(patches), rows_per_page), start=1):
        chunk = patches[start:start + rows_per_page]
        figure = Figure(figsize=(13, 3.3 * len(chunk)), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(chunk), 4, squeeze=False)
        for row, patch in enumerate(chunk):
            scored = scored_masks[id(patch)]
            control, method = results[id(patch)][CONTROL], results[id(patch)][METHOD]
            axes[row, 0].imshow(patch["rgb"], interpolation="nearest")
            overlay(axes[row, 0], patch["kl"], KL_COLOR, alpha=0.5)
            for column, result in ((1, control), (2, method)):
                show_u(axes[row, column], result["u"], scored | (patch["kl"] & result["valid"]), patch["kl_edge"])
            show_errors(axes[row, 3], error_map(scored, method["p"], patch["water"]), patch["kl_edge"])
            window = patch["window"]
            axes[row, 0].set_title(f"{patch['event']} c{window.col_off} r{window.row_off}\n"
                                   f"{', '.join(patch['rules'])}", fontsize=8)
            for column, title in ((1, "u kl_off"), (2, "u C"), (3, "errors C")):
                axes[row, column].set_title(title, fontsize=8)
            for axis in axes[row]:
                axis.set_axis_off()
        figure.savefig(out_dir / f"contact_sheet_p{page:02d}.png", dpi=80)
        figure.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", default="best",
                        help="best, last, or an epoch number; with --pred-root, the evaluation subdirectory")
    parser.add_argument("--pred-root", type=Path,
                        help="Use saved predictions under <pred-root>/<checkpoint>/<arm>/predictions/<split>/EDL")
    parser.add_argument("--ablation-root", type=Path, default=ROOT / "artifacts/models/edl_water_kl_ablation")
    parser.add_argument("--mask-root", type=Path, default=ROOT / "artifacts/kl_masks/worldfloods_v2")
    parser.add_argument("--featured-per-rule", type=int, default=2)
    parser.add_argument("--contact-rows", type=int, default=6, help="Windows per contact-sheet page")
    parser.add_argument("--window", nargs=3, action="append", metavar=("EVENT", "COL", "ROW"),
                        help="Plot only this grid window instead of the rule-based selection (repeatable)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/eval/edl_water_kl_ablation/case_studies")
    args = parser.parse_args()
    if args.pred_root is None and args.checkpoint not in ("best", "last") and not args.checkpoint.isdigit():
        parser.error("--checkpoint must be best, last, or an epoch number")
    if min(args.featured_per_rule, args.contact_rows) < 1:
        parser.error("--featured-per-rule and --contact-rows must be at least 1")
    torch.set_num_threads(args.threads)

    data = load_mode_config(str(ROOT / f"configurations/edl_water_kl_ablation/{CONTROL}.json"), "train").data_params
    data_root = Path(data.path_to_splits)
    split = json.loads(Path(data.train_test_split_file).read_text())[args.split]
    pairs = sorted(zip(split[data.input_folder], split[data.target_folder], strict=True))

    if args.window:
        featured, contact = manual_patches(pairs, args.window), []
    else:
        selections, featured = select_windows(pairs, args.split, args.mask_root, data_root, args.featured_per_rule)
        contact = sorted(selections, key=lambda s: (s["event"], s["window"].row_off, s["window"].col_off))
        if args.pred_root is not None:
            featured += outcome_windows(pairs, args.split, args.mask_root, data_root, args.pred_root, args.checkpoint)
        featured = merge_featured(featured)
    patches = featured + contact
    if not patches:
        raise ValueError("No window qualifies under the selection rules")
    load_patch_arrays(patches, args.split, args.mask_root, data_root)
    results = (predict_saved(patches, args.pred_root, args.checkpoint, args.split) if args.pred_root is not None
               else predict_on_the_fly(patches, data, args.ablation_root, args.checkpoint, args.mask_root, args.device))

    source = "saved" if args.pred_root is not None else "on-the-fly"
    label = f"{args.split} | {args.checkpoint} | {source} predictions"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / f"{args.split}_{args.checkpoint}_{source}_{stamp}"  # fresh: no stale figures
    out_dir.mkdir(parents=True, exist_ok=False)
    scored_masks, rows = {}, []
    for patch in patches:
        labelled = (patch["water"] == 1) | (patch["water"] == 2)
        valid_all = np.logical_and.reduce([results[id(patch)][arm]["valid"] for arm, _ in ARMS])
        scored = labelled & valid_all
        scored_masks[id(patch)] = scored
        stats = {arm: patch_stats(patch, results[id(patch)][arm], scored, valid_all) for arm, _ in ARMS}
        patch["stats"] = stats
        window = patch["window"]
        for arm, _ in ARMS:
            rows.append(dict(event=patch["event"], col=window.col_off, row=window.row_off,
                             role="featured" if "reason" in patch else "contact",
                             selected_by=patch.get("reason", "+".join(patch.get("rules", {}))),
                             rule_counts=json.dumps(patch.get("rules", {})), arm=arm,
                             **results[id(patch)][arm]["meta"], **stats[arm]))
    for patch in featured:
        window = patch["window"]
        path = out_dir / f"{patch['reason']}_{patch['event']}_c{window.col_off}_r{window.row_off}.png"
        heading = (f"{patch['event']} | col={window.col_off}, row={window.row_off} | {label} | "
                   f"selected by {patch['reason']}")
        plot_featured(path, patch, results[id(patch)], scored_masks[id(patch)], patch["stats"], heading)
        print(path, flush=True)
    if contact:
        plot_contact_sheets(out_dir, contact, results, scored_masks, args.contact_rows)
    with (out_dir / "patch_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
