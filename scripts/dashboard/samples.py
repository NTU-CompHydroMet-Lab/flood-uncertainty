"""Per-image sample rendering for the dashboard (needs numpy, rasterio, Pillow: run the server with .venv/bin/python).

Confusion maps come from eval_metrics.py --save_tif (``cm_<image>_output_<MT>.tif``: 0 invalid, 1 TN, 2 FP, 3 FN,
4 TP), so the picture matches the reported metrics. When a cm tif is missing it is rebuilt the same way as
flood_uncertainty.metrics.segmentation.plot_spatial_confusion_matrix: GT band 2 (0 invalid / 1 land / 2 water) vs
water probability band 3 of the prediction tif > 0.5.
"""

import io
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window

# 0 invalid, 1 TN, 2 FP, 3 FN, 4 TP. TP/FP/FN use the first three categorical slots (CVD-safe as a set);
# TN is a light neutral so errors stand out; invalid is muted gray.
CM_COLORS = np.array([[0x89, 0x87, 0x81], [0xf0, 0xef, 0xec], [0xeb, 0x68, 0x34], [0x1b, 0xaf, 0x7a],
                      [0x2a, 0x78, 0xd6]], dtype=np.uint8)
# Model A vs B: invalid / both right / only A right (violet) / only B right (yellow) / both wrong (near-black).
# Violet and yellow are not used in CM_COLORS, so this panel never reads as a TP/FP/FN map.
DIFF_COLORS = np.array([[0x89, 0x87, 0x81], [0xf0, 0xef, 0xec], [0x62, 0x50, 0xd6], [0xed, 0xa1, 0x00],
                        [0x2b, 0x2b, 0x29]], dtype=np.uint8)
# GT panel: invalid / land / water
GT_COLORS = np.array([[0x89, 0x87, 0x81], [0xf0, 0xef, 0xec], [0x2a, 0x78, 0xd6]], dtype=np.uint8)
RGB_BANDS = (4, 3, 2)  # Sentinel-2 B4, B3, B2 (1-based) in the 13-band WorldFloods S2 tif
RGB_SCALE = 3500.0     # same stretch as segmentation.plot_spatial_confusion_matrix


def run_config(result_dir):
    try:
        return json.loads((result_dir / "run_config.json").read_text())
    except (OSError, ValueError):
        return {}


def model_type(result_dir):
    cfg = run_config(result_dir)
    if cfg.get("model_type"):
        return cfg["model_type"]
    for p in result_dir.glob("metrics_*.csv"):
        return p.stem[len("metrics_"):]
    return "EDL"


def _out_shape(height, width, max_px):
    f = max(1, math.ceil(max(height, width) / max_px)) if max_px else 1
    return max(1, height // f), max(1, width // f)


def _read(path, bands, max_px, window=None):
    """Nearest-neighbour decimated read of an optional pixel window (uses COG overviews when present)."""
    with rasterio.open(path) as src:
        win = window or Window(0, 0, src.width, src.height)
        h, w = _out_shape(int(win.height), int(win.width), max_px)
        if isinstance(bands, (list, tuple)):
            return src.read(list(bands), window=win, out_shape=(len(bands), h, w), resampling=Resampling.nearest)
        return src.read(bands, window=win, out_shape=(h, w), resampling=Resampling.nearest)


def labeled_window(result_dir, image, margin=0.04):
    """Pixel window around GT-valid pixels (cm != 0), so small labelled AOIs are not lost in a thumbnail."""
    mt = model_type(result_dir)
    src_path = result_dir / f"cm_{image}_output_{mt}.tif"
    band = 1
    if not src_path.exists():
        src_path, band = gt_file(result_dir, image), 2
        if src_path is None:
            return None
    with rasterio.open(src_path) as src:
        H, W = src.height, src.width
        small = src.read(band, out_shape=_out_shape(H, W, 512), resampling=Resampling.nearest)
    rows, cols = np.nonzero(small)
    if not len(rows):
        return None
    sy, sx = H / small.shape[0], W / small.shape[1]
    r0, r1 = rows.min() * sy, (rows.max() + 1) * sy
    c0, c1 = cols.min() * sx, (cols.max() + 1) * sx
    m = margin * max(r1 - r0, c1 - c0)
    r0, c0 = max(0, int(r0 - m)), max(0, int(c0 - m))
    r1, c1 = min(H, int(math.ceil(r1 + m))), min(W, int(math.ceil(c1 + m)))
    return Window(c0, r0, c1 - c0, r1 - r0)


def cm_map(result_dir, image, max_px, window=None):
    mt = model_type(result_dir)
    cm_path = result_dir / f"cm_{image}_output_{mt}.tif"
    if cm_path.exists():
        return _read(cm_path, 1, max_px, window).astype(np.uint8)
    pred_path = result_dir / f"{image}_output_{mt}.tif"
    gt_path = gt_file(result_dir, image)
    if not pred_path.exists() or gt_path is None:
        raise FileNotFoundError(f"no cm / prediction tif for {image}")
    band = 2 if mt == "v2" else 3  # 1-based: v2 -> band 2, EDL family -> band 3 (Water_Probability)
    prob = _read(pred_path, band, max_px, window)
    gt = _read(gt_path, 2, max_px, window)
    if gt.shape != prob.shape:
        gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(prob.shape[::-1], Image.NEAREST))
    water_pred, water_gt, valid = prob > 0.5, gt == 2, gt != 0
    out = np.zeros(prob.shape, np.uint8)
    out[valid & ~water_gt & ~water_pred] = 1
    out[valid & ~water_gt & water_pred] = 2
    out[valid & water_gt & ~water_pred] = 3
    out[valid & water_gt & water_pred] = 4
    return out


def data_file(result_dir, image, folder):
    cfg = run_config(result_dir)
    root = cfg.get("data_root")
    if not root:
        return None
    subset = result_dir.parent.name
    p = Path(root) / subset / folder / f"{image}.tif"
    return p if p.exists() else None


def gt_file(result_dir, image):
    return data_file(result_dir, image, "gt")


def render(result_dir, image, kind, max_px=900, crop=False):
    """PNG bytes for kind in {rgb, gt, cm}. GT is derived from the cm map so both panels share one grid.
    crop=True cuts every panel to the same window around the labelled pixels."""
    window = labeled_window(result_dir, image) if crop else None
    if kind == "rgb":
        s2 = data_file(result_dir, image, "S2")
        if s2 is None:
            raise FileNotFoundError(f"S2 image not found for {image} (data_root in run_config.json)")
        with rasterio.open(s2) as src:
            bands = RGB_BANDS if src.count >= 4 else tuple(range(1, min(3, src.count) + 1))
        arr = _read(s2, list(bands), max_px, window).astype(np.float32)
        rgb = (np.clip(arr / RGB_SCALE, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
    else:
        cm = cm_map(result_dir, image, max_px, window)
        if kind == "cm":
            rgb = CM_COLORS[np.clip(cm, 0, 4)]
        elif kind == "gt":
            gt = np.where(cm == 0, 0, np.where((cm == 3) | (cm == 4), 2, 1))
            rgb = GT_COLORS[gt]
        else:
            raise ValueError(f"unknown kind {kind}")
    return _png(rgb)


def render_diff(dir_a, dir_b, image, max_px=900, crop=False):
    """Where two models disagree: 0 invalid, 1 both right, 2 only A right, 3 only B right, 4 both wrong.
    Both maps are read on the same window (the GT-valid area is model independent)."""
    window = labeled_window(dir_a, image) if crop else None
    a, b = cm_map(dir_a, image, max_px, window), cm_map(dir_b, image, max_px, window)
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize(a.shape[::-1], Image.NEAREST))
    ok_a, ok_b = (a == 1) | (a == 4), (b == 1) | (b == 4)
    valid = (a != 0) & (b != 0)
    out = np.zeros(a.shape, np.uint8)
    out[valid & ok_a & ok_b] = 1
    out[valid & ok_a & ~ok_b] = 2
    out[valid & ~ok_a & ok_b] = 3
    out[valid & ~ok_a & ~ok_b] = 4
    return _png(DIFF_COLORS[out])


def _png(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=False)
    return buf.getvalue()
