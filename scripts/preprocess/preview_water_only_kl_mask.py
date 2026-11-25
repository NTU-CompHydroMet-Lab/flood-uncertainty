"""Audit one saved KL mask and plot an overview plus a native-resolution crop.

Run from the project root with: python -m scripts.preprocess.preview_water_only_kl_mask
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window

from scripts.preprocess.generate_water_only_kl_masks import (
    EventFiles,
    collect_events,
    iter_windows,
    load_aoi,
    read_input_valid,
    validate_rasters,
)


GT_COLORS = ("#343a40", "#b99a72", "#2998df")
KL_COLOR = "#ff006e"
AOI_COLOR = "#ffe066"


def first_pixel(values: np.ndarray, window: Window):
    positions = np.flatnonzero(values)
    if positions.size == 0:
        return None
    row, col = np.unravel_index(positions[0], values.shape)
    return [int(window.col_off + col), int(window.row_off + row)]


def audit_saved_mask(event: EventFiles, block_size: int = 1024) -> dict:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    with rasterio.open(event.image) as image, rasterio.open(event.gt) as gt, rasterio.open(event.output) as mask:
        validate_rasters(image, gt)
        if mask.count != 1 or mask.dtypes != ("uint8",) or mask.nodata is not None:
            raise ValueError("KL mask must be single-band uint8 with no nodata value")
        if mask.crs != gt.crs or mask.transform != gt.transform or mask.shape != gt.shape:
            raise ValueError("KL mask/GT grid mismatch: CRS, transform and dimensions must match")
        geometries = load_aoi(event.floodmap, gt.crs)
        report = {
            "event_id": event.event_id,
            "split": event.split,
            "grid_matches": True,
            "crs": str(gt.crs),
            "transform": list(gt.transform)[:6],
            "shape": [gt.height, gt.width],
            "total_pixels": gt.height * gt.width,
            "aoi_pixels": 0,
            "kl_pixels": 0,
            "expected_kl_pixels": 0,
            "mismatch_pixels": 0,
            "kl_outside_aoi_pixels": 0,
            "kl_invalid_input_pixels": 0,
            "kl_valid_label_pixels": 0,
        }
        first_mismatch = first_edge = first_selected = None
        for window in iter_windows(gt.width, gt.height, block_size):
            water = gt.read(2, window=window, masked=False)
            values = mask.read(1, window=window, masked=False)
            if not np.isin(water, (0, 1, 2)).all() or not np.isin(values, (0, 1)).all():
                raise ValueError(f"Unexpected GT or KL mask values at {window}")
            aoi = rasterize(
                geometries,
                out_shape=water.shape,
                transform=gt.window_transform(window),
                default_value=1,
                fill=0,
                all_touched=True,
                dtype="uint8",
                skip_invalid=False,
            ).astype(bool)
            valid = read_input_valid(image, window)
            selected = values == 1
            expected = valid & aoi & (water == 0)
            mismatch = selected != expected
            report["aoi_pixels"] += int(aoi.sum())
            report["kl_pixels"] += int(selected.sum())
            report["expected_kl_pixels"] += int(expected.sum())
            report["mismatch_pixels"] += int(mismatch.sum())
            report["kl_outside_aoi_pixels"] += int((selected & ~aoi).sum())
            report["kl_invalid_input_pixels"] += int((selected & ~valid).sum())
            report["kl_valid_label_pixels"] += int((selected & (water != 0)).sum())
            if first_mismatch is None:
                first_mismatch = first_pixel(mismatch, window)
            if first_selected is None:
                first_selected = first_pixel(selected, window)
            if first_edge is None:
                edge = np.zeros_like(selected)
                edge[:, 1:] |= selected[:, 1:] != selected[:, :-1]
                edge[1:, :] |= selected[1:, :] != selected[:-1, :]
                first_edge = first_pixel(edge, window)
        if report["aoi_pixels"] == 0:
            raise ValueError("AOI does not cover any GT pixels")
        report["rule_matches"] = report["mismatch_pixels"] == 0
        report["detail_center_col_row"] = first_mismatch or first_edge or first_selected or [gt.width // 2, gt.height // 2]
        return report


def detail_window(width: int, height: int, center: list[int], size: int) -> Window:
    crop_width, crop_height = min(width, size), min(height, size)
    col = min(max(center[0] - crop_width // 2, 0), width - crop_width)
    row = min(max(center[1] - crop_height // 2, 0), height - crop_height)
    return Window(col, row, crop_width, crop_height)


def aoi_pixel_segments(geometries, transform) -> list[np.ndarray]:
    inverse = ~transform
    segments = []
    for geometry in geometries:
        polygons = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
        for polygon in polygons:
            for ring in (polygon.exterior, *polygon.interiors):
                xy = np.asarray(ring.coords)
                cols = inverse.a * xy[:, 0] + inverse.b * xy[:, 1] + inverse.c
                rows = inverse.d * xy[:, 0] + inverse.e * xy[:, 1] + inverse.f
                segments.append(np.column_stack([cols, rows]))
    return segments


def draw_view(image, gt, mask, segments, window, report, path, max_side, rgb_max, detail=None):
    scale = min(1.0, max_side / max(window.height, window.width))
    height = max(1, math.ceil(window.height * scale))
    width = max(1, math.ceil(window.width * scale))
    # Use the same nearest-neighbor sampling grid for all panels. The detail
    # crop has scale=1, so its arrays retain the original raster resolution.
    rgb = image.read(
        [4, 3, 2], window=window, out_shape=(3, height, width),
        resampling=Resampling.nearest, masked=True,
    ).astype(np.float32).filled(0)
    rgb = np.nan_to_num(rgb, nan=0, posinf=0, neginf=0)
    rgb = np.moveaxis(np.clip(rgb / rgb_max, 0, 1), 0, -1)
    water = gt.read(2, window=window, out_shape=(height, width), resampling=Resampling.nearest, masked=False)
    selected = mask.read(1, window=window, out_shape=(height, width), resampling=Resampling.nearest, masked=False)
    extent = (window.col_off, window.col_off + window.width, window.row_off + window.height, window.row_off)
    kwargs = dict(extent=extent, origin="upper", interpolation="nearest")
    figure = Figure(figsize=(13, 10), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True, sharey=True).ravel()
    axes[0].imshow(rgb, **kwargs)
    axes[1].imshow(water, cmap=ListedColormap(GT_COLORS), vmin=0, vmax=2, **kwargs)
    axes[2].imshow(selected, cmap=ListedColormap(["#15191e", KL_COLOR]), vmin=0, vmax=1, **kwargs)
    axes[3].imshow(rgb, **kwargs)
    overlay = np.zeros((*selected.shape, 4), dtype=np.float32)
    overlay[..., :3] = (1, 0, 110 / 255)
    overlay[..., 3] = (selected == 1) * 0.5
    axes[3].imshow(overlay, **kwargs)
    titles = ("S2 RGB + AOI", "Water GT", "Saved KL mask", "S2 + KL mask + AOI")
    for axis, title in zip(axes, titles):
        axis.add_collection(LineCollection(segments, colors=AOI_COLOR, linewidths=0.9))
        if detail is not None:
            axis.add_patch(Rectangle(
                (detail.col_off, detail.row_off), detail.width, detail.height,
                fill=False, edgecolor="white", linewidth=1.2, linestyle="--",
            ))
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("Source pixel column")
        axis.set_ylabel("Source pixel row")
    kind = "overview (nearest-neighbor downsample)" if scale < 1 else "native raster resolution"
    status = "PASS" if report["rule_matches"] else "FAIL"
    figure.suptitle(
        f"{report['event_id']} | {kind}\n"
        f"Grid: MATCH | Rule: {status} | KL: {report['kl_pixels']:,} pixels | "
        f"Mismatches: {report['mismatch_pixels']:,}",
        fontsize=13,
    )
    handles = [
        Patch(facecolor=color, label=label)
        for color, label in zip(GT_COLORS, ("GT invalid (0)", "GT land (1)", "GT water (2)"))
    ]
    handles.extend([
        Patch(facecolor=KL_COLOR, label="KL selected (1)"),
        Line2D([], [], color=AOI_COLOR, label="Annotated AOI boundary"),
    ])
    if detail is not None:
        handles.append(Line2D([], [], color="gray", linestyle="--", label="Detail crop"))
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    figure.savefig(path, dpi=160)
    figure.clear()


def create_preview(event: EventFiles, output_dir: Path, max_side=1200, zoom_size=512, rgb_max=3500.0, window=None):
    if min(max_side, zoom_size) <= 0 or not np.isfinite(rgb_max) or rgb_max <= 0:
        raise ValueError("Preview dimensions and rgb_max must be positive and finite")
    report = audit_saved_mask(event)
    with rasterio.open(event.image) as image, rasterio.open(event.gt) as gt, rasterio.open(event.output) as mask:
        crop = detail_window(gt.width, gt.height, report["detail_center_col_row"], zoom_size)
        if window is not None:
            crop = Window(*window)
        if (
            min(crop.width, crop.height) <= 0 or min(crop.col_off, crop.row_off) < 0
            or crop.col_off + crop.width > gt.width or crop.row_off + crop.height > gt.height
        ):
            raise ValueError("Detail window must be positive and entirely inside the source raster")
        segments = aoi_pixel_segments(load_aoi(event.floodmap, gt.crs), gt.transform)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {name: output_dir / f"{event.event_id}_{name}.png" for name in ("overview", "detail")}
        draw_view(image, gt, mask, segments, Window(0, 0, gt.width, gt.height), report,
                  paths["overview"], max_side, rgb_max, detail=crop)
        draw_view(image, gt, mask, segments, crop, report, paths["detail"],
                  int(max(crop.width, crop.height)), rgb_max)
    report["detail_window_col_row_width_height"] = [int(crop.col_off), int(crop.row_off), int(crop.width), int(crop.height)]
    report["source_s2"] = str(event.image.resolve())
    report["source_gt"] = str(event.gt.resolve())
    report["source_mask"] = str(event.output.resolve())
    report_path = output_dir / f"{event.event_id}_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for path in (*paths.values(), report_path):
        print(path, flush=True)
    print(json.dumps({key: value for key, value in report.items() if key.endswith("pixels")}, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--event", required=True, help="Event stem, without .tif")
    parser.add_argument("--output-dir", type=Path, help="Default: <mask-root>/<split>/preview")
    parser.add_argument("--max-side", type=int, default=1200, help="Longest overview array side")
    parser.add_argument("--zoom-size", type=int, default=512, help="Native-resolution detail crop size")
    parser.add_argument("--rgb-max", type=float, default=3500.0, help="Shared RGB display scaling for both views")
    parser.add_argument("--window", type=int, nargs=4, metavar=("COL", "ROW", "WIDTH", "HEIGHT"))
    args = parser.parse_args(argv)
    event = collect_events(args.data_root, args.mask_root, [args.split], [args.event])[0]
    output_dir = args.output_dir or args.mask_root / args.split / "preview"
    report = create_preview(event, output_dir, args.max_side, args.zoom_size, args.rgb_max, args.window)
    if not report["rule_matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
