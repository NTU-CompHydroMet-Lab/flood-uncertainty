"""Build WorldFloods v2 KL-only masks without changing source imagery or GT.

Rule: valid raw S2 input AND annotated AOI AND water_GT == 0.
This selection is a research proxy for hard-to-interpret pixels, not a cloud label.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window


# Match the spectral bands used to create WorldFloods v2 GT. Appended QA60 and
# cloud-probability bands are not spectral input and do not determine validity.
S2_BANDS = tuple(range(1, 14))
MASK_RULE = "input_valid AND inside_AOI AND (water_GT == 0)"
MASK_VERSION = "1"
FULL_INPUT_COUNT_FIELDS = (
    "input_valid_pixels",
    "excluded_invalid_input_pixels",
    "excluded_outside_aoi_pixels",
)
COUNT_FIELDS = (
    "total_pixels",
    "input_valid_pixels",
    "aoi_pixels",
    "water_invalid_pixels",
    "excluded_invalid_input_pixels",
    "excluded_outside_aoi_pixels",
    "kl_pixels",
    "candidate_pixels",
    "candidate_invalid_input_pixels",
    "water_invalid_outside_aoi_pixels",
    "s2_windows_total",
    "s2_windows_read",
    "s2_windows_skipped",
    "s2_pixels_read",
)


@dataclass(frozen=True)
class EventFiles:
    split: str
    event_id: str
    image: Path
    gt: Path
    floodmap: Path
    output: Path


class MissingAOIError(ValueError):
    """The floodmap contains no explicitly annotated area of interest."""


def collect_events(
    data_root: Path,
    output_root: Path,
    splits: list[str],
    event_ids: list[str] | None = None,
) -> list[EventFiles]:
    requested = set(event_ids or [])
    events = []
    for split in dict.fromkeys(splits):
        gt_dir = data_root / split / "gt"
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"Missing GT directory: {gt_dir}")
        gt_paths = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"})
        if not gt_paths:
            raise ValueError(f"No GT GeoTIFFs found in {gt_dir}")
        for gt_path in gt_paths:
            if requested and gt_path.stem not in requested:
                continue
            event = EventFiles(
                split=split,
                event_id=gt_path.stem,
                image=data_root / split / "S2" / gt_path.name,
                gt=gt_path,
                floodmap=data_root / split / "floodmaps" / f"{gt_path.stem}.geojson",
                output=output_root / split / "kl_mask" / gt_path.name,
            )
            for source in (event.image, event.gt, event.floodmap):
                if not source.is_file():
                    raise FileNotFoundError(f"Missing paired file for {split}/{event.event_id}: {source}")
            events.append(event)
    missing = requested - {event.event_id for event in events}
    if missing:
        raise ValueError(f"Events absent from the selected splits: {', '.join(sorted(missing))}")
    return events


def load_aoi(path: Path, target_crs):
    floodmap = gpd.read_file(path)
    if "w_class" not in floodmap.columns:
        raise ValueError(f"Missing w_class column: {path}")
    aoi = floodmap.loc[floodmap["w_class"] == "area_of_interest"].copy()
    if aoi.empty:
        raise MissingAOIError(f"No area_of_interest polygons: {path}")
    if aoi.crs is None:
        raise ValueError(f"Missing AOI CRS: {path}")
    geometry = aoi.geometry
    if geometry.isna().any() or geometry.is_empty.any():
        raise ValueError(f"Empty or null AOI geometry: {path}")
    if not geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"AOI must contain only Polygon/MultiPolygon geometries: {path}")
    if not geometry.is_valid.all():
        raise ValueError(f"Invalid AOI geometry; review the source annotation: {path}")
    projected = aoi.to_crs(target_crs).geometry
    if not projected.is_valid.all() or projected.is_empty.any():
        raise ValueError(f"AOI geometry became invalid after CRS transformation: {path}")
    return list(projected)


def validate_rasters(image, gt) -> None:
    if gt.count != 2:
        raise ValueError(f"Expected v2 GT with [cloud, water] bands: {gt.name} has {gt.count}")
    if image.count < len(S2_BANDS):
        raise ValueError(f"Expected at least 13 S2 spectral bands: {image.name} has {image.count}")
    if image.crs is None or gt.crs is None:
        raise ValueError(f"S2 and GT must both have a CRS: {image.name}, {gt.name}")
    if image.crs != gt.crs or image.shape != gt.shape or image.transform != gt.transform:
        raise ValueError(f"S2/GT grid mismatch; align the source rasters first: {image.name}, {gt.name}")


def iter_windows(width: int, height: int, block_size: int):
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield Window(col, row, min(block_size, width - col), min(block_size, height - row))


def read_input_valid(image, window: Window) -> np.ndarray:
    raw = image.read(S2_BANDS, window=window, masked=True)
    # Source validity masks handle nodata and dataset masks. All-zero padding
    # and nonfinite values are also excluded, even without nodata metadata.
    return (
        ~np.ma.getmaskarray(raw).any(axis=0)
        & np.isfinite(raw.data).all(axis=0)
        & np.any(raw.data != 0, axis=0)
    )


def read_existing_mask(event: EventFiles) -> dict[str, int | str]:
    """Check completion/provenance metadata without rereading source pixel arrays."""
    error = f"Cannot reuse existing mask {event.output}"
    with rasterio.open(event.image) as image, rasterio.open(event.gt) as gt, rasterio.open(event.output) as mask:
        validate_rasters(image, gt)
        if mask.count != 1 or mask.dtypes != ("uint8",) or mask.nodata is not None:
            raise ValueError(f"{error}: expected a single-band uint8 mask with no nodata")
        if mask.crs != gt.crs or mask.shape != gt.shape or mask.transform != gt.transform:
            raise ValueError(f"{error}: mask/GT grid mismatch")
        tags = mask.tags()
        expected_tags = {
            "rule": MASK_RULE,
            "mask_version": MASK_VERSION,
            "s2_bands": ",".join(map(str, S2_BANDS)),
            "aoi_all_touched": "true",
        }
        for key, expected in expected_tags.items():
            if tags.get(key) != expected:
                raise ValueError(f"{error}: incompatible or missing {key} tag")
        sources = {"source_s2": event.image, "source_gt": event.gt, "source_floodmap": event.floodmap}
        for key, source in sources.items():
            if key not in tags or Path(tags[key]).resolve() != source.resolve():
                raise ValueError(f"{error}: {key} does not match the current source")
        # Include GDAL-listed sidecars (e.g. S2 validity masks) in the timestamp
        # check. Source paths/mtimes are a lightweight check, not content hashes.
        source_files = {Path(path) for path in (*image.files, *gt.files)} | {event.floodmap}
        completed_at = event.output.stat().st_mtime_ns
        if any(source.stat().st_mtime_ns > completed_at for source in source_files):
            raise ValueError(f"{error}: a source was modified after the mask; regenerate it with --overwrite")

        # Version 1 outputs predate the optimization and always had full stats.
        version = tags.get("generator_version", "1")
        mode = "full" if version == "1" else tags.get("statistics_mode")
        if version not in {"1", "2"} or mode not in {"candidate", "full"}:
            raise ValueError(f"{error}: unsupported generator version or statistics mode")
        required = ["total_pixels", "aoi_pixels", "water_invalid_pixels", "kl_pixels"]
        if mode == "full":
            required.extend(FULL_INPUT_COUNT_FIELDS)
        if version == "2":
            required.extend(["candidate_pixels", "candidate_invalid_input_pixels", "water_invalid_outside_aoi_pixels"])
        counts = {}
        for key in required:
            if not tags.get(key, "").isdecimal():
                raise ValueError(f"{error}: missing or invalid completion statistic {key}")
            counts[key] = int(tags[key])
        total = gt.width * gt.height
        if counts["total_pixels"] != total or any(value > total for value in counts.values()):
            raise ValueError(f"{error}: inconsistent completion statistics")
        if counts["aoi_pixels"] == 0 or counts["kl_pixels"] > min(counts["aoi_pixels"], counts["water_invalid_pixels"]):
            raise ValueError(f"{error}: invalid AOI/KL counts")
        if mode == "full" and counts["water_invalid_pixels"] != (
            counts["excluded_invalid_input_pixels"] + counts["excluded_outside_aoi_pixels"] + counts["kl_pixels"]
        ):
            raise ValueError(f"{error}: inconsistent full-input statistics")
        if version == "2" and (
            counts["water_invalid_pixels"] != counts["water_invalid_outside_aoi_pixels"] + counts["candidate_pixels"]
            or counts["candidate_pixels"] != counts["candidate_invalid_input_pixels"] + counts["kl_pixels"]
        ):
            raise ValueError(f"{error}: inconsistent candidate statistics")
    # Do not copy historical S2 read counters into this run's performance fields.
    return {"statistics_mode": mode, **counts}


def generate_mask(
    event: EventFiles,
    block_size: int = 1024,
    overwrite: bool = False,
    *,
    full_input_statistics: bool = False,
) -> dict[str, int | None]:
    """Write the same selection in both modes; full mode additionally scans all S2 input."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if event.output.resolve() in {event.image.resolve(), event.gt.resolve(), event.floodmap.resolve()}:
        raise ValueError(f"Output must not replace an input file: {event.output}")
    if event.output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace generated masks: {event.output}")

    with rasterio.open(event.image) as image, rasterio.open(event.gt) as gt:
        validate_rasters(image, gt)
        geometries = load_aoi(event.floodmap, gt.crs)
        counts = dict.fromkeys(COUNT_FIELDS, 0)
        event.output.parent.mkdir(parents=True, exist_ok=True)
        # Publish only completed TIFFs; failures must not leave partial masks
        # or replace an existing result with an incomplete one.
        with TemporaryDirectory(prefix=f".{event.event_id}.", dir=event.output.parent) as temporary:
            temporary_path = Path(temporary) / event.output.name
            with rasterio.open(
                temporary_path,
                "w",
                driver="GTiff",
                width=gt.width,
                height=gt.height,
                count=1,
                dtype="uint8",
                crs=gt.crs,
                transform=gt.transform,
                nodata=None,  # Both 0 (not selected) and 1 (KL-only) are valid values.
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF="IF_SAFER",
            ) as output:
                output.set_band_description(1, "KL-only selection: 0=not selected, 1=selected")
                output.update_tags(
                    rule=MASK_RULE,
                    mask_version=MASK_VERSION,
                    generator_version="2",
                    statistics_mode="full" if full_input_statistics else "candidate",
                    block_size=block_size,
                    s2_bands=",".join(map(str, S2_BANDS)),
                    aoi_all_touched="true",
                    source_s2=str(event.image.resolve()),
                    source_gt=str(event.gt.resolve()),
                    source_floodmap=str(event.floodmap.resolve()),
                )
                for window in iter_windows(gt.width, gt.height, block_size):
                    # GT nodata=0 is an invalid LABEL, not necessarily invalid INPUT.
                    water = gt.read(2, window=window, masked=False)
                    if not np.isin(water, (0, 1, 2)).all():
                        raise ValueError(f"Water GT contains values outside {{0, 1, 2}}: {event.gt}, {window}")
                    inside_aoi = rasterize(
                        geometries,
                        out_shape=water.shape,
                        transform=gt.window_transform(window),
                        fill=0,
                        default_value=1,
                        all_touched=True,  # Match ml4floods.data.create_gt.compute_water.
                        dtype="uint8",
                        skip_invalid=False,
                    ).astype(bool)
                    water_invalid = water == 0
                    candidate = inside_aoi & water_invalid
                    candidate_count = int(candidate.sum())
                    counts["total_pixels"] += water.size
                    counts["aoi_pixels"] += int(inside_aoi.sum())
                    counts["water_invalid_pixels"] += int(water_invalid.sum())
                    counts["candidate_pixels"] += candidate_count
                    counts["water_invalid_outside_aoi_pixels"] += int((water_invalid & ~inside_aoi).sum())
                    counts["s2_windows_total"] += 1
                    if candidate_count == 0 and not full_input_statistics:
                        # No pixel in this window can contribute to KL. Preserve
                        # the output grid without reading any S2 pixel arrays.
                        output.write(np.zeros(water.shape, dtype=np.uint8), 1, window=window)
                        counts["s2_windows_skipped"] += 1
                        continue

                    input_valid = read_input_valid(image, window)
                    counts["s2_windows_read"] += 1
                    counts["s2_pixels_read"] += water.size
                    selected = input_valid & candidate
                    output.write(selected.astype(np.uint8), 1, window=window)
                    counts["candidate_invalid_input_pixels"] += int((candidate & ~input_valid).sum())
                    counts["kl_pixels"] += int(selected.sum())
                    if full_input_statistics:
                        counts["input_valid_pixels"] += int(input_valid.sum())
                        counts["excluded_invalid_input_pixels"] += int((water_invalid & ~input_valid).sum())
                        counts["excluded_outside_aoi_pixels"] += int((water_invalid & input_valid & ~inside_aoi).sum())
                if counts["aoi_pixels"] == 0:
                    raise ValueError(f"AOI does not cover any GT pixels: {event.floodmap}")
                if not full_input_statistics:
                    # Unknown full-image statistics are blank in CSV and absent
                    # from TIFF tags, rather than misleading zeros or partial sums.
                    counts.update(dict.fromkeys(FULL_INPUT_COUNT_FIELDS))
                output.update_tags(**{key: value for key, value in counts.items() if value is not None})
            temporary_path.replace(event.output)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing split/{S2,gt,floodmaps}.")
    parser.add_argument("--output-root", type=Path, required=True, help="Write split/kl_mask/<event>.tif under this root.")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--events", nargs="+", help="Optional exact event stems, without .tif.")
    parser.add_argument("--block-size", type=int, default=1024, help="Process windows of this size (default: 1024).")
    parser.add_argument(
        "--full-input-statistics", action="store_true",
        help="Read every S2 window for full-image input statistics; default skips windows without KL candidates.",
    )
    parser.add_argument("--skip-missing-aoi", action="store_true", help="Record and skip missing AOIs; default is to fail.")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true", help="Replace existing generated masks.")
    existing.add_argument("--skip-existing", action="store_true", help="Validate and reuse completed masks; generate only missing outputs.")
    args = parser.parse_args(argv)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    events = collect_events(args.data_root, args.output_root, args.splits, args.events)
    if not args.overwrite and not args.skip_existing:
        existing = [event.output for event in events if event.output.exists()]
        if existing:
            raise FileExistsError(
                f"{len(existing)} masks already exist; use --skip-existing to resume or --overwrite to replace. "
                f"First: {existing[0]}"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = args.output_root / f"kl_mask_summary_{stamp}.csv"
    fields = ["split", "event_id", "status", "reason", "statistics_mode", *COUNT_FIELDS, "elapsed_seconds", "mask_path"]
    written = skipped = reused = total_kl = 0
    statistics_mode = "full" if args.full_input_statistics else "candidate"
    print(f"Summary: {report_path}", flush=True)
    print(f"Statistics mode for new masks: {statistics_mode}", flush=True)
    with report_path.open("x", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=fields)
        writer.writeheader()
        report.flush()
        for index, event in enumerate(events, 1):
            print(f"[{index}/{len(events)}] {event.split}/{event.event_id}", flush=True)
            row = {
                "split": event.split, "event_id": event.event_id,
                "mask_path": str(event.output), "statistics_mode": statistics_mode,
            }
            started = perf_counter()
            if args.skip_existing and event.output.exists():
                saved_counts = read_existing_mask(event)
                row.update(
                    status="skipped_existing", reason="",
                    elapsed_seconds=round(perf_counter() - started, 3), **saved_counts,
                )
                writer.writerow(row)
                report.flush()
                reused += 1
                total_kl += saved_counts["kl_pixels"]
                print(f"  SKIPPED existing: {event.output}", flush=True)
                continue
            try:
                counts = generate_mask(
                    event, args.block_size, args.overwrite,
                    full_input_statistics=args.full_input_statistics,
                )
            except MissingAOIError as error:
                existing_mask = event.output.exists()
                row.update(
                    status="missing_aoi",
                    reason=str(error),
                    mask_path=str(event.output) if existing_mask else "",
                    elapsed_seconds=round(perf_counter() - started, 3),
                )
                writer.writerow(row)
                report.flush()
                if existing_mask:
                    raise ValueError(
                        f"Missing AOI but a previous mask remains: {event.output}. "
                        "Use a fresh --output-root before skipping this event."
                    ) from error
                if not args.skip_missing_aoi:
                    raise
                skipped += 1
                print(f"  SKIPPED: {error}", flush=True)
                continue
            elapsed = perf_counter() - started
            row.update(status="written", reason="", elapsed_seconds=round(elapsed, 3), **counts)
            writer.writerow(row)
            report.flush()
            written += 1
            total_kl += counts["kl_pixels"]
            print(
                f"  KL pixels: {counts['kl_pixels']:,}/{counts['total_pixels']:,}; "
                f"S2 windows read: {counts['s2_windows_read']}/{counts['s2_windows_total']} "
                f"(skipped {counts['s2_windows_skipped']}); {elapsed:.1f}s; {event.output}",
                flush=True,
            )
    print(
        f"Written: {written}; existing reused: {reused}; missing AOI skipped: {skipped}; "
        f"KL pixels in available masks: {total_kl:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
