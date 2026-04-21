import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
os.environ.setdefault("CARTOPY_DATA_DIR", str(Path("/tmp") / "cartopy"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "report" / "image_of_md"
DEFAULT_OUTPUT_NAME = "worldfloods_event_map.png"

SPLIT_ORDER = ["train", "val", "test"]
SPLIT_LABELS = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}
SPLIT_COLORS = {
    "train": "blue",
    "val": "green",
    "test": "#B22222",
}

LAND_COLOR = "#F2F1DD"
OCEAN_COLOR = "#9DBBE0"
EDGE_COLOR = "black"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot WorldFloods v2 flood event locations by train/validation/test split."
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help=f"WorldFloods v2 data root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output file name. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--level",
        choices=["event", "map"],
        default="event",
        help="Plot article-style flood events or all flood maps. Default: event",
    )
    parser.add_argument("--dpi", default=300, type=int, help="Figure DPI. Default: 300")
    parser.add_argument(
        "--marker-size",
        default=42,
        type=float,
        help="Scatter marker size. Default: 42",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Use the GeoPandas fallback even when Cartopy is installed.",
    )
    return parser.parse_args()


def event_key(row):
    if row["source"] == "glofimr":
        return row["event id"]
    return row["ems_code"]


def read_json(path):
    with path.open() as f:
        return json.load(f)


def bbox_center(bbox):
    west = float(bbox["west"])
    east = float(bbox["east"])
    south = float(bbox["south"])
    north = float(bbox["north"])
    return (west + east) / 2, (south + north) / 2


def union_bbox(bboxes):
    return {
        "west": min(float(b["west"]) for b in bboxes),
        "east": max(float(b["east"]) for b in bboxes),
        "south": min(float(b["south"]) for b in bboxes),
        "north": max(float(b["north"]) for b in bboxes),
    }


def load_map_rows(data_root):
    metadata_path = data_root / "dataset_metadata.csv"
    with metadata_path.open() as f:
        rows = list(csv.DictReader(f))

    map_rows = []
    for row in rows:
        meta_path = data_root / row["split"] / "meta" / f"{row['event id']}.json"
        meta = read_json(meta_path)
        lon, lat = bbox_center(meta["bounding box"])
        map_rows.append(
            {
                "split": row["split"],
                "source": row["source"],
                "event_key": event_key(row),
                "event_id": row["event id"],
                "lon": lon,
                "lat": lat,
                "bbox": meta["bounding box"],
            }
        )
    return map_rows


def build_plot_rows(map_rows, level):
    if level == "map":
        return map_rows

    groups = {}
    for row in map_rows:
        key = (row["split"], row["source"], row["event_key"])
        groups.setdefault(key, []).append(row)

    event_rows = []
    for (split, source, key), rows in groups.items():
        bbox = union_bbox([row["bbox"] for row in rows])
        lon, lat = bbox_center(bbox)
        event_rows.append(
            {
                "split": split,
                "source": source,
                "event_key": key,
                "event_id": rows[0]["event_id"],
                "lon": lon,
                "lat": lat,
                "bbox": bbox,
                "n_maps": len(rows),
            }
        )

    return event_rows


def count_by_split(rows):
    return {split: sum(1 for row in rows if row["split"] == split) for split in SPLIT_ORDER}


def draw_points(ax, rows, transform=None, marker_size=42):
    for split in SPLIT_ORDER:
        split_rows = [row for row in rows if row["split"] == split]
        kwargs = {}
        if transform is not None:
            kwargs["transform"] = transform
        ax.scatter(
            [row["lon"] for row in split_rows],
            [row["lat"] for row in split_rows],
            s=marker_size,
            c=SPLIT_COLORS[split],
            edgecolors="none",
            zorder=5,
            **kwargs,
        )


def add_legend(fig):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SPLIT_COLORS[split],
            markeredgecolor="none",
            markersize=9,
            label=SPLIT_LABELS[split],
        )
        for split in SPLIT_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        columnspacing=3.0,
        handletextpad=0.8,
    )


def plot_with_cartopy(rows, output_path, dpi, marker_size):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    projection = ccrs.InterruptedGoodeHomolosine()
    fig = plt.figure(figsize=(15.5, 7.2))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_global()
    ax.set_facecolor(OCEAN_COLOR)

    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=OCEAN_COLOR, linewidth=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND_COLOR, edgecolor=EDGE_COLOR, linewidth=0.45)
    ax.coastlines(resolution="110m", linewidth=0.45, color=EDGE_COLOR)

    draw_points(ax, rows, transform=ccrs.PlateCarree(), marker_size=marker_size)
    add_legend(fig)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.12)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def pyogrio_naturalearth_path():
    try:
        import pyogrio
    except ImportError:
        return None

    return Path(pyogrio.__file__).resolve().parent / "tests" / "fixtures" / "naturalearth_lowres" / "naturalearth_lowres.shp"


def plot_with_geopandas_fallback(rows, output_path, dpi, marker_size):
    import geopandas as gpd

    world_path = pyogrio_naturalearth_path()
    if world_path is None or not world_path.exists():
        raise FileNotFoundError("Cartopy is not installed and no local Natural Earth fallback shapefile was found.")

    world = gpd.read_file(world_path)

    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_facecolor(OCEAN_COLOR)
    world.plot(ax=ax, color=LAND_COLOR, edgecolor=EDGE_COLOR, linewidth=0.45)
    draw_points(ax, rows, marker_size=marker_size)
    add_legend(fig)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-62, 85)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.12)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    map_rows = load_map_rows(data_root)
    plot_rows = build_plot_rows(map_rows, args.level)

    used_backend = "cartopy"
    if args.force_fallback:
        used_backend = "geopandas-fallback"
        plot_with_geopandas_fallback(plot_rows, output_path, args.dpi, args.marker_size)
    else:
        try:
            plot_with_cartopy(plot_rows, output_path, args.dpi, args.marker_size)
        except ImportError:
            used_backend = "geopandas-fallback"
            plot_with_geopandas_fallback(plot_rows, output_path, args.dpi, args.marker_size)

    counts = count_by_split(plot_rows)
    print(f"Saved {args.level}-level map to {output_path}")
    print(f"Backend: {used_backend}")
    print("Counts:", ", ".join(f"{SPLIT_LABELS[split]}={counts[split]}" for split in SPLIT_ORDER))


if __name__ == "__main__":
    main()
