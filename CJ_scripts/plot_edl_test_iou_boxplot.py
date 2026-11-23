"""Scatter EDL member test IoUs with ensemble, UNet++ and single EDL references."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "artifacts/results"
NAS_TEST_ROOT = Path(
    "/home/NAS/house/chlunchen-10030/datasets/cj/cj/ML4FloodsUncertainty"
    "/result/val_test_inference/test"
)
DEFAULT_MEMBERS = [0, 4, 5, 6, 7, 8, 9]
COUNT_COLUMNS = ["TP", "TN", "FP", "FN"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, nargs="+", default=DEFAULT_MEMBERS,
                        help="EDL member IDs to include; default: 0 4 5 6 7 8 9.")
    parser.add_argument("--members-root", type=Path,
                        default=RESULTS_ROOT / "edl_ensemble_members")
    parser.add_argument("--ensemble-csv", type=Path,
                        default=NAS_TEST_ROOT / "ensemble/metrics_ensemble.csv")
    parser.add_argument("--unet-csv", type=Path,
                        default=NAS_TEST_ROOT / "v2/metrics_v2.csv")
    parser.add_argument("--edl-csv", type=Path,
                        default=RESULTS_ROOT / "val_test_inference/test/EDL/metrics_EDL.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=RESULTS_ROOT / "figures/edl_test_iou_boxplot")
    args = parser.parse_args()
    if len(args.members) < 2 or len(set(args.members)) != len(args.members):
        parser.error("--members must contain at least two distinct member IDs.")
    if any(member < 0 for member in args.members):
        parser.error("--members must be nonnegative integers.")
    return args


def read_result(path):
    """Read global Water IoU and per-event GT counts for coverage checks."""
    data = pd.read_csv(path)
    required = {"filename", "subset", "iou", *COUNT_COLUMNS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    if data["filename"].isna().any() or data["filename"].duplicated().any():
        raise ValueError(f"Missing or duplicate event names in {path}")
    if not data["subset"].eq("test").all():
        raise ValueError(f"Expected only test results in {path}")

    rows = data.set_index("filename")
    if "OVERALL" not in rows.index:
        raise ValueError(f"Missing OVERALL row in {path}")
    events = rows.drop(index=["OVERALL", "AVERAGE"], errors="ignore").sort_index()
    if events.empty:
        raise ValueError(f"Missing per-event results in {path}")

    counts = events[COUNT_COLUMNS].astype(float)
    overall_counts = rows.loc["OVERALL", COUNT_COLUMNS].astype(float)
    if not np.isfinite(counts.to_numpy()).all() or (counts < 0).any().any():
        raise ValueError(f"Invalid confusion counts in {path}")
    if not np.array_equal(counts.sum().to_numpy(), overall_counts.to_numpy()):
        raise ValueError(f"OVERALL counts do not match the event totals in {path}")
    score = float(rows.loc["OVERALL", "iou"])
    denominator = overall_counts[["TP", "FP", "FN"]].sum()
    expected = overall_counts["TP"] / denominator if denominator > 0 else 0.0
    if not 0 <= score <= 1 or not np.isclose(score, expected, rtol=1e-9, atol=1e-12):
        raise ValueError(f"Invalid OVERALL Water IoU in {path}")

    coverage = pd.DataFrame({
        "water_pixels": counts["TP"] + counts["FN"],
        "land_pixels": counts["TN"] + counts["FP"],
    })
    config_path = path.with_name("run_config.json")
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    result = {
        "subset": "test",
        "iou": score,
        "n_events": len(events),
        "max_tile_size": config.get("max_tile_size", "unknown"),
        "metrics_csv": str(path.resolve()),
        **overall_counts.to_dict(),
    }
    return result, coverage


def build_table(args):
    sources = [
        ("member", f"EDL member {member}", member,
         args.members_root / f"member_{member}/test/EDL/metrics_EDL.csv")
        for member in args.members
    ]
    sources.extend([
        ("reference", "Ensemble", "", args.ensemble_csv),
        ("reference", "UNet++", "", args.unet_csv),
        ("reference", "Single EDL", "", args.edl_csv),
    ])

    records = []
    expected_coverage = None
    for kind, label, member, path in sources:
        result, coverage = read_result(path)
        if expected_coverage is None:
            expected_coverage = coverage
        if not coverage.equals(expected_coverage):
            raise ValueError(
                f"Test events or per-event ground-truth pixel counts differ in {path}"
            )
        records.append({"kind": kind, "label": label, "member": member, **result})
    return pd.DataFrame(records)


def draw_plot(table, output_path, *, show_legend_values=True):
    members = table.loc[table["kind"].eq("member")]
    references = table.loc[table["kind"].eq("reference")]
    blue, red = "#24658C", "#C62828"
    with plt.rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 11}):
        fig, ax = plt.subplots(figsize=(4.6, 6.5))
        member_label = (
            f"Individual EDL runs (n = {len(members)})"
            if show_legend_values else "Individual EDL runs"
        )
        # Fixed offsets show all runs without introducing another random seed.
        ax.scatter(1 + np.linspace(-0.14, 0.14, len(members)), members["iou"],
                   s=42, color=blue, edgecolors="white", linewidths=0.7,
                   zorder=3, label=member_label)
        # References share the same category; offsets keep nearby IoUs visible.
        reference_positions = 1 + np.linspace(-0.1, 0.1, len(references))
        for position, marker, row in zip(
            reference_positions, ["o", "D", "^"], references.itertuples(index=False)
        ):
            ax.scatter(position, row.iou, s=40, color=red, marker=marker,
                       edgecolors="white", linewidths=0.8, zorder=4,
                       label=f"{row.label} ({row.iou:.4f})" if show_legend_values else row.label)

        ax.set_xticks([])
        ax.set_xlim(0.5, 1.5)
        padding = max(float(table["iou"].max() - table["iou"].min()) * 0.3, 0.01)
        # Use at least 0.75–0.90, expanding when needed to keep all scores visible.
        ax.set_ylim(min(0.75, max(0, table["iou"].min() - padding)),
                    max(0.90, min(1, table["iou"].max() + padding)))
        ax.set_ylabel("Test Water IoU (OVERALL)")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_title("EDL test IoU across training runs", fontsize=12, pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=1, frameon=False, fontsize=10)

        member_ids = ", ".join(members["member"].astype(str))
        member_tiles = ", ".join(dict.fromkeys(members["max_tile_size"].astype(str)))
        reference_tiles = ", ".join(dict.fromkeys(references["max_tile_size"].astype(str)))
        caption = (
            f"Selected members: {member_ids}\n"
            f"Each point: one test result | Same {int(table['n_events'].iloc[0])} events\n"
            f"Inference tile size: EDL runs = {member_tiles};\n"
            f"references = {reference_tiles}."
        )
        fig.subplots_adjust(left=0.20, right=0.96, top=0.90, bottom=0.42)
        fig.text(0.08, 0.04, caption, fontsize=8, color="0.35", linespacing=1.4)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    args = parse_args()
    table = build_table(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "edl_test_iou_boxplot.png"
    names_only_path = args.output_dir / "edl_test_iou_scatter_names_only.png"
    csv_path = args.output_dir / "edl_test_iou_comparison.csv"
    draw_plot(table, png_path)
    draw_plot(table, names_only_path, show_legend_values=False)
    table.to_csv(csv_path, index=False)
    print(table[["label", "iou", "n_events", "max_tile_size"]].to_string(index=False))
    for path in (png_path, names_only_path, csv_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
