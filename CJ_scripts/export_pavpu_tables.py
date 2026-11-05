"""Export overall and per-event PAvPU tables from existing result CSVs."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "EDL": ("Single-pass", "EDL"),
    "ensemble": ("Multi-pass", "Deep Ensemble"),
    "mcdropout": ("Multi-pass", "MC Dropout"),
}
CRITERIA = [
    ("Water_DST_Uncertainty", "DST"),
    ("Water_Aleatoric", "Aleatoric"),
    ("Water_Epistemic", "Epistemic"),
    ("Water_Aleatoric_Plus_Epistemic", "Aleatoric + Epistemic"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["val", "test"], default="test")
    parser.add_argument("--threshold-mode", choices=["curve", "per-image"], default="curve",
                        help="PAvPU result mode to read. Default: curve.")
    parser.add_argument("--retention-target", type=float, default=0.9,
                        help="Per-image uncertainty quantile used to select results. Default: 0.9.")
    parser.add_argument("--curve-subset", choices=["val", "test"], default="val",
                        help="Threshold curve source, used only in curve mode.")
    parser.add_argument("--event", default="EMSR466_AOI01_DEL_PRODUCT")
    parser.add_argument("--analysis-root", type=Path,
                        default=PROJECT_ROOT / "artifacts/results/analysis_S2")
    parser.add_argument("--output-dir", type=Path,
                        help="Override the output folder for the PNG and two CSV tables.")
    args = parser.parse_args()
    if args.event in {"OVERALL", "AVERAGE"}:
        parser.error("--event must be an event filename, not OVERALL or AVERAGE.")
    if not 0 < args.retention_target <= 1:
        parser.error("--retention-target must be in (0, 1].")
    return args


def result_suffix(args):
    if args.threshold_mode == "per-image":
        return f"per_image_q{args.retention_target * 100:g}"
    return f"from_{args.curve_subset}"


def read_scores(path, args):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing PAvPU CSV: {path}\n"
            "Complete compute_pavpu.py for this model and threshold mode before exporting."
        )
    data = pd.read_csv(path)
    required = {"filename", "pavpu"}
    if args.threshold_mode == "per-image":
        required.update({"subset", "threshold_mode", "retention_target", "patch_size", "acc_threshold"})
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    expected_metadata = {"subset": args.subset, "threshold_mode": args.threshold_mode}
    if args.threshold_mode == "curve":
        expected_metadata["curve_subset"] = args.curve_subset
    for column, expected in expected_metadata.items():
        if column in data and not data[column].eq(expected).all():
            raise ValueError(f"Expected {column}={expected} in {path}")

    rows = data.set_index("filename", verify_integrity=True)
    if not {"OVERALL", args.event}.issubset(rows.index):
        raise ValueError(f"Expected OVERALL and event {args.event} in {path}")
    scores = rows.loc[["OVERALL", args.event], "pavpu"].astype(float)
    if not scores.between(0, 1).all():
        raise ValueError(f"PAvPU values must be finite and within [0, 1]: {path}")

    protocol = []
    protocol_columns = ["retention_target", "patch_size"]
    if args.threshold_mode == "per-image":
        protocol_columns.append("acc_threshold")
    for column in protocol_columns:
        if column not in data:
            continue
        if data[column].isna().any() or data[column].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent {column} in {path}")
        protocol.append((column, data[column].iloc[0]))

    if args.threshold_mode == "per-image":
        if not data["retention_target"].eq(args.retention_target).all():
            raise ValueError(f"Expected retention_target={args.retention_target} in {path}")
        if not data["acc_threshold"].between(0, 1).all():
            raise ValueError(f"Expected acc_threshold in [0, 1] in {path}")
        patch_size = float(data["patch_size"].iloc[0])
        if not (patch_size > 0 and patch_size.is_integer()):
            raise ValueError(f"Expected a positive integer patch_size in {path}")
    return scores["OVERALL"], scores[args.event], tuple(protocol)


def build_tables(args):
    overall_rows, event_rows = [], []
    protocols = set()
    for model, (inference_type, method) in MODELS.items():
        model_dir = args.analysis_root / args.subset / model / f"pavpu_{result_suffix(args)}"
        event_row = {"Model": method}
        event_row.update({f"PAvPU ({label})": float("nan") for _, label in CRITERIA})
        criteria = CRITERIA if model == "EDL" else CRITERIA[1:]
        for key, label in criteria:
            path = model_dir / f"pavpu_{key}.csv"
            overall, event_score, protocol = read_scores(path, args)
            protocols.add(protocol)
            overall_rows.append({
                "Inference type": inference_type,
                "Method": method,
                "Uncertainty type": "DST uncertainty" if label == "DST" else label,
                "PAvPU": overall,
            })
            event_row[f"PAvPU ({label})"] = event_score
        event_rows.append(event_row)

    if len(protocols) != 1:
        raise ValueError("Input CSVs use different evaluation settings "
                         "(retention target, patch size, or fixed accuracy threshold in per-image mode).")
    return pd.DataFrame(overall_rows), pd.DataFrame(event_rows), dict(protocols.pop())


def draw_table(ax, data, title, widths, bbox, numeric_start):
    ax.axis("off")
    ax.text(0, 0.98, title, transform=ax.transAxes, va="top", fontsize=14)
    cells = [
        [
            "—" if pd.isna(value) else f"{value:.4f}" if column >= numeric_start else str(value)
            for column, value in enumerate(row)
        ]
        for row in data.itertuples(index=False, name=None)
    ]
    table = ax.table(
        cellText=cells, colLabels=list(data.columns),
        cellLoc="left", colLoc="left", colWidths=widths, bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, column), cell in table.get_celld().items():
        cell.visible_edges = {0: "TB", len(data): "B"}.get(row, "")
        cell.set_linewidth(0.7)
        cell.set_facecolor("white")
        cell.PAD = 0.06
        if column >= numeric_start:
            cell.get_text().set_ha("center")


def main():
    args = parse_args()
    overall, event, protocol = build_tables(args)
    output_dir = args.output_dir or (
        PROJECT_ROOT / "artifacts/results/figures/pavpu_tables"
        / f"{args.subset}_{result_suffix(args)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_caption = f"Threshold curves: {args.curve_subset}"
    if args.threshold_mode == "per-image":
        patch_size = int(protocol["patch_size"])
        threshold_caption = (
            f"Threshold: Q{args.retention_target * 100:g} of valid pixels per event | "
            f"Patch: {patch_size}x{patch_size} | Accuracy: Water IoU >= {protocol['acc_threshold']:g}"
        )

    with plt.rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"]}):
        fig, axes = plt.subplots(2, 1, figsize=(15.5, 9),
                                 gridspec_kw={"height_ratios": [3, 1.3]})
        draw_table(
            axes[0], overall,
            "Table 4. Overall PAvPU values across uncertainty definitions.",
            widths=[0.23, 0.25, 0.38, 0.14], bbox=[0.18, 0, 0.66, 0.86], numeric_start=3,
        )
        draw_table(
            axes[1], event,
            f"Table 5. Uncertainty reliability for the representative event {args.event}.",
            widths=[0.17, 0.13, 0.20, 0.20, 0.30], bbox=[0.01, 0.05, 0.98, 0.72],
            numeric_start=1,
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.07, hspace=0.20)
        fig.text(0.02, 0.02,
                 f"Evaluation: {args.subset} | {threshold_caption}",
                 fontsize=10, color="0.35")
        png_path = output_dir / "pavpu_tables.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    overall_path = output_dir / "table4_pavpu_overall.csv"
    event_path = output_dir / "table5_pavpu_event.csv"
    overall.to_csv(overall_path, index=False, float_format="%.4f")
    event.assign(filename=args.event).to_csv(
        event_path, index=False, float_format="%.4f", na_rep="—"
    )
    for path in (png_path, overall_path, event_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
