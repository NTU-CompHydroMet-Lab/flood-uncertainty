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
    parser.add_argument("--curve-subset", choices=["val", "test"], default="val")
    parser.add_argument("--event", default="EMSR466_AOI01_DEL_PRODUCT")
    parser.add_argument("--analysis-root", type=Path,
                        default=PROJECT_ROOT / "artifacts/results/analysis_S2")
    parser.add_argument("--output-dir", type=Path,
                        help="Override the output folder for the PNG and two CSV tables.")
    args = parser.parse_args()
    if args.event in {"OVERALL", "AVERAGE"}:
        parser.error("--event must be an event filename, not OVERALL or AVERAGE.")
    return args


def read_scores(path, args):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing PAvPU CSV: {path}\n"
            "Complete compute_pavpu.py for this model and curve source before exporting."
        )
    data = pd.read_csv(path)
    missing = {"filename", "pavpu"} - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    for column, expected in (("subset", args.subset), ("curve_subset", args.curve_subset)):
        if column in data and not data[column].eq(expected).all():
            raise ValueError(f"Expected {column}={expected} in {path}")

    rows = data.set_index("filename", verify_integrity=True)
    if not {"OVERALL", args.event}.issubset(rows.index):
        raise ValueError(f"Expected OVERALL and event {args.event} in {path}")
    scores = rows.loc[["OVERALL", args.event], "pavpu"].astype(float)
    if not scores.between(0, 1).all():
        raise ValueError(f"PAvPU values must be finite and within [0, 1]: {path}")

    protocol = []
    for column in ("retention_target", "patch_size"):
        if column not in data:
            continue
        if data[column].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent {column} in {path}")
        protocol.append((column, data[column].iloc[0]))
    return scores["OVERALL"], scores[args.event], tuple(protocol)


def build_tables(args):
    overall_rows, event_rows = [], []
    protocols = set()
    for model, (inference_type, method) in MODELS.items():
        model_dir = args.analysis_root / args.subset / model / f"pavpu_from_{args.curve_subset}"
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
        raise ValueError("Input CSVs use different retention targets or patch sizes/metadata.")
    return pd.DataFrame(overall_rows), pd.DataFrame(event_rows)


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
    overall, event = build_tables(args)
    output_dir = args.output_dir or (
        PROJECT_ROOT / "artifacts/results/figures/pavpu_tables"
        / f"{args.subset}_from_{args.curve_subset}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
                 f"Evaluation: {args.subset} | Threshold curves: {args.curve_subset}",
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
