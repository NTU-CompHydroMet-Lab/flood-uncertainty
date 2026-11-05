"""Compare val/test retention curves for ensemble, EDL or mcdropout using existing CSVs."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "artifacts/results/analysis_S2"
NAS_ANALYSIS_ROOT = Path(
    "/home/NAS/house/chlunchen-10030/datasets/cj/cj/ML4FloodsUncertainty"
    "/result/analysis_S2"
)
DEFAULT_DIRECTORIES = {
    "ensemble": {
        "val": ANALYSIS_ROOT / "val/ensemble",
        "test": NAS_ANALYSIS_ROOT / "ensemble",
    },
    "EDL": {
        "val": ANALYSIS_ROOT / "val/EDL",
        "test": ANALYSIS_ROOT / "test/EDL",
    },
    "mcdropout": {
        "val": ANALYSIS_ROOT / "val/mcdropout",
        "test": NAS_ANALYSIS_ROOT / "mcdropout",
    },
}
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/results/figures/retention_val_test"
CRITERIA = [
    ("Water_Aleatoric", "Aleatoric"),
    ("Water_Epistemic", "Epistemic"),
    ("Water_Aleatoric_Plus_Epistemic", "Aleatoric + Epistemic"),
]
MODEL_CRITERIA = {
    "ensemble": CRITERIA,
    "EDL": [("Water_DST_Uncertainty", "DST")] + CRITERIA,
    "mcdropout": CRITERIA,
}
METRIC_LABELS = {
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "accuracy": "Accuracy",
}
SUBSET_STYLES = {
    "val": {"color": "#005A8D", "linestyle": "-", "marker": "o"},
    "test": {"color": "#B35A00", "linestyle": "--", "marker": "s"},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=MODEL_CRITERIA, default="ensemble")
    parser.add_argument("--val-dir", type=Path,
                        help="Override the selected model's val retention CSV folder.")
    parser.add_argument("--test-dir", type=Path,
                        help="Override the selected model's test retention CSV folder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", choices=METRIC_LABELS, default="iou")
    return parser.parse_args()


def load_curve(path, metric):
    data = pd.read_csv(path, usecols=["retention_rate", metric], dtype=float)
    if data.empty:
        raise ValueError(f"Retention CSV is empty: {path}")

    valid = data["retention_rate"].between(0, 1, inclusive="right") & data[metric].between(0, 1)
    if not valid.all():
        raise ValueError(f"Invalid retention rate or {metric} value in: {path}")

    if data["retention_rate"].duplicated().any():
        raise ValueError(f"Duplicate retention rates in: {path}")
    return data.sort_values("retention_rate")


def main():
    args = parse_args()
    model_type = args.model_type
    criteria = MODEL_CRITERIA[model_type]
    defaults = DEFAULT_DIRECTORIES[model_type]
    directories = {
        "val": args.val_dir or defaults["val"],
        "test": args.test_dir or defaults["test"],
    }
    curves = {
        subset: [
            load_curve(directory / f"retention_{key}.csv", args.metric)
            for key, _ in criteria
        ]
        for subset, directory in directories.items()
    }

    fig, axes = plt.subplots(
        1, len(criteria), figsize=(5.2 * len(criteria), 4.8),
        sharex=True, sharey=True, layout="constrained",
    )
    for index, (ax, (_, title)) in enumerate(zip(axes, criteria)):
        for subset, style in SUBSET_STYLES.items():
            data = curves[subset][index]
            ax.plot(
                data["retention_rate"] * 100, data[args.metric],
                label=subset, linewidth=1.9, markevery=1,
                markersize=5, markerfacecolor="white", **style,
            )
        ax.set_title(title)
        ax.set_xlabel("Retention Rate (%)")
        ax.grid(True, linestyle="--", alpha=0.15)

    axes[0].set_xlim(100, 0)
    axes[0].set_ylabel(METRIC_LABELS[args.metric])
    axes[-1].legend(loc="lower right")
    fig.suptitle(f"{model_type} Retention Curves: val vs test ({METRIC_LABELS[args.metric]})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"retention_curve_{model_type}_val_test_{args.metric}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
