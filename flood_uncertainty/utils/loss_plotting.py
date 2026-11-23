"""Save EDL epoch losses and headless plots from local Lightning CSV logs."""

from pathlib import Path

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import pandas as pd


LOSS_COLUMNS = (
    "train_loss_epoch",
    "val_bce_loss",
    "val_bce_clear_cloud",
    "val_bce_land_water",
)
HISTORY_COLUMNS = ("epoch", *LOSS_COLUMNS)


def read_epoch_losses(metrics_path: Path) -> pd.DataFrame:
    """Combine sparse train/validation rows without averaging step-level losses."""
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    frame = pd.read_csv(metrics_path).reindex(columns=HISTORY_COLUMNS)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([float("inf"), -float("inf")], float("nan"))
    frame = frame.dropna(subset=["epoch"]).dropna(subset=LOSS_COLUMNS, how="all")
    if frame.empty:
        return frame
    frame["epoch"] = frame["epoch"].astype(int)
    # Lightning writes validation and training aggregates on separate CSV rows.
    return frame.groupby("epoch", as_index=False, sort=True).last()


def collect_loss_history(run_dir: Path) -> pd.DataFrame:
    """Merge logger versions, replacing stale future epochs after a resume."""
    histories = []
    for path in (run_dir / "logs").glob("version_*/metrics.csv"):
        version = path.parent.name.removeprefix("version_")
        if version.isdigit():
            histories.append((int(version), path))
    history = pd.DataFrame(columns=HISTORY_COLUMNS)
    for _, path in sorted(histories):
        current = read_epoch_losses(path)
        if current.empty:
            continue
        history = history.loc[history["epoch"] < current["epoch"].min()]
        if history.empty:
            history = current
            continue
        history = pd.concat([history, current], ignore_index=True)
    return history.sort_values("epoch").reset_index(drop=True)


def _figure(title: str):
    figure = Figure(figsize=(12, 4.8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    figure.suptitle(title)
    for axis in axes:
        axis.set_xlabel("Epoch (1-based)")
        axis.set_ylabel("EDL loss")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
    figure.text(
        0.5, 0.015,
        "Training and validation use different class/task weights and KL schedules.",
        ha="center", fontsize=9,
    )
    return figure, axes


def _plot_series(axis, history: pd.DataFrame, column: str, label: str, **kwargs):
    series = history.dropna(subset=[column])
    if not series.empty:
        axis.plot(series["epoch"] + 1, series[column], label=label, **kwargs)


def _save_figure(figure, axes, output_path: Path) -> Path:
    for axis in axes:
        if axis.lines:
            axis.legend(fontsize=8, ncol=2)
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    figure.clear()
    return output_path


def plot_run_losses(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    history = collect_loss_history(run_dir)
    if history.empty:
        print(f"No epoch losses available to plot in {run_dir}")
        return None
    history.to_csv(run_dir / "loss_history.csv", index=False)
    figure, axes = _figure(run_dir.name)
    axes[0].set_title("Training (weighted EDL)")
    axes[1].set_title("Validation (unweighted EDL)")
    _plot_series(axes[0], history, "train_loss_epoch", "Train", color="tab:blue")
    for column, label in (
        ("val_bce_loss", "Total"),
        ("val_bce_clear_cloud", "Clear/cloud"),
        ("val_bce_land_water", "Land/water (checkpoint monitor)"),
    ):
        _plot_series(axes[1], history, column, label)
    return _save_figure(figure, axes, run_dir / "loss_curve.png")


def plot_ensemble_losses(run_dirs: list[Path], output_path: Path) -> Path | None:
    histories = []
    for run_dir in run_dirs:
        history = read_epoch_losses(run_dir / "loss_history.csv")
        if not history.empty:
            histories.append((run_dir.name, history))
    if not histories:
        print("No completed member loss histories available for an ensemble plot.")
        return None
    figure, axes = _figure(f"EDL ensemble loss curves ({len(histories)} members)")
    axes[0].set_title("Training (weighted EDL)")
    axes[1].set_title("Validation land/water (checkpoint monitor)")
    for index, (name, history) in enumerate(histories):
        label = f"Member {name.rsplit('_', 1)[-1]}"
        color = colormaps["tab10"](index % 10)
        _plot_series(axes[0], history, "train_loss_epoch", label, color=color)
        _plot_series(axes[1], history, "val_bce_land_water", label, color=color)
    return _save_figure(figure, axes, output_path)
