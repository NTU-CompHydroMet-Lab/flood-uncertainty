"""SAR result plots independent of model inference and optical RGB readers."""

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def plot_sar_result(
    vv,
    vh,
    target,
    probability,
    output_path: str | Path,
    *,
    input_scale: str,
    threshold: float = 0.5,
    uncertainty=None,
    uncertainty_label: str = "Uncertainty",
    db_limits: tuple[float, float] = (-30.0, 0.0),
    title: str = "SAR result",
    dpi: int = 150,
) -> dict:
    """Save a six-panel PNG and return per-image metrics and confusion map.

    Inputs are aligned, nonempty (H, W) NumPy arrays or torch tensors (CPU/GPU).
    target uses the project's *mapped* labels: 1=land, 2=water; all other
    labels are ignored. Do not pass raw KuroSiwo labels. probability is the
    water probability, not logits; values on valid targets must be finite
    and within [0, 1]. Prediction uses >= threshold, as in eval_sar.py.

    input_scale must be 'linear', 'db', or 'normalized'. Linear power is
    converted to dB for display (nonpositive values masked); dB is displayed
    as supplied, using db_limits for both channels. Normalized inputs are
    shown in their supplied units with a shared 2nd/98th percentile stretch,
    never relabeled as physical backscatter. No preprocessing is applied to
    model inputs and no input array is mutated.

    Panels: VV, VH, target, prediction, spatial confusion, and optional
    uncertainty (water probability when omitted). Invalid target pixels are
    gray in result panels. Uncertainty retains its supplied numeric units.
    cm_map: 0=invalid, 1=TN, 2=FP, 3=FN, 4=TP (same colors as optical).
    Undefined ratios are None; mIoU averages only classes with nonzero union
    and is None if there are no valid pixels. This is per-image mIoU, not
    a replacement for test-set metrics computed from aggregate counts.
    """
    def array(value, name):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.asarray(value)
        if result.ndim != 2 or not result.size:
            raise ValueError(f"{name} must be a nonempty (H, W) array")
        return result

    vv, vh, target, probability = [
        array(value, name) for value, name in
        [(vv, "vv"), (vh, "vh"), (target, "target"), (probability, "probability")]
    ]
    if any(value.shape != target.shape for value in (vv, vh, probability)):
        raise ValueError("All inputs must have the same spatial shape")
    if input_scale not in {"linear", "db", "normalized"}:
        raise ValueError("input_scale must be linear, db, or normalized")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    if len(db_limits) != 2 or not np.all(np.isfinite(db_limits)) or db_limits[0] >= db_limits[1]:
        raise ValueError("db_limits must be two finite increasing values")
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("output_path must end in .png")
    valid = (target == 1) | (target == 2)
    p = probability[valid]
    if not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("Water probabilities on valid pixels must be finite and within [0, 1]")
    if uncertainty is not None:
        uncertainty = array(uncertainty, "uncertainty")
        if uncertainty.shape != target.shape:
            raise ValueError("uncertainty must have the same spatial shape")

    water = target == 2
    predicted = probability >= threshold
    cm_map = np.zeros(target.shape, dtype=np.uint8)
    cm_map[valid & ~water & ~predicted] = 1
    cm_map[valid & ~water & predicted] = 2
    cm_map[valid & water & ~predicted] = 3
    cm_map[valid & water & predicted] = 4
    tn, fp, fn, tp = [int(np.count_nonzero(cm_map == code)) for code in range(1, 5)]

    def ratio(a, b):
        return a / b if b else None

    water_iou = ratio(tp, tp + fp + fn)
    nonwater_iou = ratio(tn, tn + fp + fn)
    defined = [v for v in (water_iou, nonwater_iou) if v is not None]
    metrics = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "valid_pixels": int(valid.sum()), "invalid_pixels_excluded": int((~valid).sum()),
        "threshold": threshold,
        "water_iou": water_iou, "nonwater_iou": nonwater_iou,
        "miou": float(np.mean(defined)) if defined else None,
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }

    channels = np.stack((vv, vh)).astype(float)
    if input_scale == "linear":
        positive = np.isfinite(channels) & (channels > 0)
        channels = 10 * np.log10(np.where(positive, channels, np.nan))
    limits = db_limits
    units = "dB"
    if input_scale == "normalized":
        finite = channels[np.isfinite(channels)]
        limits = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
        if limits[0] == limits[1]:
            limits = (limits[0] - 0.5, limits[1] + 0.5)
        units = "normalized units"

    fig = Figure(figsize=(15, 9), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    gray = colormaps["gray"].copy()
    gray.set_bad("#808080")
    class_colors = ["#808080", "black", "#2474ed"]
    cm_colors = ["#808080", "black", "red", "orange", "green"]
    try:
        for ax, channel, name in zip(axes[0, :2], channels, ("VV", "VH")):
            artist = ax.imshow(np.ma.masked_invalid(channel), cmap=gray,
                               vmin=limits[0], vmax=limits[1], interpolation="nearest")
            ax.set_title(f"{name} ({units})")
            fig.colorbar(artist, ax=ax, shrink=0.75)
        for ax, values, name in (
            (axes[0, 2], np.where(valid, target, 0), "Ground truth"),
            (axes[1, 0], np.where(valid, np.where(predicted, 2, 1), 0),
             f"Prediction (water probability >= {threshold:g})"),
        ):
            ax.imshow(values, cmap=ListedColormap(class_colors), vmin=-0.5, vmax=2.5,
                      interpolation="nearest")
            ax.set_title(name)
            ax.legend(handles=[Patch(color=c, label=l) for c, l in
                               zip(class_colors, ("Invalid", "Land", "Water"))],
                      loc="upper right", fontsize=8)
        axes[1, 1].imshow(cm_map, cmap=ListedColormap(cm_colors), vmin=-0.5, vmax=4.5,
                          interpolation="nearest")
        axes[1, 1].set_title("Spatial confusion")
        axes[1, 1].legend(handles=[Patch(color=c, label=l) for c, l in
                                  zip(cm_colors, ("Invalid", "TN", "FP", "FN", "TP"))],
                         loc="upper right", fontsize=8)
        values = probability if uncertainty is None else uncertainty
        cmap = colormaps["viridis"].copy()
        cmap.set_bad("#808080")
        artist = axes[1, 2].imshow(
            np.ma.masked_where(~valid | ~np.isfinite(values), values), cmap=cmap,
            interpolation="nearest", **({"vmin": 0, "vmax": 1} if uncertainty is None else {}))
        axes[1, 2].set_title("Water probability" if uncertainty is None else uncertainty_label)
        fig.colorbar(artist, ax=axes[1, 2], shrink=0.75)
        for ax in axes.flat:
            ax.set_axis_off()
        def formatted(value):
            return "N/A" if value is None else f"{value:.4f}"
        fig.suptitle(
            f"{title}\nWater IoU={formatted(water_iou)} | "
            f"Land IoU={formatted(nonwater_iou)} | mIoU={formatted(metrics['miou'])}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, format="png")
    finally:
        fig.clear()
    return {"metrics": metrics, "cm_map": cm_map, "png_path": str(output_path)}
