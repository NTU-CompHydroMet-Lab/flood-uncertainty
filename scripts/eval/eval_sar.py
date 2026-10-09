"""Evaluate a SAR EDL checkpoint on the configured held-out test split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from flood_uncertainty.data.kurosiwo import create_kurosiwo_loaders
from flood_uncertainty.data.sen1floods11 import create_sen1floods11_loaders
from flood_uncertainty.inference.infer_sar import load_sar_model
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.utils.config_loader import load_mode_config


def main(default_config="configurations/edl_sar.json",
         default_output="artifacts/results/sar_test/metrics.json", default_plot=True) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--plot-png", action=argparse.BooleanOptionalAction, default=default_plot,
                        help="Save result PNGs (enabled by default; --no-plot-png disables)")
    parser.add_argument("--max-plots", type=int, default=16,
                        help="Maximum number of tiles to plot; does not limit evaluation")
    parser.add_argument("--min-water-fraction", type=float, default=0.1,
                        help="Minimum GT water / valid pixel fraction for plots (default: 0.1)")
    parser.add_argument("--plot-dir", default=None,
                        help="Defaults to <output parent>/<output stem>_plots")
    parser.add_argument("--plot-input-scale", choices=["normalized", "linear", "db"],
                        default=None, help="Display units; inferred for normalize loader mode")
    args = parser.parse_args()
    if not 0 <= args.min_water_fraction <= 1:
        parser.error("min water fraction must be within [0, 1]")
    if args.max_plots < 0:
        parser.error("max plots cannot be negative")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and num workers cannot be negative")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max batches must be positive")

    config = load_mode_config(args.config, mode="train")
    config.data_params.batch_size = args.batch_size
    config.data_params.num_workers = args.num_workers
    output = Path(args.output)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output.parent / f"{output.stem}_plots"
    plot_enabled = args.plot_png and args.max_plots > 0
    dataset_type = config.data_params.get("dataset_type", "kurosiwo")
    optical = config.data_params.get("modality") == "optical"
    plot_scale = args.plot_input_scale
    if plot_enabled and optical:
        from flood_uncertainty.visualization.optical import plot_optical_result
        plot_scale = "rgb"
    if plot_enabled and not optical:
        from flood_uncertainty.visualization.sar import plot_sar_result

        if plot_scale is None:
            if dataset_type == "sen1floods11" or config.data_params.get("scale_input") == "normalize":
                plot_scale = "normalized"
            else:
                parser.error("Specify --plot-input-scale for a loader not using normalize")
        channels = [c.lower() for c in config.data_params.get("channels", ["vv", "vh"] if dataset_type == "sen1floods11" else [])]
        if "vv" not in channels or "vh" not in channels:
            parser.error("SAR result plots require VV and VH channels")
        vv_index, vh_index = channels.index("vv"), channels.index("vh")
    plots_written = evaluated_tiles = 0
    plotted_tiles = []
    if dataset_type == "sen1floods11":
        if optical and config.model_params.hyperparameters.get("task_mode") != "water_only":
            parser.error("Sen1 optical evaluation requires water_only")
        _, _, test_loader = create_sen1floods11_loaders(config.data_params)
    elif dataset_type == "kurosiwo":
        _, _, test_loader = create_kurosiwo_loaders(config.data_params)
    else:
        parser.error(f"Unknown SAR dataset_type: {dataset_type}")
    if optical:
        model = EDL_ML4FloodsModel(config.model_params, normalized_data=True)
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    else:
        model = load_sar_model(config, args.weights)
    device = next(model.parameters()).device

    tp = fp = fn = tn = valid_pixels = invalid_pixels = brier_sum = batches = 0
    with torch.inference_mode():
        for batch in test_loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)[:, 1 if optical else 0]
            logits = model.network(image)
            probability = model.edl_logits_to_probs(logits)[:, 0]
            valid = (target == 1) | (target == 2)
            water = target == 2
            predicted = probability >= 0.5
            tp += int((predicted & water & valid).sum().item())
            fp += int((predicted & ~water & valid).sum().item())
            fn += int((~predicted & water & valid).sum().item())
            tn += int((~predicted & ~water & valid).sum().item())
            valid_pixels += int(valid.sum().item())
            invalid_pixels += int((~valid).sum().item())
            brier_sum += float((probability[valid] - water[valid].float()).square().sum().item())
            if plot_enabled and plots_written < args.max_plots:
                # Reuse GT masks; selecting plots never filters aggregate metrics.
                valid_counts = valid.flatten(1).sum(1)
                water_counts = water.flatten(1).sum(1)
                fractions = water_counts.double() / valid_counts.clamp_min(1)
                selected = torch.nonzero(
                    (valid_counts > 0) & (fractions >= args.min_water_fraction),
                    as_tuple=True,
                )[0][:args.max_plots - plots_written]
                uncertainty = (
                    model.edl_logits_to_output(logits[selected])["dst_u"][:, 0]
                    if selected.numel() else None
                )
                for j, i in enumerate(selected.tolist()):
                    tile_number = evaluated_tiles + i
                    if optical:
                        plot_optical_result(
                            image[i], target[i], probability[i],
                            plot_dir / f"tile_{tile_number:06d}.png",
                            channel_configuration=config.data_params.channel_configuration,
                            uncertainty=uncertainty[j],
                            title=f"Optical test tile {tile_number:06d} | GT water={fractions[i].item():.1%}",
                        )
                    else:
                        plot_sar_result(
                            image[i, vv_index], image[i, vh_index], target[i], probability[i],
                            plot_dir / f"tile_{tile_number:06d}.png",
                            input_scale=plot_scale, uncertainty=uncertainty[j],
                            uncertainty_label="EDL DST uncertainty",
                            title=f"SAR test tile {tile_number:06d} | GT water={fractions[i].item():.1%}",
                        )
                    plotted_tiles.append({
                        "tile_index": tile_number,
                        "water_fraction": fractions[i].item(),
                        "png_path": str(plot_dir / f"tile_{tile_number:06d}.png"),
                    })
                    plots_written += 1
            evaluated_tiles += image.shape[0]
            batches += 1
            if args.max_batches is not None and batches >= args.max_batches:
                break

    def ratio(numerator: int | float, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    class_ious = [ratio(tp, tp + fp + fn), ratio(tn, tn + fp + fn)]
    defined_ious = [value for value in class_ious if value is not None]
    result = {
        "checkpoint": str(Path(args.weights).resolve()),
        "split": "test",
        "modality": "optical" if optical else "sar",
        "confusion_matrix_pred_rows_gt_columns": [[tn, fn], [fp, tp]],
        "threshold": 0.5,
        "evaluated_batches": batches,
        "evaluated_tiles": evaluated_tiles,
        "plots_written": plots_written,
        "min_water_fraction": args.min_water_fraction,
        "plotted_tiles": plotted_tiles,
        "plot_dir": str(plot_dir.resolve()) if plots_written else None,
        "plot_input_scale": plot_scale if plots_written else None,
        "total_test_tiles": len(test_loader.dataset),
        "valid_pixels": valid_pixels,
        "invalid_pixels_excluded": invalid_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "water_iou": class_ious[0],
        "nonwater_iou": class_ious[1],
        "miou": sum(defined_ious) / len(defined_ious) if defined_ious else None,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": ratio(tp + tn, valid_pixels),
        "brier": ratio(brier_sum, valid_pixels),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
