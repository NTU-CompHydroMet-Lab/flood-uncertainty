"""Run EDL SAR inference on a multiband GeoTIFF."""

import argparse

import numpy as np
import rasterio

from flood_uncertainty.inference.infer_sar import load_sar_model, predict_sar
from flood_uncertainty.utils.config_loader import load_mode_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tif")
    parser.add_argument("output_tif")
    parser.add_argument("--config", default="configurations/edl_sar.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_mode_config(args.config, mode="infer")
    model = load_sar_model(config, args.weights)
    with rasterio.open(args.input_tif) as source:
        image = source.read().astype(np.float32)
        result = predict_sar(
            model,
            image,
            config.data_params,
            threshold=args.threshold,
            max_tile_size=config.model_params.hyperparameters.max_tile_size,
        )
        profile = source.profile.copy()
        profile.update(count=7, dtype="float32")
        with rasterio.open(args.output_tif, "w", **profile) as target:
            target.write(result["classification"].astype(np.float32), 1)
            target.write(result["dst_u"].astype(np.float32), 2)
            target.write(result["probability"].astype(np.float32), 3)
            target.write(result["evidence"][0].astype(np.float32), 4)
            target.write(result["evidence"][1].astype(np.float32), 5)
            target.write(result["aleatoric"].astype(np.float32), 6)
            target.write(result["epistemic"].astype(np.float32), 7)


if __name__ == "__main__":
    main()
