"""Run TerraMind S1 SAR EDL inference on a VV/VH GeoTIFF (KuroSiwo or Sen1Floods11 model)."""

import argparse

import numpy as np
import rasterio

from flood_uncertainty.inference.infer_terramind_sar import (
    load_terramind_sar_model,
    predict_terramind_sar,
)
from flood_uncertainty.utils.config_loader import load_mode_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tif", help="2-band VV/VH TIFF, or VV only together with --vh")
    parser.add_argument("output_tif")
    parser.add_argument("--config", default="configurations/terramind_sar/tm_s1grd.json")
    parser.add_argument(
        "--weights", default=None,
        help="Trained TerraMind SAR checkpoint (default: model_params.inference_weights in --config)",
    )
    parser.add_argument("--vh", default=None, help="VH TIFF when input_tif contains only VV")
    parser.add_argument("--valid-mask", default=None, help="KuroSiwo MK0_MNA TIFF (1=valid)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_mode_config(args.config, mode="infer")
    model = load_terramind_sar_model(config, args.weights)
    with rasterio.open(args.input_tif) as source:
        def read_aligned_band(path: str) -> np.ndarray:
            with rasterio.open(path) as other:
                if (
                    other.count != 1
                    or source.shape != other.shape
                    or source.transform != other.transform
                    or source.crs != other.crs
                ):
                    raise ValueError(f"SAR bands must be aligned single-band TIFFs: {path}")
                return other.read(1)

        if args.vh:
            image = np.stack((source.read(1), read_aligned_band(args.vh))).astype(np.float32)
        else:
            # masked read: nodata pixels become NaN and are marked invalid
            image = source.read([1, 2], masked=True).astype(np.float32).filled(np.nan)
        result = predict_terramind_sar(
            model,
            image,
            config.data_params,
            threshold=args.threshold,
            max_tile_size=config.model_params.hyperparameters.get("max_tile_size", 224),
        )
        if args.valid_mask:
            valid = read_aligned_band(args.valid_mask) == 1
            result["classification"][~valid] = 0
            for key in ("dst_u", "probability", "aleatoric", "epistemic"):
                result[key][~valid] = np.nan
            result["evidence"][:, ~valid] = np.nan
        profile = source.profile.copy()
        profile.update(count=7, dtype="float32", nodata=None)
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
