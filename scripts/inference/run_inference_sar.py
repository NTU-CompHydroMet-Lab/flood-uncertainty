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
    parser.add_argument("--vh", default=None, help="VH TIFF when input_tif contains only VV")
    parser.add_argument("--pre1-vv", default=None)
    parser.add_argument("--pre1-vh", default=None)
    parser.add_argument("--pre2-vv", default=None)
    parser.add_argument("--pre2-vh", default=None)
    parser.add_argument("--valid-mask", default=None, help="KuroSiwo MK0_MNA TIFF (1=valid)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    pre_paths = (args.pre1_vv, args.pre1_vh, args.pre2_vv, args.pre2_vh)
    if any(pre_paths) and (not all(pre_paths) or not args.vh):
        parser.error("pre-event inference requires --vh and all four pre-event bands")

    config = load_mode_config(args.config, mode="infer")
    model = load_sar_model(config, args.weights)
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
            image = source.read().astype(np.float32)
        if all(pre_paths):
            if image.shape[0] != 2:
                raise ValueError("Pre-event inputs require single-band post-event VV and VH")
            image = np.concatenate((image, np.stack([read_aligned_band(p) for p in pre_paths])))
        result = predict_sar(
            model,
            image,
            config.data_params,
            threshold=args.threshold,
            max_tile_size=config.model_params.hyperparameters.max_tile_size,
        )
        if args.valid_mask:
            with rasterio.open(args.valid_mask) as mask_source:
                if (
                    mask_source.count != 1
                    or mask_source.shape != source.shape
                    or mask_source.transform != source.transform
                    or mask_source.crs != source.crs
                ):
                    raise ValueError("Valid mask must align with the SAR TIFFs")
                valid = mask_source.read(1) == 1
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
