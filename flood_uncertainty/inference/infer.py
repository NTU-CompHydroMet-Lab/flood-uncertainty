import glob
import os
import re
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from georeader import plot
from georeader.geotensor import GeoTensor
from georeader.save import save_cog
from ml4floods.data import create_gt
from ml4floods.data.worldfloods.configs import BANDS_L8, BANDS_S2
from ml4floods.models.model_setup import (
    get_channel_configuration_bands,
    get_model,
    get_model_inference_function,
)
from ml4floods.models.postprocess import get_pred_mask_v2
from ml4floods.models.utils.configuration import AttrDict

from flood_uncertainty.models.edl import EDL_ML4FloodsModel, EDL_SAR_Unet
from flood_uncertainty.utils.config_loader import load_mode_config


def find_best_checkpoint(checkpoint_dir: str) -> str:
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")

    meta = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
    for callback_key, callback_state in meta.get("callbacks", {}).items():
        if "ModelCheckpoint" not in str(callback_key):
            continue
        best_k = callback_state.get("best_k_models", {})
        if not best_k:
            break
        best_key = min(best_k, key=lambda p: best_k[p])
        best_score = best_k[best_key]
        best_path = os.path.join(checkpoint_dir, os.path.basename(best_key))
        print(f"  Best checkpoint: {os.path.basename(best_path)} (score={best_score:.4f})")
        return best_path

    def _get_step(path: str) -> int:
        matched = re.search(r"step=(\d+)", os.path.basename(path))
        return int(matched.group(1)) if matched else 0

    fallback = max(ckpt_files, key=_get_step)
    print(f"  [fallback] Using checkpoint with largest step: {os.path.basename(fallback)}")
    return fallback


def run_ensemble_inference(
    s2l89tensor: torch.Tensor,
    ensemble_dirs: list,
    config: AttrDict,
    channels: list,
    th_water: float = 0.5,
    th_brightness: float = create_gt.BRIGHTNESS_THRESHOLD,
    max_tile_size: int = 1024,
    apply_normalization: bool = True,
    collection_name: str = "S2",
    distinguish_flood_traces: bool = False,
):
    config["model_params"]["max_tile_size"] = max_tile_size

    mndwi_indexes = None
    if distinguish_flood_traces:
        if collection_name == "S2":
            band_names = [BANDS_S2[iband] for iband in channels]
            mndwi_indexes = [band_names.index(band) for band in ["B3", "B11"]]
        elif collection_name == "Landsat":
            band_names = [BANDS_L8[iband] for iband in channels]
            mndwi_indexes = [band_names.index(band) for band in ["B3", "B6"]]

    all_probs = []
    n_models = len(ensemble_dirs)
    for idx, checkpoint_dir in enumerate(ensemble_dirs):
        print(f"[{idx+1}/{n_models}] Loading model from {os.path.basename(checkpoint_dir)} ...")
        best_ckpt = find_best_checkpoint(checkpoint_dir)
        checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        config["model_params"]["test"] = False
        model = get_model(config.model_params)
        model.load_state_dict(state_dict)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

        config["model_params"]["test"] = True
        infer_fn = get_model_inference_function(
            model,
            config,
            apply_normalization=apply_normalization,
            activation="sigmoid",
            disable_pbar=True,
        )
        with torch.no_grad():
            prob = infer_fn(s2l89tensor.unsqueeze(0))[0]
        all_probs.append(prob.cpu())

        del model, infer_fn, checkpoint, state_dict
        torch.cuda.empty_cache()
        print(f"  done, prob shape: {prob.shape}")

    stack = torch.stack(all_probs, dim=0)
    mean_prob = stack.mean(dim=0)
    n, c, h, w = stack.shape
    all_probs_flat = stack.reshape(n * c, h, w)
    aleatoric_uncertainty = (1.0 / n) * (stack * (1.0 - stack)).sum(dim=0)
    epistemic_uncertainty = (1.0 / n) * ((stack - mean_prob.unsqueeze(0)) ** 2).sum(dim=0)

    land_water_cloud = get_pred_mask_v2(
        s2l89tensor.cpu(),
        mean_prob,
        channels_input=channels,
        th_water=th_water,
        th_brightness=th_brightness,
        collection_name=collection_name,
    )

    invalids = land_water_cloud == 0
    mean_prob[:, invalids] = -1
    all_probs_flat[:, invalids] = -1
    aleatoric_uncertainty[:, invalids] = -1
    epistemic_uncertainty[:, invalids] = -1

    if distinguish_flood_traces and mndwi_indexes is not None:
        s2mndwi = s2l89tensor[mndwi_indexes].float()
        mndwi = (s2mndwi[0] - s2mndwi[1]) / (s2mndwi[0] + s2mndwi[1] + 1e-6)
        land_water_cloud[(land_water_cloud == 2) & (mndwi < 0)] = 4

    return land_water_cloud, all_probs_flat, mean_prob, aleatoric_uncertainty, epistemic_uncertainty


def run_mcdropout_inference(
    s2l89tensor: torch.Tensor,
    checkpoint_dir: str,
    config: AttrDict,
    channels: list,
    n_samples: int = 10,
    th_water: float = 0.5,
    th_brightness: float = create_gt.BRIGHTNESS_THRESHOLD,
    max_tile_size: int = 1024,
    apply_normalization: bool = True,
    collection_name: str = "S2",
    distinguish_flood_traces: bool = False,
):
    config["model_params"]["max_tile_size"] = max_tile_size

    mndwi_indexes = None
    if distinguish_flood_traces:
        if collection_name == "S2":
            band_names = [BANDS_S2[iband] for iband in channels]
            mndwi_indexes = [band_names.index(band) for band in ["B3", "B11"]]
        elif collection_name == "Landsat":
            band_names = [BANDS_L8[iband] for iband in channels]
            mndwi_indexes = [band_names.index(band) for band in ["B3", "B6"]]

    print(f"Loading MC Dropout model from {os.path.basename(checkpoint_dir)} ...")
    best_ckpt = find_best_checkpoint(checkpoint_dir)
    checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    config["model_params"]["test"] = False
    model = get_model(config.model_params)
    model.load_state_dict(state_dict)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    config["model_params"]["test"] = True
    infer_fn = get_model_inference_function(
        model,
        config,
        apply_normalization=apply_normalization,
        activation="sigmoid",
        disable_pbar=True,
    )

    enabled_dropout = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()
            enabled_dropout += 1

    print(f"  MC Dropout mode enabled (n_samples={n_samples}, dropout_layers={enabled_dropout})")

    all_probs = []
    for idx in range(n_samples):
        with torch.no_grad():
            prob = infer_fn(s2l89tensor.unsqueeze(0))[0]
        all_probs.append(prob.cpu())
        print(f"  Sample [{idx+1}/{n_samples}] done, prob shape: {prob.shape}")

    del model, infer_fn, checkpoint, state_dict
    torch.cuda.empty_cache()

    stack = torch.stack(all_probs, dim=0)
    mean_prob = stack.mean(dim=0)
    k, c, h, w = stack.shape
    all_probs_flat = stack.reshape(k * c, h, w)
    aleatoric_uncertainty = (1.0 / k) * (stack * (1.0 - stack)).sum(dim=0)
    epistemic_uncertainty = (1.0 / k) * ((stack - mean_prob.unsqueeze(0)) ** 2).sum(dim=0)

    land_water_cloud = get_pred_mask_v2(
        s2l89tensor.cpu(),
        mean_prob,
        channels_input=channels,
        th_water=th_water,
        th_brightness=th_brightness,
        collection_name=collection_name,
    )

    invalids = land_water_cloud == 0
    mean_prob[:, invalids] = -1
    all_probs_flat[:, invalids] = -1
    aleatoric_uncertainty[:, invalids] = -1
    epistemic_uncertainty[:, invalids] = -1

    if distinguish_flood_traces and mndwi_indexes is not None:
        s2mndwi = s2l89tensor[mndwi_indexes].float()
        mndwi = (s2mndwi[0] - s2mndwi[1]) / (s2mndwi[0] + s2mndwi[1] + 1e-6)
        land_water_cloud[(land_water_cloud == 2) & (mndwi < 0)] = 4

    return land_water_cloud, all_probs_flat, mean_prob, aleatoric_uncertainty, epistemic_uncertainty


def load_model(config_path: str, model_type: str, collection_name: str = "S2", mode: str = "infer"):
    config = load_mode_config(config_path, mode=mode)

    if model_type == "EDL-SAR" or collection_name == "S1":
        channels = [0, 1]
    else:
        channel_configuration = config["data_params"]["channel_configuration"]
        channels = get_channel_configuration_bands(channel_configuration, collection_name=collection_name)
        print(f"channel configuration: {channel_configuration}")
    print(f"used channels: {channels}")

    weights_path = config.model_params.get("checkpoint_path")
    if weights_path is None:
        raise ValueError("checkpoint_path not found in config.model_params")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if model_type == "EDL":
        model = EDL_ML4FloodsModel(config.model_params)
    elif model_type == "EDL-SAR":
        model = EDL_SAR_Unet(config.model_params)
    elif model_type == "v2":
        # We load weights manually from `checkpoint_path`, so temporarily force train-mode
        # model construction to avoid auto-loading path assertions in get_model(test=True).
        original_train_flag = config.model_params.get("train", False)
        original_test_flag = config.model_params.get("test", False)
        config["model_params"]["train"] = True
        config["model_params"]["test"] = False
        model = get_model(config.model_params)
        config["model_params"]["train"] = original_train_flag
        config["model_params"]["test"] = original_test_flag
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(state_dict)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"loaded {model_type} model")
    return model, channels, config


def load_inference_function(
    model: torch.nn.Module,
    config: AttrDict,
    max_tile_size: int = 1024,
    th_water: float = 0.5,
    th_brightness: float = create_gt.BRIGHTNESS_THRESHOLD,
    apply_normalization: bool = True,
    used_EDL: bool = True,
    collection_name: str = "S2",
    disable_pbar: bool = True,
    distinguish_flood_traces: bool = False,
) -> Tuple[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], AttrDict]:
    config["model_params"]["max_tile_size"] = max_tile_size
    config["model_params"]["test"] = True
    activation_param = "None" if used_EDL else "sigmoid"
    inference_function = get_model_inference_function(
        model,
        config,
        apply_normalization=apply_normalization,
        activation=activation_param,
        disable_pbar=disable_pbar,
    )

    channels = get_channel_configuration_bands(
        config.data_params.channel_configuration, collection_name=collection_name
    )
    mndwi_indexes = None
    if distinguish_flood_traces:
        if collection_name == "S2":
            band_names = [BANDS_S2[iband] for iband in channels]
            mndwi_indexes = [band_names.index(band) for band in ["B3", "B11"]]
        elif collection_name == "Landsat":
            band_names = [BANDS_L8[iband] for iband in channels]
            mndwi_indexes = [band_names.index(band) for band in ["B3", "B6"]]

    def predict_fn(s2l89tensor: torch.Tensor):
        with torch.no_grad():
            logits = inference_function(s2l89tensor.unsqueeze(0))[0]
            if used_EDL:
                output = model.edl_logits_to_output(logits.unsqueeze(0))
                pred = output["prob"][0]
                dst_u = output["dst_u"][0]
                evidence = output["evidence"][0]
                aleatoric = output["aleatoric"][0]
                epistemic = output["epistemic"][0]
            else:
                pred = logits
                dst_u = evidence = aleatoric = epistemic = None

            land_water_cloud = get_pred_mask_v2(
                s2l89tensor,
                pred,
                channels_input=channels,
                th_water=th_water,
                th_brightness=th_brightness,
                collection_name=collection_name,
            )

            invalids = land_water_cloud == 0
            pred[0][invalids] = -1
            pred[1][invalids] = -1

            if distinguish_flood_traces and mndwi_indexes is not None:
                s2mndwi = s2l89tensor[mndwi_indexes, ...].float()
                mndwi = (s2mndwi[0] - s2mndwi[1]) / (s2mndwi[0] + s2mndwi[1] + 1e-6)
                land_water_cloud[(land_water_cloud == 2) & (mndwi < 0)] = 4

        return land_water_cloud, pred, dst_u, evidence, aleatoric, epistemic

    return predict_fn, config


def predict(input_tensor, channels=None, inference_function=None):
    if channels is None:
        channels = [1, 2, 3, 7, 11, 12]
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor[channels]
    torch_inputs = torch.tensor(np.nan_to_num(input_tensor))
    return inference_function(torch_inputs)


COLORS_PRED = np.array(
    [[0, 0, 0], [139, 64, 0], [0, 0, 240], [220, 220, 220], [60, 85, 92]],
    dtype=np.float32,
) / 255


def plot_prediction(s2rst_rgb, prediction_raster, subset, filename, output_dir="result"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    plot.show((s2rst_rgb / 3500).clip(0, 1), ax=ax[0], add_scalebar=True)
    ax[0].set_title(f"{subset}/{filename}")
    plot.plot_segmentation_mask(
        prediction_raster,
        COLORS_PRED,
        ax=ax[1],
        interpretation_array=["invalids", "land", "water", "cloud", "flood_trace"],
    )
    ax[1].set_title(f"{subset}/{filename} floodmap")
    output_path = f"{output_dir}/{filename}_prediction.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")
    return output_path


def save_prediction_tif(
    prediction,
    pred_prob,
    dst_uncertainty,
    evidence,
    aleatoric,
    epistemic,
    s2rst,
    filename,
    model_type,
    output_dir="result",
):
    if dst_uncertainty is not None:
        band_descriptions = [
            "Classification (0=invalid, 1=land, 2=water, 3=cloud, 4=flood_trace)",
            "Water_DST_Uncertainty",
            "Water_Probability",
            "Water_Evidence_Neg",
            "Water_Evidence_Pos",
            "Water_Aleatoric",
            "Water_Epistemic",
        ]
        output_data = np.stack(
            [
                prediction.numpy().astype(np.float32),
                dst_uncertainty[1].numpy(),
                pred_prob[1].numpy(),
                evidence[2].numpy(),
                evidence[3].numpy(),
                aleatoric[1].numpy(),
                epistemic[1].numpy(),
            ],
            axis=0,
        )
        output_path = f"{output_dir}/{filename}_output_{model_type}.tif"
    else:
        band_descriptions = [
            "Classification (0=invalid, 1=land, 2=water, 3=cloud, 4=flood_trace)",
            "Water_Probability",
        ]
        output_data = np.stack(
            [prediction.numpy().astype(np.float32), pred_prob[1].numpy()],
            axis=0,
        )
        output_path = f"{output_dir}/{filename}_prediction_{model_type}.tif"

    output_raster = GeoTensor(
        output_data,
        transform=s2rst.transform,
        fill_value_default=-1,
        crs=s2rst.crs,
    )
    save_cog(output_raster, output_path, descriptions=band_descriptions)
    return output_path, output_raster


def save_ensemble_tif(
    prediction: torch.Tensor,
    all_probs_flat: torch.Tensor,
    mean_prob: torch.Tensor,
    aleatoric_uncertainty: torch.Tensor,
    epistemic_uncertainty: torch.Tensor,
    s2rst,
    filename: str,
    n_models: int,
    output_dir: str = "result",
):
    band_descriptions = [
        "Classification (0=invalid, 1=land, 2=water, 3=cloud, 4=flood_trace)",
        "Mean_Land_Probability",
        "Mean_Water_Probability",
        "Aleatoric_Land",
        "Aleatoric_Water",
        "Epistemic_Land",
        "Epistemic_Water",
    ]
    for idx in range(n_models):
        band_descriptions += [f"Model{idx}_Land_Probability", f"Model{idx}_Water_Probability"]

    output_data = np.concatenate(
        [
            prediction.numpy().astype(np.float32)[np.newaxis],
            mean_prob.numpy(),
            aleatoric_uncertainty.numpy(),
            epistemic_uncertainty.numpy(),
            all_probs_flat.numpy(),
        ],
        axis=0,
    )
    output_raster = GeoTensor(
        output_data,
        transform=s2rst.transform,
        fill_value_default=-1,
        crs=s2rst.crs,
    )
    output_path = f"{output_dir}/{filename}_output_ensemble.tif"
    save_cog(output_raster, output_path, descriptions=band_descriptions)
    print(f"Ensemble output saved to: {output_path}  [{output_data.shape[0]} bands]")
    return output_path, output_raster
