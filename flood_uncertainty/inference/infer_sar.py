"""Inference helpers for the KuroSiwo SAR EDL model."""

from typing import Any

import numpy as np
import torch

from ml4floods.models.model_setup import get_pred_function

from flood_uncertainty.models.edl import EDL_SAR_Unet
from flood_uncertainty.preprocessing.sar import preprocess_sar


def load_sar_model(config: Any, weights_path: str | None = None) -> EDL_SAR_Unet:
    """Load an EDL SAR model from a Lightning checkpoint or state dict."""

    path = (
        weights_path
        or config.model_params.get("inference_weights")
        or config.model_params.get("pretrained_path")
    )
    if not path:
        raise ValueError("A SAR checkpoint path is required")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if config.model_params.hyperparameters.get("model_type") == "edl_terramind":
        from flood_uncertainty.models.edl_terramind import EDL_TerraMind_SAR

        # weights come from the checkpoint; do not fetch the HF TerraMind weights first
        config.model_params.hyperparameters["backbone_pretrained"] = False
        model = EDL_TerraMind_SAR(config.model_params, normalized_data=False)
    else:
        model = EDL_SAR_Unet(config.model_params, normalized_data=False)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def predict_sar(
    model: EDL_SAR_Unet,
    image: np.ndarray | torch.Tensor,
    data_config: Any,
    *,
    threshold: float = 0.5,
    max_tile_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Predict a SAR image and return class and EDL uncertainty products."""

    values = preprocess_sar(image, data_config)
    tensor = values if isinstance(values, torch.Tensor) else torch.from_numpy(values)
    return predict_sar_normalized(model, tensor, threshold=threshold, max_tile_size=max_tile_size)


def predict_sar_normalized(
    model: EDL_SAR_Unet,
    tensor: torch.Tensor,
    *,
    threshold: float = 0.5,
    max_tile_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Predict an already normalized (C, H, W) SAR tensor and return class and EDL uncertainty products."""

    if tensor.ndim != 3:
        raise ValueError(f"Expected (C, H, W) SAR input, got {tuple(tensor.shape)}")
    expected_channels = model.hparams["model_params"]["hyperparameters"]["num_channels"]
    if tensor.shape[0] != expected_channels:
        raise ValueError(
            f"SAR input has {tensor.shape[0]} channels, model expects {expected_channels}"
        )

    inference = get_pred_function(
        model,
        next(model.parameters()).device,
        module_shape=8,
        max_tile_size=max_tile_size,
        disable_pbar=True,
    )
    with torch.no_grad():
        logits = inference(tensor.unsqueeze(0))[0]
        output = model.edl_logits_to_output(logits.unsqueeze(0))
        probability = output["prob"][0, 0].cpu().numpy()

    classification = np.where(probability >= threshold, 2, 1).astype(np.uint8)
    return {
        "classification": classification,
        "probability": probability,
        "evidence": output["evidence"][0].cpu().numpy(),
        "dst_u": output["dst_u"][0, 0].cpu().numpy(),
        "aleatoric": output["aleatoric"][0, 0].cpu().numpy(),
        "epistemic": output["epistemic"][0, 0].cpu().numpy(),
    }
