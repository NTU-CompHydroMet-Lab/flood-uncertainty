"""Inference helpers for the TerraMind S1 SAR EDL model (KuroSiwo or Sen1Floods11 fine-tune)."""

from typing import Any

import numpy as np
import torch

from flood_uncertainty.inference.infer_sar import predict_sar_normalized
from flood_uncertainty.models.edl_terramind import EDL_TerraMind_SAR
from flood_uncertainty.preprocessing.sar import preprocess_sar

# Sen1Floods11 official SAR normalisation (same values as the Sen1Floods11 loader)
SEN1_SAR_MEAN = [0.6851, 0.5235]
SEN1_SAR_STD = [0.0820, 0.1102]


def check_s1_input_matches_data(config: Any) -> None:
    """Make sure ``backbone_s1_input`` describes the same normalisation that inference applies."""

    s1 = config.model_params.hyperparameters.get("backbone_s1_input")
    if not s1:
        raise ValueError("TerraMind SAR inference requires backbone_s1_input in the config")
    dp = config.data_params
    dataset_type = dp.get("dataset_type", "kurosiwo")
    if dataset_type == "kurosiwo":
        expected_units, mean, std = "linear_zscore", list(dp.data_mean), list(dp.data_std)
    elif dataset_type == "sen1floods11":
        if dp.get("sar_units", "db") != "db":
            raise ValueError("Official Sen1 SAR normalization requires dB input")
        expected_units, mean, std = "sen1_scaled_db_zscore", SEN1_SAR_MEAN, SEN1_SAR_STD
    else:
        raise ValueError(f"Unknown SAR dataset_type: {dataset_type}")
    if s1.get("units") != expected_units or not (
        np.allclose(s1["mean"], mean) and np.allclose(s1["std"], std)
    ):
        raise ValueError(
            f"backbone_s1_input {s1} does not match {dataset_type} input ({expected_units}, {mean}, {std})"
        )


def load_terramind_sar_model(config: Any, weights_path: str | None = None) -> EDL_TerraMind_SAR:
    """Load a TerraMind SAR EDL model from a Lightning checkpoint or state dict.

    The checkpoint is ``weights_path``, else ``model_params.inference_weights``. ``pretrained_path``
    is never used: in a fine-tune config it is the initialisation checkpoint (e.g. the KuroSiwo model
    for ``tm_s1grd_kuro_s1f11``), not the trained model.
    """

    weights_path = weights_path or config.model_params.get("inference_weights")
    if not weights_path:
        raise ValueError("A TerraMind SAR checkpoint is required (--weights or model_params.inference_weights)")
    hp = config.model_params.hyperparameters
    if hp.get("model_type") != "edl_terramind":
        raise ValueError(f"Expected model_type edl_terramind, got {hp.get('model_type')}")
    if list(hp.get("backbone_modalities", ["S1GRD"])) != ["S1GRD"]:
        raise ValueError(f"Expected backbone_modalities ['S1GRD'], got {hp.get('backbone_modalities')}")
    check_s1_input_matches_data(config)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # weights come from the checkpoint; do not fetch the HF TerraMind weights first
    hp["backbone_pretrained"] = False
    model = EDL_TerraMind_SAR(config.model_params, normalized_data=False)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def preprocess_terramind_sar(image: np.ndarray, data_config: Any) -> np.ndarray:
    """Normalise post-event VV/VH the same way as the training loader of ``dataset_type``."""

    values = np.asarray(image, dtype=np.float32)
    if data_config.get("dataset_type", "kurosiwo") == "sen1floods11":
        # Official Train.ipynb: fill NaNs, clip dB, scale, then normalize VV/VH.
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = (np.clip(values, -50.0, 1.0) + 50.0) / 51.0
        mean = np.asarray(SEN1_SAR_MEAN, dtype=np.float32)[:, None, None]
        std = np.asarray(SEN1_SAR_STD, dtype=np.float32)[:, None, None]
        return (values - mean) / std
    return preprocess_sar(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), data_config)


def predict_terramind_sar(
    model: EDL_TerraMind_SAR,
    image: np.ndarray,
    data_config: Any,
    *,
    threshold: float = 0.5,
    max_tile_size: int = 224,
) -> dict[str, np.ndarray]:
    """Predict a (2, H, W) VV/VH image and return class and EDL uncertainty products.

    Pixels with a non-finite input band get class 0 and NaN uncertainty (the loaders mark them invalid).
    """

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError(f"Expected (2, H, W) VV/VH input, got {values.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(preprocess_terramind_sar(values, data_config)))
    result = predict_sar_normalized(model, tensor, threshold=threshold, max_tile_size=max_tile_size)

    invalid = ~np.isfinite(values).all(axis=0)
    result["classification"][invalid] = 0
    for key in ("dst_u", "probability", "aleatoric", "epistemic"):
        result[key][invalid] = np.nan
    result["evidence"][:, invalid] = np.nan
    return result
