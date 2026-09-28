"""SAR input preprocessing shared by training and inference."""

from typing import Any

import numpy as np
import torch


def preprocess_sar(
    image: np.ndarray | torch.Tensor,
    config: Any,
) -> np.ndarray | torch.Tensor:
    """Apply KuroSiwo's dB handling, clamp, and channel normalization."""

    is_tensor = isinstance(image, torch.Tensor)
    values = image.float() if is_tensor else np.asarray(image, dtype=np.float32)
    clamp_input = config.get("clamp_input")
    mean = config.get("feature_mean", config.get("data_mean"))
    std = config.get("feature_std", config.get("data_std"))

    if is_tensor:
        if torch.any(values < 0):
            values = torch.pow(10.0, values / 10.0)
        if clamp_input is not None:
            values = values.clamp(0.0, float(clamp_input))
        if mean is not None and std is not None:
            values = (values - values.new_tensor(mean).view(-1, 1, 1)) / (
                values.new_tensor(std).view(-1, 1, 1) + 1e-6
            )
        return values

    if np.any(values < 0):
        values = np.power(10.0, values / 10.0)
    if clamp_input is not None:
        values = np.clip(values, 0.0, float(clamp_input))
    if mean is not None and std is not None:
        values = (values - np.asarray(mean, dtype=np.float32)[:, None, None]) / (
            np.asarray(std, dtype=np.float32)[:, None, None] + 1e-6
        )
    return values
