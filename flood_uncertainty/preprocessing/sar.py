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

    if mean is not None and std is not None:
        if config.get("use_pre_event", False) and values.shape[0] == 3 * len(mean):
            mean = list(mean) * 3
            std = list(std) * 3
        if values.shape[0] != len(mean) or values.shape[0] != len(std):
            raise ValueError("SAR channel count does not match normalization statistics")
        if any(value <= 0 for value in std):
            raise ValueError("SAR normalization standard deviations must be positive")

    if is_tensor:
        if torch.any(values < 0):
            values = torch.pow(10.0, values / 10.0)
        if clamp_input is not None:
            values = values.clamp(0.0, float(clamp_input))
        if mean is not None and std is not None:
            values = (values - values.new_tensor(mean).view(-1, 1, 1)) / (
                values.new_tensor(std).view(-1, 1, 1)
            )
        return values

    if np.any(values < 0):
        values = np.power(10.0, values / 10.0)
    if clamp_input is not None:
        values = np.clip(values, 0.0, float(clamp_input))
    if mean is not None and std is not None:
        values = (values - np.asarray(mean, dtype=np.float32)[:, None, None]) / (
            np.asarray(std, dtype=np.float32)[:, None, None]
        )
    return values
