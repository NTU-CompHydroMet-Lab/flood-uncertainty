import copy
import json
from typing import Any, Dict

from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from ml4floods.models.utils.configuration import AttrDict


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _normalize_and_validate(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    if "model_params" not in config_dict or "data_params" not in config_dict:
        raise ValueError("Config must include 'model_params' and 'data_params'")
    if "resume_from_checkpoint" not in config_dict:
        raise ValueError("Config must include 'resume_from_checkpoint'")
    if "gpus" not in config_dict:
        raise ValueError("Config must include 'gpus'")

    model_params = config_dict["model_params"]
    data_params = config_dict["data_params"]
    hyperparameters = model_params.setdefault("hyperparameters", {})

    model_channel_cfg = hyperparameters.get("channel_configuration")
    data_channel_cfg = data_params.get("channel_configuration")

    if model_channel_cfg and data_channel_cfg and model_channel_cfg != data_channel_cfg:
        raise ValueError(
            "Set the same channel configuration for model/data: "
            f"{model_channel_cfg} != {data_channel_cfg}"
        )

    channel_cfg = model_channel_cfg or data_channel_cfg
    if channel_cfg is None:
        raise ValueError("Config is missing 'channel_configuration'")
    if channel_cfg not in CHANNELS_CONFIGURATIONS:
        raise ValueError(f"Unknown channel configuration: {channel_cfg}")

    hyperparameters["channel_configuration"] = channel_cfg
    data_params["channel_configuration"] = channel_cfg

    num_channels = len(CHANNELS_CONFIGURATIONS[channel_cfg])
    if data_params.get("add_mndwi_input", False):
        num_channels += 1
    hyperparameters["num_channels"] = num_channels

    return config_dict


def load_mode_config(config_path: str, mode: str = "train") -> AttrDict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    if "shared" in raw:
        if mode not in raw:
            raise ValueError(f"Mode '{mode}' not found in config: {config_path}")
        merged = _deep_merge(raw["shared"], raw[mode])
    else:
        merged = raw

    normalized = _normalize_and_validate(merged)
    return AttrDict.from_nested_dicts(normalized)
