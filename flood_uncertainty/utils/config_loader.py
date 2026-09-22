import copy
import json
from typing import Any, Dict

from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from ml4floods.models.utils.configuration import AttrDict

SUPPORTED_MODES = {"train", "infer", "validate_only"}
REQUIRED_TOP_LEVEL_BLOCKS = {"shared", "train", "infer", "validate_only"}
LEGACY_MODE_FLAGS = {"train", "test", "val_only"}
RUNTIME_MODE_FLAGS = {
    "train": {"train": True, "test": False, "val_only": False},
    "infer": {"train": False, "test": True, "val_only": False},
    "validate_only": {"train": False, "test": True, "val_only": True},
}


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

    if channel_cfg == "sar":
        num_channels = data_params.get(
            "num_channels", hyperparameters.get("num_channels", 2)
        )
        if not isinstance(num_channels, int) or num_channels < 2:
            raise ValueError("SAR num_channels must be an integer >= 2")
    else:
        num_channels = len(CHANNELS_CONFIGURATIONS[channel_cfg])
        if data_params.get("add_mndwi_input", False):
            num_channels += 1
    hyperparameters["num_channels"] = num_channels

    return config_dict


def _validate_top_level_schema(raw: Dict[str, Any], config_path: str) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_BLOCKS - set(raw.keys()))
    if missing:
        raise ValueError(
            f"Config must contain top-level blocks {sorted(REQUIRED_TOP_LEVEL_BLOCKS)}; "
            f"missing: {missing} ({config_path})"
        )

    for block_name in REQUIRED_TOP_LEVEL_BLOCKS:
        if not isinstance(raw.get(block_name), dict):
            raise ValueError(f"Config block '{block_name}' must be a JSON object: {config_path}")


def _validate_no_legacy_mode_flags(raw: Dict[str, Any], config_path: str) -> None:
    for block_name in REQUIRED_TOP_LEVEL_BLOCKS:
        model_params = raw[block_name].get("model_params")
        if not isinstance(model_params, dict):
            continue
        legacy_keys = sorted(LEGACY_MODE_FLAGS.intersection(model_params.keys()))
        if legacy_keys:
            raise ValueError(
                f"Remove legacy flags {legacy_keys} from '{block_name}.model_params' in {config_path}; "
                "mode flags are runtime-only."
            )


def _apply_runtime_mode_flags(config_dict: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode not in RUNTIME_MODE_FLAGS:
        raise ValueError(f"Unsupported mode: {mode}")

    model_params = config_dict.setdefault("model_params", {})
    for key, value in RUNTIME_MODE_FLAGS[mode].items():
        model_params[key] = value
    config_dict["runtime_mode"] = mode
    return config_dict


def load_mode_config(config_path: str, mode: str = "train") -> AttrDict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(SUPPORTED_MODES)}")

    _validate_top_level_schema(raw, config_path)
    _validate_no_legacy_mode_flags(raw, config_path)

    merged = _deep_merge(raw["shared"], raw[mode])
    normalized = _normalize_and_validate(merged)
    with_runtime_flags = _apply_runtime_mode_flags(normalized, mode)
    return AttrDict.from_nested_dicts(with_runtime_flags)
