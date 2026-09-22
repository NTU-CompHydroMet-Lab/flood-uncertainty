import json

from flood_uncertainty.utils.config_loader import load_mode_config


def test_sar_config_uses_single_task_and_four_input_channels():
    config = load_mode_config("configurations/edl_sar.json", mode="train")

    assert config.model_params.hyperparameters.num_classes == 1
    assert config.model_params.hyperparameters.num_channels == 4
    assert config.data_params.channel_configuration == "sar"


def test_sar_config_is_valid_json():
    with open("configurations/edl_sar.json", encoding="utf-8") as config_file:
        config = json.load(config_file)

    assert set(config) == {"shared", "train", "infer", "validate_only"}
