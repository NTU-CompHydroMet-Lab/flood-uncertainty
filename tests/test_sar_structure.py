import json

from flood_uncertainty.utils.config_loader import load_mode_config


def test_sar_config_uses_vv_vh_for_binary_land_water_task():
    config = load_mode_config("configurations/edl_sar.json", mode="train")

    assert config.model_params.hyperparameters.num_classes == 1
    assert config.model_params.hyperparameters.num_channels == 2
    assert config.model_params.hyperparameters.label_names == [
        ["invalid", "land", "water"]
    ]
    assert config.data_params.channel_configuration == "sar"
    assert config.data_params.channels == ["vv", "vh"]
    assert config.data_params.dem is False
    assert config.data_params.slope is False


def test_sar_wandb_is_enabled():
    config = load_mode_config("configurations/edl_sar.json", mode="train")

    assert config.wandb_enabled is True
    assert config.wandb_project == "flood_uncertainty_edl_sar"


def test_sar_config_is_valid_json():
    with open("configurations/edl_sar.json", encoding="utf-8") as config_file:
        config = json.load(config_file)

    assert set(config) == {"shared", "train", "infer", "validate_only"}
