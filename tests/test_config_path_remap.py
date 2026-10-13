import json

import pytest

from flood_uncertainty.utils.config_loader import PATH_REMAP_ENV, load_mode_config


def _write_config(tmp_path):
    config = {
        "shared": {
            "experiment_name": "remap_test",
            "model_params": {
                "model_folder": "/home/NAS/house/u/repo/artifacts/models",
                "hyperparameters": {"channel_configuration": "all"},
            },
            "data_params": {
                "path_to_splits": "/home/NAS/house/u/data/worldfloods_v2",
                "input_folder": "S2",
            },
            "resume_from_checkpoint": False,
            "gpus": 0,
        },
        "train": {},
        "infer": {},
        "validate_only": {
            "model_params": {"zarr_save_path": "/home/NAS/house/u/repo/artifacts/zarr/x.zarr"}
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return str(path)


def test_no_remap_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(PATH_REMAP_ENV, raising=False)
    config = load_mode_config(_write_config(tmp_path), mode="train")
    assert config.data_params.path_to_splits == "/home/NAS/house/u/data/worldfloods_v2"


def test_remap_longest_prefix_and_path_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv(
        PATH_REMAP_ENV,
        "/home/NAS/house/u/repo=/work/u/repo; /home/NAS/house/u=/work/u/nas/ ;/home/NAS/hou=/nope",
    )
    config = load_mode_config(_write_config(tmp_path), mode="validate_only")
    assert config.model_params.model_folder == "/work/u/repo/artifacts/models"
    assert config.data_params.path_to_splits == "/work/u/nas/data/worldfloods_v2"
    assert config.model_params.zarr_save_path == "/work/u/repo/artifacts/zarr/x.zarr"
    assert config.data_params.input_folder == "S2"


def test_remap_rejects_malformed_entry(tmp_path, monkeypatch):
    monkeypatch.setenv(PATH_REMAP_ENV, "/home/NAS")
    with pytest.raises(ValueError):
        load_mode_config(_write_config(tmp_path), mode="train")
