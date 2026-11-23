import json
from pathlib import Path

import pytest
import torch

from flood_uncertainty.utils.config_loader import load_mode_config
from scripts.train.train_edl_ensemble import (
    HEAD_KEYS,
    find_pretraining_checkpoint,
    member_config,
    validate_pretraining_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def test_selection_uses_surviving_scores_across_old_metadata(tmp_path):
    old = tmp_path / "epoch=1-step=20.ckpt"
    best = tmp_path / "epoch=3-step=40.ckpt"
    latest = tmp_path / "epoch=9-step=100.ckpt"
    torch.save({"callbacks": {"ModelCheckpoint": {
        "monitor": "val_bce_land_water",
        "best_k_models": {"/old/mount/deleted.ckpt": 0.01, f"/old/mount/{old.name}": 0.5},
    }}}, old)
    torch.save({"callbacks": {"ModelCheckpoint": {
        "monitor": "val_bce_land_water",
        "best_k_models": {f"/old/mount/{best.name}": 0.2, f"/old/mount/{old.name}": 0.5},
    }}}, best)
    torch.save({"callbacks": {"ModelCheckpoint": {
        "monitor": "another_metric",
        "best_k_models": {f"/old/mount/{latest.name}": 0.001},
    }}}, latest)

    checkpoint, score = find_pretraining_checkpoint(tmp_path)

    assert checkpoint == best
    assert score == pytest.approx(0.2)


def test_selection_falls_back_to_numeric_step_without_scores(tmp_path):
    for name in ("epoch=0-step=9.ckpt", "epoch=1-step=100.ckpt"):
        torch.save({"state_dict": {}}, tmp_path / name)
    checkpoint, score = find_pretraining_checkpoint(tmp_path)
    assert checkpoint.name == "epoch=1-step=100.ckpt"
    assert score is None


def test_selection_rejects_missing_members(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .ckpt files"):
        find_pretraining_checkpoint(tmp_path)


@pytest.fixture
def transfer_fixture():
    source = {
        "network.conv1.weight": torch.zeros(64, 6, 3, 3),
        "network.conv_last.weight": torch.zeros(2, 64, 1, 1),
        "network.conv_last.bias": torch.zeros(2),
    }
    expected = {
        "network.conv1.weight": (64, 6, 3, 3),
        "network.conv_last.weight": (4, 64, 1, 1),
        "network.conv_last.bias": (4,),
    }
    return source, expected


def test_transfer_accepts_only_the_output_head_size_change(tmp_path, transfer_fixture):
    source, expected = transfer_fixture
    path = tmp_path / "member.ckpt"
    torch.save({"state_dict": source, "epoch": 8, "global_step": 18450}, path)

    report = validate_pretraining_checkpoint(path, expected, "bgriswirs")

    assert report["source_epoch"] == 8
    assert report["loaded_parameter_tensors"] == 1
    assert set(report["randomly_initialized_parameters"]) == HEAD_KEYS


@pytest.mark.parametrize("problem", ["missing_backbone", "wrong_backbone", "edl_head", "bands"])
def test_transfer_rejects_incompatible_sources(tmp_path, transfer_fixture, problem):
    source, expected = transfer_fixture
    channels = "bgriswirs"
    if problem == "missing_backbone":
        source.pop("network.conv1.weight")
    if problem == "wrong_backbone":
        source["network.conv1.weight"] = torch.zeros(64, 13, 3, 3)
    if problem == "edl_head":
        source["network.conv_last.weight"] = torch.zeros(4, 64, 1, 1)
    if problem == "bands":
        channels = "other_six_bands"
    path = tmp_path / "member.ckpt"
    torch.save({
        "state_dict": source,
        "hyper_parameters": {"model_params": {"hyperparameters": {
            "channel_configuration": channels,
        }}},
    }, path)

    with pytest.raises(ValueError):
        validate_pretraining_checkpoint(path, expected, "bgriswirs")


@pytest.mark.parametrize("base_seed, member_offset", [(42, 0), (7, 10)])
def test_member_configs_round_trip_with_independent_seeds_and_weights(
    tmp_path, base_seed, member_offset
):
    base = load_mode_config(str(ROOT / "configurations/edl_ensemble.json"), "train")
    base.seed = base_seed
    base.pop("member_index_offset", None)
    if member_offset:
        base.member_index_offset = member_offset
    assert len(base.model_params.ensemble_dirs) == 10
    for member in range(10):
        checkpoint = tmp_path / f"source_{member}.ckpt"
        config = member_config(base, member, checkpoint, tmp_path / "data")
        config_path = tmp_path / f"member_{member}.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        restored = load_mode_config(str(config_path), "train")
        assert restored.seed == base_seed + member
        assert restored.experiment_name == f"edl_from_ensemble_{member_offset + member}"
        assert "member_index_offset" not in restored
        assert restored.model_params.pretrained_path == str(checkpoint)
        assert restored.model_params.train is True
        assert restored.model_params.val_only is False
        assert restored.data_params.path_to_splits == str(tmp_path / "data")
    assert base.seed == base_seed
    assert base.get("member_index_offset", 0) == member_offset
    assert base.model_params.pretrained_path is None
    assert len(base.model_params.ensemble_dirs) == 10
