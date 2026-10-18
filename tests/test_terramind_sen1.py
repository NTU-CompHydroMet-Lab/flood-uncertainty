"""Sen1Floods11 TerraMind fine-tune: config checks, stage-1 weight loading, paired windows (random init, CPU)."""

import importlib.util

import pytest
import torch

from flood_uncertainty.data.sen1floods11_paired import PairedSen1Floods11Dataset
from flood_uncertainty.utils.config_loader import load_mode_config

spec = importlib.util.spec_from_file_location("train_sen1", "scripts/train/train_edl_terramind_sen1.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def config(name):
    cfg = load_mode_config(f"configurations/terramind_sen1/{name}.json", mode="train")
    cfg.model_params.hyperparameters["backbone_pretrained"] = False
    return cfg


@pytest.mark.parametrize("name", ["tm_l1c13_wf_s1f11", "tm_l1c13_s1f11", "tm_s1grd_kuro_s1f11",
                                  "tm_s1grd_s1f11", "tm_l1c13_s1grd_s1f11"])
def test_shipped_configs_are_consistent(name):
    train.check_config(config(name))


def test_rejects_kurosiwo_s1_stats_on_sen1():
    cfg = config("tm_s1grd_s1f11")
    cfg.model_params.hyperparameters["backbone_s1_input"]["units"] = "linear_zscore"
    with pytest.raises(ValueError, match="backbone_s1_input"):
        train.check_config(cfg)


def test_rejects_wrong_modalities_for_paired():
    cfg = config("tm_l1c13_s1grd_s1f11")
    cfg.model_params.hyperparameters["backbone_modalities"] = ["S2L1C"]
    with pytest.raises(ValueError, match="backbone_modalities"):
        train.check_config(cfg)


@pytest.fixture(scope="module")
def multimodal_model():
    from flood_uncertainty.models.edl_terramind import EDL_TerraMind_ML4FloodsModel

    torch.manual_seed(0)
    return EDL_TerraMind_ML4FloodsModel(config("tm_l1c13_s1grd_s1f11").model_params, normalized_data=True)


def test_multimodal_water_only_step(multimodal_model):
    mask = torch.zeros(2, 2, 64, 64, dtype=torch.long)
    mask[:, 1] = torch.randint(0, 3, (2, 64, 64))
    loss = multimodal_model.training_step({"image": torch.randn(2, 15, 64, 64), "mask": mask}, 0)
    assert torch.isfinite(loss)
    with torch.no_grad():
        assert multimodal_model.network(torch.randn(1, 15, 64, 64)).shape == (1, 2, 64, 64)
        with pytest.raises(ValueError):
            multimodal_model.network(torch.randn(1, 13, 64, 64))


def test_stage1_loader_maps_cloud_water_head_and_skips_renorm(tmp_path):
    from flood_uncertainty.models.edl_terramind import EDL_TerraMind_ML4FloodsModel

    model = EDL_TerraMind_ML4FloodsModel(config("tm_l1c13_wf_s1f11").model_params, normalized_data=True)
    stage1 = {k: torch.randn_like(v.float()).to(v.dtype) for k, v in model.state_dict().items()}
    for key in train.HEAD_KEYS:
        stage1[key] = torch.randn(4, *stage1[key].shape[1:])
    renorm = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(train.RENORM_PREFIX)}
    torch.save({"state_dict": stage1}, tmp_path / "s1.ckpt")
    train.load_stage1_weights(model, str(tmp_path / "s1.ckpt"))
    state = model.state_dict()
    assert torch.equal(state["network.conv_last.weight"], stage1["network.conv_last.weight"][2:4])
    assert all(torch.equal(state[k], v) for k, v in renorm.items())

    del stage1["network.conv_last.bias"]
    torch.save({"state_dict": stage1}, tmp_path / "s1_missing.ckpt")
    with pytest.raises(ValueError, match="missing"):
        train.load_stage1_weights(model, str(tmp_path / "s1_missing.ckpt"))


class _Fake:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, i):
        return self.items[i]


def test_paired_item_keeps_label_only_where_both_valid():
    gt = torch.tensor([[1, 2], [2, 1]])
    opt_mask = torch.stack([torch.zeros_like(gt), gt * torch.tensor([[1, 1], [0, 1]])])  # one optical no-data pixel
    sar_mask = (gt * torch.tensor([[1, 0], [1, 1]]))[None]  # one SAR no-data pixel
    ds = PairedSen1Floods11Dataset.__new__(PairedSen1Floods11Dataset)
    ds.optical = _Fake([{"image": torch.ones(13, 2, 2), "mask": opt_mask, "id": "a"}])
    ds.sar = _Fake([{"image": torch.zeros(2, 2, 2), "mask": sar_mask, "id": "a"}])
    item = ds[0]
    assert item["image"].shape == (15, 2, 2) and torch.equal(item["image"][13:], torch.zeros(2, 2, 2))
    assert torch.equal(item["mask"][1], torch.tensor([[1, 0], [0, 1]]))
    assert torch.equal(item["mask"][0], torch.zeros(2, 2, dtype=torch.long))
