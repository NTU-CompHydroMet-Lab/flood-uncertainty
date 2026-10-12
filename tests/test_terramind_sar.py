"""TerraMind S1 path: dB re-normalisation and the single-task SAR EDL model (random init, CPU)."""

import pytest
import torch

from flood_uncertainty.models.backbones.terramind_backbone import TERRAMIND_V1_S1_STATS, S1InputRenormalizer
from flood_uncertainty.utils.config_loader import load_mode_config

KURO_MEAN, KURO_STD = [0.0953, 0.0264], [0.0427, 0.0215]
TM_MEAN, TM_STD = (torch.tensor(v).view(1, 2, 1, 1) for v in TERRAMIND_V1_S1_STATS["S1GRD"])


def test_linear_zscore_maps_to_terramind_db():
    lin = torch.rand(2, 2, 8, 8) * 0.15 + 1e-3  # KuroSiwo clamps linear power to [0, 0.15]
    x = (lin - torch.tensor(KURO_MEAN).view(1, 2, 1, 1)) / torch.tensor(KURO_STD).view(1, 2, 1, 1)
    expected = (10 * torch.log10(lin) - TM_MEAN) / TM_STD
    got = S1InputRenormalizer("S1GRD", "linear_zscore", KURO_MEAN, KURO_STD)(x)
    assert torch.allclose(got, expected, atol=1e-4)


def test_linear_zero_is_floored_not_minus_inf():
    lin = torch.zeros(1, 2, 4, 4)  # no-data / zero backscatter
    x = (lin - torch.tensor(KURO_MEAN).view(1, 2, 1, 1)) / torch.tensor(KURO_STD).view(1, 2, 1, 1)
    got = S1InputRenormalizer("S1GRD", "linear_zscore", KURO_MEAN, KURO_STD, linear_floor=1e-4)(x)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, (torch.full_like(got, -40.0) - TM_MEAN) / TM_STD, atol=1e-4)


def test_sen1_scaled_db_maps_to_terramind_db():
    mean, std = [0.6851, 0.5235], [0.0820, 0.1102]  # Sen1Floods11 official SAR normalisation
    db = torch.rand(2, 2, 8, 8) * 30 - 30
    x = ((db + 50) / 51 - torch.tensor(mean).view(1, 2, 1, 1)) / torch.tensor(std).view(1, 2, 1, 1)
    got = S1InputRenormalizer("S1GRD", "sen1_scaled_db_zscore", mean, std)(x)
    assert torch.allclose(got, (db - TM_MEAN) / TM_STD, atol=1e-3)


def test_rejects_unknown_units_and_modality():
    with pytest.raises(ValueError, match="units"):
        S1InputRenormalizer("S1GRD", "decibel", KURO_MEAN, KURO_STD)
    with pytest.raises(ValueError, match="Sentinel-1"):
        S1InputRenormalizer("S2L1C", "linear_zscore", KURO_MEAN, KURO_STD)


@pytest.fixture(scope="module")
def sar_model():
    from flood_uncertainty.models.edl_terramind import EDL_TerraMind_SAR

    cfg = load_mode_config("configurations/terramind_sar/tm_s1grd.json", mode="train")
    cfg.model_params.hyperparameters["backbone_pretrained"] = False
    torch.manual_seed(0)
    return EDL_TerraMind_SAR(cfg.model_params, normalized_data=False)


def test_sar_model_shapes_and_single_task(sar_model):
    assert sar_model.num_class == 1
    with torch.no_grad():
        assert sar_model.network(torch.randn(2, 2, 224, 224)).shape == (2, 2, 224, 224)
        assert sar_model.network(torch.randn(1, 2, 200, 150)).shape == (1, 2, 200, 150)


def test_sar_training_step_backward(sar_model):
    batch = {"image": torch.randn(2, 2, 64, 64), "mask": torch.randint(0, 3, (2, 1, 64, 64))}
    sar_model.train()
    loss = sar_model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert sar_model.network.backbone.encoder.encoder[0].attn.qkv.weight.grad is not None


def test_sar_optimizer_groups_and_global_iou_hook(sar_model):
    groups = sar_model.configure_optimizers()["optimizer"].param_groups
    assert [g["lr"] for g in groups] == [1e-4, 1e-5]
    # the epoch-level pooled IoU of EDL_SAR_Unet is inherited
    from flood_uncertainty.models.edl import EDL_SAR_Unet

    assert type(sar_model).on_validation_epoch_end is EDL_SAR_Unet.on_validation_epoch_end


def test_train_script_rejects_mismatched_s1_stats():
    import importlib.util

    spec = importlib.util.spec_from_file_location("t", "scripts/train/train_edl_terramind_sar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = load_mode_config("configurations/terramind_sar/tm_s1grd.json", mode="train")
    mod.check_s1_input_matches_loader(cfg)  # shipped config is consistent
    cfg.model_params.hyperparameters["backbone_s1_input"]["mean"] = [0.1, 0.03]
    with pytest.raises(ValueError, match="data_mean"):
        mod.check_s1_input_matches_loader(cfg)
