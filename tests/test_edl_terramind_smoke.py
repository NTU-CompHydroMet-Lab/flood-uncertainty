"""One forward + backward through EDL_TerraMind_ML4FloodsModel on CPU (plan milestone F4)."""

import torch

from ml4floods.models.utils.configuration import AttrDict
from flood_uncertainty.models.edl_terramind import EDL_TerraMind_ML4FloodsModel


def _model_params(**overrides):
    hp = {
        "channel_configuration": "bgriswirs",
        "num_channels": 6,
        "num_classes": 2,
        "label_names": [["invalid", "clear", "cloud"], ["invalid", "land", "water"]],
        "metric_monitor": "val_bce_land_water",
        "weight_problem": [0.2, 0.8],
        "pos_weight": [[1, 20], [1, 6]],
        "annealing_mode": "linear",
        "annealing_step": 10,
        "annealing_coefficient": 0.5,
        "lr": 1e-4,
        "lr_decay": 0.5,
        "lr_patience": 2,
        "backbone": "terramind_v1_base",
        "backbone_modalities": ["S2L2A"],
        "backbone_pretrained": False,
        "backbone_local_ckpt": None,
        "backbone_select_layers": [2, 5, 8, 11],
        "backbone_freeze": False,
        "backbone_renormalize_input": True,
        "backbone_lr_mult": 0.1,
        "decoder_channels": [512, 256, 128, 64],
    }
    hp.update(overrides)
    return AttrDict.from_nested_dicts({"hyperparameters": hp, "train": True, "test": False, "val_only": False})


def _batch(b=2, c=6, h=32, w=32):
    torch.manual_seed(0)
    return {
        "image": torch.randn(b, c, h, w),
        "mask": torch.randint(0, 3, (b, 2, h, w)),  # {0: invalid, 1: neg, 2: pos}
    }


def test_forward_shape_is_two_channels_per_task():
    model = EDL_TerraMind_ML4FloodsModel(_model_params(), normalized_data=True)
    out = model(_batch()["image"])
    assert out.shape == (2, 4, 32, 32)


def test_training_step_backward_is_finite():
    model = EDL_TerraMind_ML4FloodsModel(_model_params(), normalized_data=True)
    model.train()
    loss = model.training_step(_batch(), batch_idx=1)  # batch_idx != 0 avoids logger calls
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.network.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients reached the network"
    assert all(torch.isfinite(g).all() for g in grads)


def test_edl_outputs_are_probabilities_and_uncertainties():
    model = EDL_TerraMind_ML4FloodsModel(_model_params(), normalized_data=True)
    with torch.no_grad():
        out = model.edl_logits_to_output(model(_batch()["image"]))
    assert out["prob"].shape == (2, 2, 32, 32)
    assert ((out["prob"] >= 0) & (out["prob"] <= 1)).all()
    assert ((out["dst_u"] > 0) & (out["dst_u"] <= 1)).all()


def test_optimizer_uses_lower_lr_for_backbone():
    model = EDL_TerraMind_ML4FloodsModel(_model_params(lr=1e-3, backbone_lr_mult=0.1))
    opt = model.configure_optimizers()["optimizer"]
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs == [1e-3, 1e-4]


def test_optimizer_drops_frozen_backbone():
    model = EDL_TerraMind_ML4FloodsModel(_model_params(backbone_freeze=True))
    opt = model.configure_optimizers()["optimizer"]
    assert len(opt.param_groups) == 1
    n_enc = sum(p.numel() for p in model.network.backbone.encoder.parameters())
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt < n_enc  # decoder + pyramid only


def test_forward_accepts_sizes_not_divisible_by_patch():
    """Inference tiles are arbitrary sizes; the model pads to the ViT patch grid and crops back."""
    model = EDL_TerraMind_ML4FloodsModel(_model_params(), normalized_data=True)
    with torch.no_grad():
        out = model(torch.randn(1, 6, 45, 70))
    assert out.shape == (1, 4, 45, 70)


def test_input_skip_is_off_by_default():
    model = EDL_TerraMind_ML4FloodsModel(_model_params(), normalized_data=True)
    assert model.network.input_skip is None


def test_input_skip_forward_backward_and_param_cost():
    base = EDL_TerraMind_ML4FloodsModel(_model_params(), normalized_data=True)
    model = EDL_TerraMind_ML4FloodsModel(_model_params(decoder_input_skip_channels=32), normalized_data=True)
    extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
    assert 0 < extra < 100_000  # tiny next to ~97M
    out = model(torch.randn(1, 6, 45, 70))  # also through the pad / crop path
    assert out.shape == (1, 4, 45, 70)
    out.sum().backward()
    assert model.network.input_skip[0].weight.grad is not None
    # the skip trains at the decoder lr, not the backbone lr
    groups = model.configure_optimizers()["optimizer"].param_groups
    skip_ids = {id(p) for p in model.network.input_skip.parameters()}
    assert skip_ids <= {id(p) for p in groups[0]["params"]} and groups[0]["lr"] == 1e-4


def test_input_skip_needs_in_channels():
    import pytest
    from flood_uncertainty.models.terramind_unet import TerraMindUNet
    from flood_uncertainty.models.backbones.terramind_backbone import TerraMindBackbone

    bb = TerraMindBackbone(channel_configuration="bgriswirs", pretrained=False)
    with pytest.raises(ValueError, match="in_channels"):
        TerraMindUNet(bb, n_class=4, input_skip_channels=32)
