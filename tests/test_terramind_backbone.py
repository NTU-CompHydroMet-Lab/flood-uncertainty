"""Unit tests for flood_uncertainty.models.backbones.terramind_backbone.

The first block covers the pure helpers. The second block builds a randomly
initialised (``pretrained=False``) vendored TerraMind ViT on CPU.
"""

import numpy as np
import pytest
import torch

from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS, SENTINEL2_NORMALIZATION
from flood_uncertainty.models.backbones.terramind_backbone import (
    TERRAMIND_V1_S2L1C_MEAN,
    TERRAMIND_V1_S2L1C_STD,
    TERRAMIND_V1_S2L2A_MEAN,
    TERRAMIND_V1_S2L2A_STD,
    InputRenormalizer,
    LearnedInterpolateToPyramidal,
    ReshapeTokensToImage,
    TerraMindBackbone,
    terramind_bands_from_channel_configuration,
)

# --------------------------------------------------------------------------- #
# Helpers (no encoder needed)
# --------------------------------------------------------------------------- #


def test_bgriswirs_maps_to_terramind_band_names():
    assert terramind_bands_from_channel_configuration("bgriswirs") == [
        "BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2",
    ]


def test_channel_configuration_with_b10_is_rejected():
    # "all" includes B10 (CIRRUS), which the S2L2A pretrained embedding does not have
    with pytest.raises(ValueError, match="B10"):
        terramind_bands_from_channel_configuration("all")


def test_reshape_tokens_is_row_major():
    """Top-left token -> top-left pixel; token index increases along width first (plan risk #4)."""
    h, w, d = 3, 4, 1
    tokens = torch.arange(h * w, dtype=torch.float32).view(1, h * w, d)
    img = ReshapeTokensToImage()(tokens, (h, w))
    assert img.shape == (1, d, h, w)
    assert img[0, 0, 0, 0] == 0
    assert img[0, 0, 0, 1] == 1
    assert img[0, 0, 1, 0] == w


def test_reshape_rejects_wrong_grid():
    with pytest.raises(ValueError):
        ReshapeTokensToImage()(torch.zeros(1, 12, 8), (3, 3))


def test_learned_pyramid_strides():
    pyr = LearnedInterpolateToPyramidal([8, 8, 8, 8])
    f = [torch.randn(2, 8, 4, 4) for _ in range(4)]
    out = pyr(f)
    assert [o.shape[-1] for o in out] == [16, 8, 4, 2]
    assert pyr.scale_factors == [4, 2, 1, 0.5]


def test_all_maps_to_s2l1c_band_names_in_pretrained_order():
    # WorldFloods v2 S2 is L1C with 13 bands in the same order as TerraMind's S2L1C embedding
    assert terramind_bands_from_channel_configuration("all", "S2L1C") == [
        "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
        "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2",
    ]


def test_unsupported_s2_modality_is_rejected():
    with pytest.raises(ValueError, match="S2RGB"):
        terramind_bands_from_channel_configuration("bgriswirs", "S2RGB")


def test_input_renormalizer_uses_s2l1c_statistics():
    cfg = "all"
    bands = terramind_bands_from_channel_configuration(cfg, "S2L1C")
    ml = SENTINEL2_NORMALIZATION[CHANNELS_CONFIGURATIONS[cfg]]
    raw = torch.rand(2, 13, 4, 4) * 5000.0
    x_ml = (raw - torch.tensor(ml[:, 0]).view(1, -1, 1, 1)) / torch.tensor(ml[:, 1]).view(1, -1, 1, 1)

    tm_mean = torch.tensor([TERRAMIND_V1_S2L1C_MEAN[b] for b in bands]).view(1, -1, 1, 1)
    tm_std = torch.tensor([TERRAMIND_V1_S2L1C_STD[b] for b in bands]).view(1, -1, 1, 1)
    expected = (raw - tm_mean) / tm_std

    got = InputRenormalizer(cfg, bands, "S2L1C")(x_ml)
    assert torch.allclose(got, expected, atol=1e-4)


def test_s2l1c_backbone_uses_13_band_l1c_embedding():
    bb = TerraMindBackbone(
        variant="terramind_v1_base", modalities=("S2L1C",), channel_configuration="all", pretrained=False,
    )
    assert bb.encoder.encoder_embeddings["untok_sen2l1c@224"].proj.in_features == 13 * 16 * 16
    with torch.no_grad():
        feats = bb({"S2L1C": torch.randn(1, 13, 32, 32)})
    assert [f.shape[-1] for f in feats] == [8, 4, 2, 1]


def test_input_renormalizer_matches_manual_computation():
    cfg = "bgriswirs"
    bands = terramind_bands_from_channel_configuration(cfg)
    ml = SENTINEL2_NORMALIZATION[CHANNELS_CONFIGURATIONS[cfg]]
    raw = torch.rand(2, 6, 4, 4) * 5000.0
    x_ml = (raw - torch.tensor(ml[:, 0]).view(1, -1, 1, 1)) / torch.tensor(ml[:, 1]).view(1, -1, 1, 1)

    tm_mean = torch.tensor([TERRAMIND_V1_S2L2A_MEAN[b] for b in bands]).view(1, -1, 1, 1)
    tm_std = torch.tensor([TERRAMIND_V1_S2L2A_STD[b] for b in bands]).view(1, -1, 1, 1)
    expected = (raw - tm_mean) / tm_std

    got = InputRenormalizer(cfg, bands)(x_ml)
    assert torch.allclose(got, expected, atol=1e-4)


# --------------------------------------------------------------------------- #
# Backbone (vendored TerraMind, random init, CPU)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    return TerraMindBackbone(
        variant="terramind_v1_base",
        modalities=("S2L2A",),
        channel_configuration="bgriswirs",
        pretrained=False,
        select_layers=(2, 5, 8, 11),
        freeze=False,
        renormalize_input=True,
    )


def test_backbone_contract_attributes(backbone):
    assert backbone.out_channels == [768, 768, 768, 768]
    assert backbone.downsample_ratios == [4, 8, 16, 32]


def test_backbone_pyramid_shapes_match_contract(backbone):
    x = torch.randn(2, 6, 64, 64)
    with torch.no_grad():
        feats = backbone({"S2L2A": x})
    assert len(feats) == 4
    for f, c, r in zip(feats, backbone.out_channels, backbone.downsample_ratios):
        assert f.shape == (2, c, 64 // r, 64 // r), f.shape


def test_backbone_rejects_missing_modality(backbone):
    with pytest.raises(KeyError):
        backbone({"S1GRD": torch.randn(1, 2, 32, 32)})


def test_backbone_rejects_non_multiple_of_patch(backbone):
    with pytest.raises(ValueError, match="divisible"):
        backbone({"S2L2A": torch.randn(1, 6, 40, 40)})


def test_patch_embedding_has_six_input_channels(backbone):
    proj = backbone.encoder.encoder_embeddings["untok_sen2l2a@224"].proj
    assert proj.in_features == 6 * 16 * 16


def test_freeze_disables_encoder_grads_only():
    bb = TerraMindBackbone(channel_configuration="bgriswirs", pretrained=False, freeze=True)
    assert all(not p.requires_grad for p in bb.encoder.parameters())
    assert all(p.requires_grad for p in bb.pyramid.parameters())
    bb.train()
    assert not bb.encoder.training  # frozen encoder stays in eval mode


def test_local_ckpt_path_skips_huggingface_download(tmp_path, monkeypatch):
    import flood_uncertainty.models.backbones.terramind_vendored.factory as reg

    def _no_download(*a, **k):
        raise AssertionError("hf_hub_download must not be called when backbone_local_ckpt is set")

    monkeypatch.setattr(reg, "hf_hub_download", _no_download)

    # A local ckpt must be in the raw HuggingFace layout (12-band S2L2A patch embedding); the band
    # subset is applied *after* loading, exactly like the online path. Build such a state dict from a
    # random full-band model, save it, reload it offline.
    src = TerraMindBackbone(channel_configuration=None, bands=None, pretrained=False, renormalize_input=False)
    assert src.encoder.encoder_embeddings["untok_sen2l2a@224"].proj.in_features == 12 * 16 * 16
    ckpt = tmp_path / "terramind_random.pt"
    torch.save(src.encoder.state_dict(), ckpt)

    bb = TerraMindBackbone(
        channel_configuration="bgriswirs",
        pretrained=True,  # would trigger a download if local_ckpt_path were ignored
        local_ckpt_path=str(ckpt),
        renormalize_input=False,
    )
    assert bb.encoder.encoder_embeddings["untok_sen2l2a@224"].proj.in_features == 6 * 16 * 16
    with torch.no_grad():
        out = bb({"S2L2A": torch.randn(1, 6, 32, 32)})
    assert out[0].shape == (1, 768, 8, 8)


# --------------------------------------------------------------------------- #
# Vendored vs terratorch equivalence (only when terratorch is installed)
# --------------------------------------------------------------------------- #


def test_vendored_matches_terratorch_with_pretrained_weights():
    pytest.importorskip("terratorch")
    torch.manual_seed(0)
    kw = dict(channel_configuration="bgriswirs", pretrained=True, renormalize_input=False)
    a = TerraMindBackbone(impl="vendored", **kw).eval()
    b = TerraMindBackbone(impl="terratorch", **kw).eval()
    # pyramid heads are randomly initialised per instance: copy them so only the encoder differs
    b.pyramid.load_state_dict(a.pyramid.state_dict())
    sa, sb = a.encoder.state_dict(), b.encoder.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        fa, fb = a({"S2L2A": x}), b({"S2L2A": x})
    for u, v in zip(fa, fb):
        assert torch.allclose(u, v, atol=1e-5)


def test_unknown_impl_is_rejected():
    with pytest.raises(ValueError, match="backbone impl"):
        TerraMindBackbone(channel_configuration="bgriswirs", pretrained=False, impl="nope")


# --------------------------------------------------------------------------- #
# Vendored vs terratorch (only runs where terratorch is installed, i.e. Python >= 3.11)
# --------------------------------------------------------------------------- #


def test_vendored_matches_terratorch_with_pretrained_weights():
    pytest.importorskip("terratorch")
    try:
        a = TerraMindBackbone(channel_configuration="bgriswirs", pretrained=True, impl="vendored")
        b = TerraMindBackbone(channel_configuration="bgriswirs", pretrained=True, impl="terratorch")
    except Exception as e:  # no HF cache / no network
        pytest.skip(f"pretrained weights unavailable: {e}")
    # identical encoder state
    sa, sb = a.encoder.state_dict(), b.encoder.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    # identical encoder outputs (pyramid heads are randomly initialised, so compare before them)
    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        ta = a.encoder({"S2L2A": x})
        tb = b.encoder({"S2L2A": x})
    assert len(ta) == len(tb) == 12
    assert all(torch.allclose(u, v, atol=1e-5) for u, v in zip(ta, tb))
