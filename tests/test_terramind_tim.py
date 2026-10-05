"""TiM ("Thinking in Modalities") path of TerraMindBackbone.

Random-init tests run everywhere; the pretrained test needs the TerraMind v1 base weights in the
HuggingFace cache (skipped otherwise, never downloads).
"""

import pytest
import torch

from flood_uncertainty.models.backbones.terramind_backbone import TerraMindBackbone


def _tim_backbone(**kw):
    args = dict(
        variant="terramind_v1_base", modalities=("S2L1C",), channel_configuration="all",
        pretrained=False, tim_modalities=["LULC"], tim_temps=0.0,
    )
    args.update(kw)
    return TerraMindBackbone(**args)


@pytest.fixture(scope="module")
def tim():
    torch.manual_seed(0)
    return _tim_backbone()


def test_tim_rejects_band_subset():
    with pytest.raises(ValueError, match="TiM needs all pre-trained"):
        _tim_backbone(modalities=("S2L2A",), channel_configuration="bgriswirs")


def test_tim_pyramid_matches_contract(tim):
    with torch.no_grad():
        feats = tim({"S2L1C": torch.randn(1, 13, 64, 64)})
    for f, c, r in zip(feats, tim.out_channels, tim.downsample_ratios):
        assert f.shape == (1, c, 64 // r, 64 // r), f.shape


def test_tim_generator_is_frozen_and_stays_eval(tim):
    gen = tim.encoder.sampler.model
    assert all(not p.requires_grad for p in gen.parameters())
    assert any(p.requires_grad for p in tim.encoder.encoder.parameters())
    tim.train()
    assert tim.encoder.encoder.training and not gen.training
    tim.eval()


def test_tim_argmax_is_deterministic(tim):
    x = torch.randn(1, 13, 32, 32)
    with torch.no_grad():
        a, b = tim({"S2L1C": x}), tim({"S2L1C": x})
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def _cached_weights():
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download("ibm-esa-geospatial/TerraMind-1.0-base", "TerraMind_v1_base.pt", local_files_only=True)
    except Exception:
        return None


@pytest.mark.skipif(_cached_weights() is None, reason="TerraMind v1 base weights not in HF cache")
def test_tim_pretrained_loads_encoder_and_generator(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    sd = torch.load(_cached_weights(), map_location="cpu", weights_only=True)
    bb = _tim_backbone(pretrained=True)
    msd = bb.encoder.state_dict()
    # fine-tuned encoder and frozen generator both start from the checkpoint
    for k in ["encoder.0.attn.qkv.weight", "encoder_embeddings.untok_sen2l1c@224.proj.weight",
              "encoder_embeddings.tok_lulc@224.token_emb.weight"]:
        assert torch.equal(msd[k], sd[k]), k
    for k in ["decoder.0.query_norm.weight", "decoder_embeddings.tok_lulc@224.to_logits.weight"]:
        if k in sd:
            assert torch.equal(msd["sampler.model." + k], sd[k]), k
    n_dec = sum(1 for k in msd if k.startswith("sampler.model.decoder."))
    assert n_dec == sum(1 for k in sd if k.startswith("decoder.")), "every generator decoder weight loaded"


def test_tim_unet_accepts_non_square_inputs():
    # inference tiles at image edges are not square; TiM's token embedding needs a square grid
    from flood_uncertainty.models.terramind_unet import TerraMindUNet

    torch.manual_seed(0)
    net = TerraMindUNet(_tim_backbone(), n_class=4).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 13, 45, 70))
    assert out.shape == (1, 4, 45, 70)


def test_tim_forward_preserves_global_torch_rng(tim):
    # upstream generate() calls torch.manual_seed; the vendored TiM forks the RNG around it
    x = torch.randn(1, 13, 64, 64)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    with torch.no_grad():
        tim({"S2L1C": x})
    assert torch.equal(torch.get_rng_state(), state), "TiM forward must not reseed the global torch RNG"
