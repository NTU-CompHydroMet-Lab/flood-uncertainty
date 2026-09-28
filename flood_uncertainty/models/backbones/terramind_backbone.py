"""TerraMind v1 encoder wrapped as a pyramid-feature backbone.

Design (docs/foundation_model_integration.md §3, §5, §6):

* The encoder comes from ``backbones/terramind_vendored`` (terratorch 1.2.11
  sources, see that package's docstring for why it is vendored). Only its
  ``build_terramind_vit`` factory is called: no Task, no registry, no terratorch
  necks. The three helpers below (select layers, tokens -> image, learned
  pyramid) are plain ``nn.Module`` so the wrapper stays self-contained.
* Input is ``dict[str, Tensor]`` keyed by modality (``"S2L2A"``), output is a
  ``list[Tensor]`` pyramid. ``out_channels`` / ``downsample_ratios`` are exposed
  so any decoder can configure itself without a dummy forward.
* Optional input re-normalisation: the ml4floods dataloader already z-scores
  each band with ``SENTINEL2_NORMALIZATION``; TerraMind was pre-trained with its
  own per-band mean/std. ``renormalize_input=True`` undoes the ml4floods
  z-score and applies the TerraMind statistics so the pretrained patch
  embedding sees the distribution it was trained on (risk #5b in the plan).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ml4floods.data.worldfloods.configs import BANDS_S2, CHANNELS_CONFIGURATIONS, SENTINEL2_NORMALIZATION

# --------------------------------------------------------------------------- #
# Band bookkeeping
# --------------------------------------------------------------------------- #

#: Sentinel-2 band id (ml4floods naming) -> TerraMind band name.
S2_BAND_TO_TERRAMIND: Dict[str, str] = {
    "B1": "COASTAL_AEROSOL",
    "B2": "BLUE",
    "B3": "GREEN",
    "B4": "RED",
    "B5": "RED_EDGE_1",
    "B6": "RED_EDGE_2",
    "B7": "RED_EDGE_3",
    "B8": "NIR_BROAD",
    "B8A": "NIR_NARROW",
    "B9": "WATER_VAPOR",
    "B10": "CIRRUS",  # L1C only; not part of the S2L2A pretrained bands
    "B11": "SWIR_1",
    "B12": "SWIR_2",
}

#: TerraMind v1 pre-training statistics for the ``S2L2A`` modality
#: (terratorch 1.2.11 ``terramind_register.v1_pretraining_mean/std``).
TERRAMIND_V1_S2L2A_MEAN: Dict[str, float] = {
    "COASTAL_AEROSOL": 1390.458,
    "BLUE": 1503.317,
    "GREEN": 1718.197,
    "RED": 1853.91,
    "RED_EDGE_1": 2199.1,
    "RED_EDGE_2": 2779.975,
    "RED_EDGE_3": 2987.011,
    "NIR_BROAD": 3083.234,
    "NIR_NARROW": 3132.22,
    "WATER_VAPOR": 3162.988,
    "SWIR_1": 2424.884,
    "SWIR_2": 1857.648,
}
TERRAMIND_V1_S2L2A_STD: Dict[str, float] = {
    "COASTAL_AEROSOL": 2106.761,
    "BLUE": 2141.107,
    "GREEN": 2038.973,
    "RED": 2134.138,
    "RED_EDGE_1": 2085.321,
    "RED_EDGE_2": 1889.926,
    "RED_EDGE_3": 1820.257,
    "NIR_BROAD": 1871.918,
    "NIR_NARROW": 1753.829,
    "WATER_VAPOR": 1797.379,
    "SWIR_1": 1434.261,
    "SWIR_2": 1334.311,
}


def terramind_bands_from_channel_configuration(channel_configuration: str) -> List[str]:
    """Map an ml4floods ``channel_configuration`` (e.g. ``"bgriswirs"``) to TerraMind S2L2A band names.

    Raises if the configuration contains a band TerraMind S2L2A was not
    pre-trained on (currently only ``B10``/CIRRUS).
    """
    if channel_configuration not in CHANNELS_CONFIGURATIONS:
        raise ValueError(f"Unknown channel_configuration: {channel_configuration}")
    band_ids = [BANDS_S2[i] for i in CHANNELS_CONFIGURATIONS[channel_configuration]]
    bands = []
    for b in band_ids:
        name = S2_BAND_TO_TERRAMIND[b]
        if name not in TERRAMIND_V1_S2L2A_MEAN:
            raise ValueError(f"Band {b} ({name}) is not available in the TerraMind S2L2A pretrained bands")
        bands.append(name)
    return bands


# --------------------------------------------------------------------------- #
# Pure-nn.Module helpers (mirroring terratorch necks without importing them)
# --------------------------------------------------------------------------- #


class InputRenormalizer(nn.Module):
    """Undo ml4floods per-band z-score, apply TerraMind per-band z-score.

    ``x_raw = x * ml_std + ml_mean``; ``x_tm = (x_raw - tm_mean) / tm_std``.
    All statistics are buffers so they follow ``.to(device)`` and are saved in
    checkpoints.
    """

    def __init__(self, channel_configuration: str, bands: Sequence[str]):
        super().__init__()
        s2_idx = CHANNELS_CONFIGURATIONS[channel_configuration]
        ml = SENTINEL2_NORMALIZATION[s2_idx]  # (C, 2) -> mean, std
        tm_mean = np.array([TERRAMIND_V1_S2L2A_MEAN[b] for b in bands], dtype=np.float32)
        tm_std = np.array([TERRAMIND_V1_S2L2A_STD[b] for b in bands], dtype=np.float32)
        if len(bands) != ml.shape[0]:
            raise ValueError("bands and channel_configuration length mismatch")
        view = lambda a: torch.as_tensor(a, dtype=torch.float32).view(1, -1, 1, 1)  # noqa: E731
        self.register_buffer("ml_mean", view(ml[:, 0]))
        self.register_buffer("ml_std", view(ml[:, 1]))
        self.register_buffer("tm_mean", view(tm_mean))
        self.register_buffer("tm_std", view(tm_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.ml_std + self.ml_mean - self.tm_mean) / self.tm_std


class SelectIndices(nn.Module):
    def __init__(self, indices: Sequence[int]):
        super().__init__()
        self.indices = list(indices)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [features[i] for i in self.indices]


class ReshapeTokensToImage(nn.Module):
    """``(B, N, D)`` row-major token grid -> ``(B, D, H', W')``.

    TerraMind patchifies with ``"b d (nh ph) (nw pw) -> b (nh nw) (ph pw d)"``,
    i.e. tokens are ordered row-major over ``(nh, nw)``. The inverse is a
    transpose + reshape with ``H' = nh``, ``W' = nw`` (risk #4 in the plan:
    the top-left token must land at the top-left pixel).
    """

    def forward(self, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = tokens.shape
        h, w = grid_hw
        if n != h * w:
            raise ValueError(f"Token count {n} != grid {h}x{w}")
        return tokens.transpose(1, 2).reshape(b, d, h, w)


class LearnedInterpolateToPyramidal(nn.Module):
    """Turn 4 same-resolution ViT feature maps (stride 16) into a 4-level pyramid.

    Level 0: two learned 2x upsamplings -> stride 4
    Level 1: one learned 2x upsampling  -> stride 8
    Level 2: identity                   -> stride 16
    Level 3: 2x max-pool                -> stride 32
    """

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("LearnedInterpolateToPyramidal expects exactly 4 input feature maps")
        c0, c1, c2, c3 = channels
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(c0, c0, kernel_size=2, stride=2),
            nn.BatchNorm2d(c0),
            nn.GELU(),
            nn.ConvTranspose2d(c0, c0, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = [c0, c1, c2, c3]
        self.scale_factors = [4, 2, 1, 0.5]

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        f0, f1, f2, f3 = feats
        return [self.fpn1(f0), self.fpn2(f1), self.fpn3(f2), self.fpn4(f3)]


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #


def _build_terramind_encoder(
    variant: str,
    modalities: Sequence[str],
    bands: Optional[Dict[str, List[str]]],
    pretrained: bool,
    local_ckpt_path: Optional[str],
) -> nn.Module:
    """The single place that touches the (vendored) TerraMind factory.

    Equivalent to ``terratorch.registry.BACKBONE_REGISTRY.build(variant, ...)``;
    swapping back to terratorch later means changing only this function.
    """
    from flood_uncertainty.models.backbones.terramind_vendored import build_terramind_vit

    kwargs = dict(modalities=list(modalities), merge_method="mean")
    if bands is not None:
        kwargs["bands"] = bands
    if local_ckpt_path:
        # Offline path: an explicit ckpt_path skips the HuggingFace download.
        kwargs["ckpt_path"] = local_ckpt_path
        kwargs["pretrained"] = False
    else:
        kwargs["pretrained"] = pretrained
    return build_terramind_vit(variant, **kwargs)


class TerraMindBackbone(nn.Module):
    """TerraMind ViT encoder -> 4-level feature pyramid.

    Args:
        variant: terratorch backbone name, e.g. ``"terramind_v1_base"``.
        modalities: TerraMind modality keys, currently a single one (``"S2L2A"``).
        channel_configuration: ml4floods channel configuration of the incoming
            tensor. Used to derive the TerraMind band subset and (optionally)
            the re-normalisation statistics. Only meaningful for ``S2L2A``.
        bands: explicit TerraMind band names; overrides ``channel_configuration``.
        pretrained: download / load HuggingFace weights.
        local_ckpt_path: offline ``.pt`` state dict; when set, ``pretrained`` is ignored.
        select_layers: 0-based encoder block indices to tap (4 required).
        freeze: set ``requires_grad=False`` on every encoder parameter.
        renormalize_input: apply :class:`InputRenormalizer` (S2L2A only).
        patch_size: TerraMind patch size (16 for v1).
    """

    def __init__(
        self,
        variant: str = "terramind_v1_base",
        modalities: Sequence[str] = ("S2L2A",),
        channel_configuration: Optional[str] = None,
        bands: Optional[Sequence[str]] = None,
        pretrained: bool = True,
        local_ckpt_path: Optional[str] = None,
        select_layers: Sequence[int] = (2, 5, 8, 11),
        freeze: bool = False,
        renormalize_input: bool = True,
        patch_size: int = 16,
    ):
        super().__init__()
        if len(modalities) != 1:
            raise NotImplementedError("TerraMindBackbone currently supports exactly one modality")
        if len(select_layers) != 4:
            raise ValueError("select_layers must list exactly 4 encoder blocks")
        self.modality = modalities[0]
        self.patch_size = patch_size
        self.select_layers = list(select_layers)

        band_list: Optional[List[str]] = list(bands) if bands is not None else None
        if band_list is None and channel_configuration is not None and self.modality == "S2L2A":
            band_list = terramind_bands_from_channel_configuration(channel_configuration)
        self.bands = band_list

        if renormalize_input:
            if self.modality != "S2L2A" or channel_configuration is None or band_list is None:
                raise ValueError("renormalize_input requires modality='S2L2A' and channel_configuration")
            self.renorm: nn.Module = InputRenormalizer(channel_configuration, band_list)
        else:
            self.renorm = nn.Identity()

        self.encoder = _build_terramind_encoder(
            variant=variant,
            modalities=modalities,
            bands={self.modality: band_list} if band_list is not None else None,
            pretrained=pretrained,
            local_ckpt_path=local_ckpt_path,
        )
        enc_channels = list(self.encoder.out_channels)  # one entry per encoder block
        if max(self.select_layers) >= len(enc_channels):
            raise ValueError(f"select_layers {self.select_layers} out of range for {len(enc_channels)} blocks")

        self.select = SelectIndices(self.select_layers)
        self.reshape = ReshapeTokensToImage()
        self.pyramid = LearnedInterpolateToPyramidal([enc_channels[i] for i in self.select_layers])

        # Contract attributes (plan §3.1)
        self.out_channels: List[int] = list(self.pyramid.out_channels)
        self.downsample_ratios: List[int] = [int(patch_size / s) for s in self.pyramid.scale_factors]

        self.frozen = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            # keep frozen encoder in eval mode (no dropout / drop-path stochasticity)
            self.encoder.eval()
        return self

    def forward(self, image_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        if self.modality not in image_dict:
            raise KeyError(f"Expected modality '{self.modality}' in input dict, got {list(image_dict)}")
        x = self.renorm(image_dict[self.modality])
        _, _, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f"Input {h}x{w} must be divisible by patch size {self.patch_size}")
        grid = (h // self.patch_size, w // self.patch_size)
        tokens = self.encoder({self.modality: x})  # list of (B, N, D)
        tokens = self.select(tokens)
        maps = [self.reshape(t, grid) for t in tokens]
        return self.pyramid(maps)
