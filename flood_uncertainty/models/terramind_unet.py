"""UNet-style decoder on top of a :class:`TerraMindBackbone`.

This is the drop-in replacement for ``ml4floods.models.architectures.unets.UNet``
used by the EDL model: same ``forward(x: Tensor) -> Tensor`` contract, same
``layer_factory.double_conv`` blocks in the decoder, same final 1x1 head. Only
the encoder changed (plan §0: "只有這顆換了；decoder、head 都不動").

Skip widths default to the original UNet's ``[64, 128, 256, 512]`` so the
decoder capacity matches the baseline; ``decoder_channels`` is given
top-down (``[512, 256, 128, 64]``) to match the plan's config field.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml4floods.models.architectures import layer_factory
from flood_uncertainty.models.backbones.terramind_backbone import TerraMindBackbone


class TerraMindUNet(nn.Module):
    """TerraMind encoder + UNet decoder + 1x1 head. Tensor in, tensor out.

    Args:
        backbone: a constructed :class:`TerraMindBackbone` (strides 4/8/16/32).
        n_class: number of output channels (``2 * num_tasks`` for EDL).
        decoder_channels: top-down decoder widths, deepest first.
    """

    def __init__(
        self,
        backbone: TerraMindBackbone,
        n_class: int,
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
    ):
        super().__init__()
        if list(backbone.downsample_ratios) != [4, 8, 16, 32]:
            raise ValueError(f"TerraMindUNet expects strides [4, 8, 16, 32], got {backbone.downsample_ratios}")
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must have 4 entries (deepest first)")

        self.backbone = backbone
        self.modality = backbone.modality
        skip = list(reversed(decoder_channels))  # shallow -> deep: [64, 128, 256, 512]

        # 1x1 laterals: ViT width (768) -> UNet skip widths
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, kernel_size=1) for c_in, c_out in zip(backbone.out_channels, skip)]
        )
        # Same block structure as ml4floods UNet: up-sample, concat skip, double_conv
        self.dconv_up3 = layer_factory.double_conv(skip[2] + skip[3], skip[2])
        self.dconv_up2 = layer_factory.double_conv(skip[1] + skip[2], skip[1])
        self.dconv_up1 = layer_factory.double_conv(skip[0] + skip[1], skip[0])
        # Stride 4 -> full resolution (the CNN UNet never leaves stride 1, so this is new)
        self.dconv_full = layer_factory.double_conv(skip[0], skip[0])
        self.conv_last = nn.Conv2d(skip[0], n_class, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # Resize to the skip's exact size rather than x2: the stride-32 level is a max-pool of the
        # stride-16 grid and floors odd sizes (e.g. 3x5 -> 1x2), so x2 would not line up.
        return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        # The ViT needs H, W divisible by the patch size. The CNN UNet only needed multiples of 8,
        # which ml4floods' tiling guarantees via SUBSAMPLE_MODULE; it knows nothing about this
        # model_type, so pad here (reflection, bottom/right) and crop the output back to (h, w).
        p = self.backbone.patch_size
        pad_h, pad_w = (-h) % p, (-w) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        feats = self.backbone({self.modality: x})  # the one translation line (plan §5)
        c1, c2, c3, c4 = [lat(f) for lat, f in zip(self.lateral, feats)]  # strides 4, 8, 16, 32

        y = torch.cat([self._up(c4, c3), c3], dim=1)
        y = self.dconv_up3(y)
        y = torch.cat([self._up(y, c2), c2], dim=1)
        y = self.dconv_up2(y)
        y = torch.cat([self._up(y, c1), c1], dim=1)
        y = self.dconv_up1(y)  # stride 4

        y = F.interpolate(y, size=(h + pad_h, w + pad_w), mode="bilinear", align_corners=True)
        y = self.dconv_full(y)
        return self.conv_last(y)[:, :, :h, :w]


def build_terramind_unet(h_params: Dict, n_class: int) -> TerraMindUNet:
    """Construct backbone + decoder from a ``model_params.hyperparameters`` dict.

    Recognised keys (all optional, plan §7 naming):
      backbone, backbone_modalities, backbone_pretrained, backbone_local_ckpt,
      backbone_select_layers, backbone_freeze, backbone_renormalize_input,
      backbone_impl, backbone_tim_modalities, backbone_tim_temps,
      decoder_channels, channel_configuration.
    """
    get = h_params.get
    backbone = TerraMindBackbone(
        variant=get("backbone", "terramind_v1_base"),
        modalities=tuple(get("backbone_modalities", ["S2L2A"])),
        channel_configuration=get("channel_configuration"),
        pretrained=bool(get("backbone_pretrained", True)),
        local_ckpt_path=get("backbone_local_ckpt", None),
        select_layers=tuple(get("backbone_select_layers", [2, 5, 8, 11])),
        freeze=bool(get("backbone_freeze", False)),
        renormalize_input=bool(get("backbone_renormalize_input", True)),
        impl=get("backbone_impl", "vendored"),
        tim_modalities=get("backbone_tim_modalities", None),
        tim_temps=float(get("backbone_tim_temps", 0.0)),
    )
    return TerraMindUNet(
        backbone=backbone,
        n_class=n_class,
        decoder_channels=tuple(get("decoder_channels", [512, 256, 128, 64])),
    )
