# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- flood-uncertainty vendoring note ---------------------------------------
# Adapted from terratorch 1.2.11 (IBM/terratorch, Apache-2.0), file
#   terratorch/models/backbones/terramind/model/terramind_register.py
# Kept: PRETRAINED_BANDS, HF weight ids, the four v1 ViT variant configs,
# select_modality_patch_embed_weights, checkpoint_filter_fn, build_terramind_vit.
# Also kept (for TiM): checkpoint_filter_fn_tim, build_terramind_tim (as build_terrammind_tim upstream).
# Dropped: TERRATORCH_*_REGISTRY registration, encoder-decoder / generation
# builders, tokenizer_dict. Edits are marked `# [vendored]`.
# -----------------------------------------------------------------------------

import logging
from functools import partial

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from .terramind_tim import TerraMindTiM
from .terramind_vit import TerraMindViT
from .tm_utils import LayerNorm

logger = logging.getLogger(__name__)

PRETRAINED_WEIGHTS = {
    "terramind_v1_tiny": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-tiny",
        "hf_hub_filename": "TerraMind_v1_tiny.pt",
    },
    "terramind_v1_small": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-small",
        "hf_hub_filename": "TerraMind_v1_small.pt",
    },
    "terramind_v1_base": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-base",
        "hf_hub_filename": "TerraMind_v1_base.pt",
    },
    "terramind_v1_large": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-large",
        "hf_hub_filename": "TerraMind_v1_large.pt",
    },
}

PRETRAINED_BANDS = {
    "untok_sen2l2a@224": [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "SWIR_1",
        "SWIR_2",
    ],
    "untok_sen2l1c@224": [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "CIRRUS",
        "SWIR_1",
        "SWIR_2",
    ],
    "untok_sen2rgb@224": ["RED", "GREEN", "BLUE"],
    "untok_sen1grd@224": ["VV", "VH"],
    "untok_sen1rtc@224": ["VV", "VH"],
    "untok_dem@224": ["DEM"],
}

# [vendored] architecture kwargs of the registered `terramind_v1_*` ViT backbones
_NORM = partial(LayerNorm, eps=1e-6, bias=False)
VARIANTS = {
    "terramind_v1_tiny": dict(
        encoder_depth=12, dim=192, num_heads=3, mlp_ratio=4, qkv_bias=True, proj_bias=True, mlp_bias=True,
        num_register_tokens=0, norm_layer=_NORM, act_layer=nn.GELU, gated_mlp=False,
    ),
    "terramind_v1_small": dict(
        encoder_depth=12, dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, proj_bias=True, mlp_bias=True,
        num_register_tokens=0, norm_layer=_NORM, act_layer=nn.GELU, gated_mlp=False,
    ),
    "terramind_v1_base": dict(
        encoder_depth=12, dim=768, num_heads=12, mlp_ratio=4, qkv_bias=False, proj_bias=False, mlp_bias=False,
        num_register_tokens=0, norm_layer=_NORM, act_layer=nn.SiLU, gated_mlp=True,
    ),
    "terramind_v1_large": dict(
        encoder_depth=24, dim=1024, num_heads=16, mlp_ratio=4, qkv_bias=False, proj_bias=False, mlp_bias=False,
        num_register_tokens=0, norm_layer=_NORM, act_layer=nn.SiLU, gated_mlp=True,
    ),
}


def select_modality_patch_embed_weights(model: TerraMindViT, bands: dict[str, list], pretrained_bands: dict[str, list]):
    """
    Update patch embeddings weights for each provided modality by selecting the pretrained weights for each band.
    Args:
         model (TerraMindViT): model
         bands (dict[str, list]): Bands with format {<modality>: [<band names>]}
         pretrained_bands (dict[str, list]): Pretrained bands of the model with format {<modality>: [<band names>]}
    """
    # Update modality names to match model layer names
    bands = {model.mod_name_mapping[k]: v for k, v in bands.items()}
    for mod, mod_bands in bands.items():
        if mod not in pretrained_bands:
            logger.info(f"Cannot load band weights for modality {mod}, not found in pretrained bands.")
            continue

        pixel_count = model.encoder_embeddings[mod].patch_size[0] * model.encoder_embeddings[mod].patch_size[1]

        pretrained_weight = model.encoder_embeddings[mod].proj.weight.clone()
        # Init new projection layer with updated number of channels
        model.encoder_embeddings[mod].proj = nn.Linear(
            pixel_count * len(mod_bands), model.encoder_embeddings[mod].dim_tokens, bias=False
        )
        temp_weight = model.encoder_embeddings[mod].proj.weight.clone()

        # Reshape to [dim, pixel, band]
        temp_weight = temp_weight.view(temp_weight.shape[0], pixel_count, -1)
        pretrained_weight = pretrained_weight.view(pretrained_weight.shape[0], pixel_count, -1)

        # Copy weights of bands
        for index, band in enumerate(mod_bands):
            if band in pretrained_bands[mod]:
                logging.info(f"Loaded weights for {band} in position {index} of patch embed")
                pretrained_index = pretrained_bands[mod].index(band)
                temp_weight[..., index] = pretrained_weight[..., pretrained_index]

        # Update model weights
        model.encoder_embeddings[mod].proj.weight = nn.Parameter(temp_weight.view(temp_weight.shape[0], -1))

    return model


def checkpoint_filter_fn(state_dict, model: TerraMindViT) -> dict:
    """Manually filter pre-trained weights for TerraMind to enable strict weight loading."""

    model_state_dict = model.state_dict()
    clean_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                clean_dict[k] = v
            else:
                logger.warning(
                    f"Shape for {k} ({list(v.shape)}) does not match model weights "
                    f"({list(model_state_dict[k].shape)}), skipping weights."
                )

    missing_params = set(model_state_dict.keys()) - set(clean_dict.keys())
    for k in missing_params:
        if not k.startswith("tokenizer"):
            logger.warning(f"Weights for {k} are missing in state dict, using random initialization.")
        clean_dict[k] = model_state_dict[k]

    state_dict = clean_dict

    return state_dict


def build_terramind_vit(
    variant: str = "terramind_v1_base",
    pretrained: bool = False,
    ckpt_path: str | None = None,
    bands: dict[str, list] | None = None,
    pretrained_bands: dict[str, list] | None = None,
    **kwargs,
) -> TerraMindViT:
    """[vendored] Equivalent of ``BACKBONE_REGISTRY.build(variant, ...)`` for the v1 ViT encoders.

    ``kwargs`` are forwarded to :class:`TerraMindViT` (e.g. ``modalities``,
    ``merge_method``, ``img_size``). Weight loading order matches upstream:
    explicit ``ckpt_path`` wins, else ``pretrained`` downloads from the
    HuggingFace hub (cache-aware), else random init. ``bands`` sub-selects the
    patch-embedding rows *after* weights are loaded.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TerraMind variant '{variant}'. Available: {sorted(VARIANTS)}")
    if pretrained_bands is None:
        pretrained_bands = PRETRAINED_BANDS

    model = TerraMindViT(pretrained=pretrained, **VARIANTS[variant], **kwargs)

    if ckpt_path is not None:
        # Load model from checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            logger.warning(f"Unexpected keys in ckpt_path {ckpt_path}: {loaded_keys.unexpected_keys}")

    elif pretrained:
        # Load model from Hugging Face
        state_dict_file = hf_hub_download(
            repo_id=PRETRAINED_WEIGHTS[variant]["hf_hub_id"], filename=PRETRAINED_WEIGHTS[variant]["hf_hub_filename"]
        )
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

    if bands is not None:
        model = select_modality_patch_embed_weights(model, bands, pretrained_bands)

    return model


def checkpoint_filter_fn_tim(state_dict, model: TerraMindTiM) -> dict:
    """Manually filter pre-trained weights for TerraMind TiM to enable strict weight loading.

    Each pretrained key is loaded into the fine-tunable encoder (if present there) *and* into the
    frozen generator ``sampler.model.<key>`` (encoder + decoder of the any-to-any model).
    """
    model_state_dict = model.state_dict()
    clean_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                clean_dict[k] = v
            else:
                logger.warning(
                    f"Shape for {k} ({list(v.shape)}) does not match model weights "
                    f"({list(model_state_dict[k].shape)}), skipping weights."
                )
        if "sampler.model." + k in model_state_dict:
            # Copy weights for MAE model for TiM
            encdec_k = "sampler.model." + k
            if v.shape == model_state_dict[encdec_k].shape:
                clean_dict[encdec_k] = v
            else:
                raise ValueError(
                    f"Shape for {k} ({list(v.shape)}) does not match MAE model weights "
                    f"({list(model_state_dict[encdec_k].shape)}). Cannot run chain of thoughts without MAE."
                )

    missing_params = set(model_state_dict.keys()) - set(clean_dict.keys())
    for k in missing_params:
        if k.startswith("sampler.model."):
            raise ValueError(f"Weights for {k} are missing in state dict, cannot run chain of thoughts without MAE.")
        if not k.startswith("tokenizer"):
            logger.warning(f"Weights for {k} are missing in state dict, using random initialization.")
        clean_dict[k] = model_state_dict[k]

    return clean_dict


def build_terramind_tim(
    variant: str = "terramind_v1_base",
    pretrained: bool = False,
    ckpt_path: str | None = None,
    bands: dict[str, list] | None = None,
    decoder_depth: int | None = None,
    **kwargs,
) -> TerraMindTiM:
    """[vendored] Equivalent of ``BACKBONE_REGISTRY.build(f"{variant}_tim", ...)``.

    Upstream (``build_terrammind_tim``) rejects ``bands``: the frozen generator only accepts the
    exact pre-trained modalities, so feed full-band inputs (e.g. ``S2L1C`` with all 13 bands).
    ``tokenizer_dict`` is not passed: tokenizers are only needed to tokenize inputs or to decode
    text targets, not to generate image-token modalities such as LULC.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TerraMind variant '{variant}'. Available: {sorted(VARIANTS)}")
    if bands is not None:
        raise NotImplementedError(
            f"Bands cannot be adapted for TerraMind TiM models and is expected to be None. "
            f"Only exact matches with pre-trained modalities are supported, got {bands}."
        )
    cfg = dict(VARIANTS[variant])
    cfg["decoder_depth"] = decoder_depth or cfg["encoder_depth"]  # [vendored] v1 decoders mirror encoder depth

    model = TerraMindTiM(pretrained=pretrained, **cfg, **kwargs)

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn_tim(state_dict, model)  # [vendored] also fill sampler.model.*
        model.load_state_dict(state_dict, strict=True)
    elif pretrained:
        state_dict_file = hf_hub_download(
            repo_id=PRETRAINED_WEIGHTS[variant]["hf_hub_id"], filename=PRETRAINED_WEIGHTS[variant]["hf_hub_filename"]
        )
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn_tim(state_dict, model)
        model.load_state_dict(state_dict, strict=True)
    else:
        logger.warning("TerraMind TiM model not pre-trained. Generation of TiM modalities will not work correctly.")

    return model
