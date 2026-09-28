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
#
# ---
#
# This project includes code adapted from the original work by EPFL and Apple Inc.,
# licensed under the Apache License, Version 2.0.
# Source: https://github.com/apple/ml-4m/

# --- flood-uncertainty vendoring note ---------------------------------------
# Copied from terratorch 1.2.11 (IBM/terratorch, Apache-2.0), file
#   terratorch/models/backbones/terramind/model/terramind.py (build_modality_embeddings only)
# Reason: terratorch >= 1.0.1 requires torchgeo >= 0.7 (Python >= 3.11); this
# project is pinned to Python 3.10. Only the ViT *encoder* path is vendored.
# Edits relative to upstream are marked with `# [vendored]`.
# -----------------------------------------------------------------------------

from torch import nn

from .encoder_embeddings import ImageEncoderEmbedding


def build_modality_embeddings(modality_info, modalities, cond_modalities=None, img_size=None, dim=None,
                              patch_size=None):
    mod_embeddings = {}
    mod_name_mapping = {}
    for modality in modalities:
        # New modalities can be provided as {"name": <num_channels>}
        if isinstance(modality, dict):
            for key, value in modality.items():
                if isinstance(value, nn.Module):
                    mod_embeddings[key] = value
                elif isinstance(value, int):
                    mod_embeddings[key] = ImageEncoderEmbedding(
                        num_channels=value,
                        dim_tokens=dim,
                        image_size=img_size,
                        patch_size=patch_size,
                        sincos_pos_emb=True,
                    )
                else:
                    raise ValueError(
                        f"Modalities must be provided as a list of strings and dicts. "
                        f"The strings can be any pre-trained modality: "
                        f"RGB, S2L1C, S2L2A, S1RTC, S1GRD, DEM, LULC, NDVI, Coords. "
                        f"Dicts define new modalities with the format {{'<name>': <num_channels>}}."
                        f"Found {key}: {value} ({type(value)})"
                    )
                mod_name_mapping[key] = key
            continue

        # Cover multiple naming conventions
        modality_renamed = (
            modality.lower()
            .replace("s2", "sen2")
            .replace("s1", "sen1")
            .replace("location", "coords")
        )

        # Get modality key in MODALITY_INFO
        if modality in modality_info.keys():
            key = modality
        elif "sen2l2a" in modality_renamed:
            key = "untok_sen2l2a@224"
        elif "sen2l1c" in modality_renamed:
            key = "untok_sen2l1c@224"
        elif "sen1rtc" in modality_renamed:
            key = "untok_sen1rtc@224"
        elif "sen1" in modality_renamed:  # Defaults to S1GRD if not specified
            key = "untok_sen1grd@224"
        elif "rgb" in modality_renamed:
            key = "untok_sen2rgb@224"
        elif "dem" in modality_renamed:
            key = "untok_dem@224"
        elif "lulc" in modality_renamed:
            key = "tok_lulc@224"
        elif "ndvi" in modality_renamed:
            key = "tok_ndvi@224"
        elif "caption" in modality_renamed:
            key = "caption"
        elif "coord" in modality_renamed:
            key = "coords"
        else:
            raise NotImplementedError(f"Could not find modality {modality} in default modality info.")

        mod_info = modality_info[key]
        mod_embeddings[key] = mod_info["encoder_embedding"](image_size=img_size, dim_tokens=dim, **mod_info)
        mod_name_mapping[modality] = key  # Requires manual mapping for loading model weights

    return mod_embeddings, mod_name_mapping
