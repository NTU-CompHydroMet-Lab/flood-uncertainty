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
#   terratorch/models/backbones/terramind/model/modality_info.py (untokenized image modalities only)
# Reason: terratorch >= 1.0.1 requires torchgeo >= 0.7 (Python >= 3.11); this
# project is pinned to Python 3.10. Only the ViT *encoder* path is vendored.
# Edits relative to upstream are marked with `# [vendored]`.
# -----------------------------------------------------------------------------

import hashlib
from functools import partial

from .encoder_embeddings import ImageEncoderEmbedding


def generate_uint15_hash(seed_str):  # [vendored] copied from terramind/utils.py
    """Generates a hash of the seed string as an unsigned int15 integer"""
    return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (2**15)


MODALITY_INFO = {
    "untok_sen2l2a@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=12),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 12,
        "id": generate_uint15_hash("untok_sen2l2a@224"),
        "path": "S2L2A_untokenized",
    },
    "untok_sen2l1c@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=13),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 13,
        "id": generate_uint15_hash("untok_sen2l1c@224"),
        "path": "S2L1C_untokenized",
    },
    "untok_sen2rgb@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=13),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 3,
        "id": generate_uint15_hash("untok_sen2rgb@224"),
        "path": "S2RGB_untokenized",
    },
    "untok_sen1grd@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("untok_sen1grd@224"),
        "path": "S1GRD_untokenized",
    },
    "untok_sen1rtc@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=2),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 2,
        "id": generate_uint15_hash("untok_sen1rtc@224"),
        "path": "S1RTC_untokenized",
    },
    "untok_dem@224": {  # untokenized version
        "input_size": 224,
        "patch_size": 16,
        "encoder_embedding": partial(ImageEncoderEmbedding, num_channels=1),
        "decoder_embedding": None,
        "min_tokens": 0,
        "max_tokens": None,  # Will be set to 196
        "type": "img",
        "num_channels": 1,
        "id": generate_uint15_hash("untok_dem@224"),
        "path": "DEM_untokenized",
    },
}
