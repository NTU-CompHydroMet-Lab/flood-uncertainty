"""Vendored TerraMind v1 ViT encoder (from terratorch 1.2.11, Apache-2.0).

Why vendored instead of ``uv add terratorch``: every terratorch release that
ships TerraMind (>= 1.0.1) requires ``torchgeo >= 0.7``, which requires
Python >= 3.11, while this project is pinned to ``>=3.10,<3.11``. terratorch
1.2.x additionally requires ``numpy >= 2.2`` against the project's
``numpy == 1.26.4``. Vendoring ~1.5k lines of encoder code keeps the lock
file untouched. See ``docs/terramind_edl_integration_log.md`` §3.7.

Contents (all under the upstream Apache-2.0 licence, see ``LICENSE``):
  tm_utils.py            transformer blocks, sin-cos pos-emb, interpolation   (verbatim)
  encoder_embeddings.py  ImageEncoderEmbedding & co.                          (verbatim)
  modality_info.py       MODALITY_INFO, untokenized image modalities only     (trimmed)
  modality_embeddings.py build_modality_embeddings                            (trimmed)
  terramind_vit.py       TerraMindViT                                         (tokenizer removed)
  factory.py             build_terramind_vit + weights / bands helpers        (adapted)

Only :func:`build_terramind_vit` is meant to be imported from outside.
"""

from .factory import PRETRAINED_BANDS, PRETRAINED_WEIGHTS, VARIANTS, build_terramind_vit  # noqa: F401
from .terramind_vit import TerraMindViT  # noqa: F401
