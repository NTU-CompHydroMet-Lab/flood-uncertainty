"""Foundation-model backbones for flood-uncertainty.

Only the TerraMind wrapper lives here for now. See
``docs/foundation_model_integration.md`` §3 for why no abstract
``BackboneProtocol`` is defined yet.
"""

from flood_uncertainty.models.backbones.terramind_backbone import (  # noqa: F401
    TerraMindBackbone,
    terramind_bands_from_channel_configuration,
)
