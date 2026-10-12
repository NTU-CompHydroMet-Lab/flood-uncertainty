"""EDL model with a TerraMind encoder instead of the ml4floods UNet encoder.

``EDL_TerraMind_ML4FloodsModel`` subclasses :class:`EDL_ML4FloodsModel` and
overrides only:

* ``__init__``  – builds ``self.network`` with :func:`build_terramind_unet`
  instead of ``configure_architecture`` (same pattern as ``EDL_SAR_Unet``).
* ``configure_optimizers`` – two param groups so the pretrained encoder can
  use a smaller learning rate (``backbone_lr_mult``) and frozen params are
  not handed to Adam.

Everything else (EDL loss, KL annealing, validation metrics, wandb images,
zarr dump in ``validate_only``) is inherited untouched, so the training
script ``scripts/train/train_edl_terramind.py`` is a near-verbatim copy of
``train_edl.py``.
"""

from __future__ import annotations

import copy
import torch

from ml4floods.models.utils.configuration import AttrDict
from ml4floods.models.worldfloods_model import METRIC_MODE

from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.models.terramind_unet import build_terramind_unet


def _with_unet_architecture(model_params: AttrDict) -> AttrDict:
    """Copy of ``model_params`` whose ``model_type`` the parent's ``configure_architecture`` can build.

    The EDL parents build ``self.network`` with ``configure_architecture``, which does not know
    ``edl_terramind``. Running the unchanged parent constructor on this copy keeps every piece of
    parent state in sync with upstream (task mode, epoch confusion matrices, ...); the throwaway UNet
    is then replaced by the TerraMind network.
    """
    params = copy.deepcopy(model_params)
    params["hyperparameters"]["model_type"] = "unet"
    return params


class EDL_TerraMind_ML4FloodsModel(EDL_ML4FloodsModel):
    """Optical EDL segmentation (cloud + water, or water only) on a TerraMind v1 encoder."""

    def __init__(self, model_params: AttrDict, normalized_data: bool = True):
        super().__init__(_with_unet_architecture(model_params), normalized_data=normalized_data)
        # record the real config (the parent saved the unet copy)
        self.save_hyperparameters({"model_params": model_params, "normalized_data": normalized_data})
        h_params_dict = model_params.get("hyperparameters", {})
        # EDL: 2 evidence channels per task
        self.network = build_terramind_unet(h_params_dict, n_class=2 * self.num_class)
        self.backbone_lr_mult = float(h_params_dict.get("backbone_lr_mult", 0.1))

    def configure_optimizers(self):
        backbone_params = [p for p in self.network.backbone.encoder.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.network.parameters() if p.requires_grad and id(p) not in backbone_ids]

        groups = [{"params": head_params, "lr": self.lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.lr * self.backbone_lr_mult})
        optimizer = torch.optim.Adam(groups, lr=self.lr)

        monitor = self.hparams["model_params"]["hyperparameters"]["metric_monitor"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=METRIC_MODE[monitor], factor=self.lr_decay, patience=self.lr_patience
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
