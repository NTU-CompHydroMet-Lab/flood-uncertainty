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

import numpy as np
import pytorch_lightning as pl
import torch

from ml4floods.data.worldfloods.configs import COLORS_WORLDFLOODS_INVCLEARCLOUD, COLORS_WORLDFLOODS_INVLANDWATER
from ml4floods.models.utils.configuration import AttrDict
from ml4floods.models.worldfloods_model import METRIC_MODE

from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.models.terramind_unet import build_terramind_unet


class EDL_TerraMind_ML4FloodsModel(EDL_ML4FloodsModel):
    """Multi-output (cloud, water) EDL segmentation on a TerraMind v1 encoder."""

    def __init__(self, model_params: AttrDict, normalized_data: bool = True):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters({"model_params": model_params, "normalized_data": normalized_data})

        h_params_dict = model_params.get("hyperparameters", {})
        self.num_class = h_params_dict.get("num_classes", 2)
        assert self.num_class == 2, "Expected 2 output classes"

        self.pos_weight = h_params_dict.get("pos_weight", [1 for _ in range(self.num_class)])
        self.weight_problem = h_params_dict.get("weight_problem", [0.2, 0.8])
        if len(self.weight_problem) != self.num_class:
            raise ValueError("weight_problem must contain one weight per output task")
        self.annealing_mode = h_params_dict.get("annealing_mode", "fixed")
        self.annealing_coefficient = h_params_dict.get("annealing_coefficient", 0.5)
        self.annealing_step = h_params_dict.get("annealing_step", 10)

        # EDL: 2 evidence channels per task -> 2 * n output channels (same as parent)
        num_output_channels = 2 * self.num_class
        self.network = build_terramind_unet(h_params_dict, n_class=num_output_channels)
        self.normalized_data = normalized_data

        self.lr = h_params_dict.get("lr", 1e-4)
        self.lr_decay = h_params_dict.get("lr_decay", 0.5)
        self.lr_patience = h_params_dict.get("lr_patience", 2)
        self.backbone_lr_mult = float(h_params_dict.get("backbone_lr_mult", 0.1))

        self.label_names = np.array(h_params_dict["label_names"])
        assert self.label_names.shape == (2, 3), "Unexpected label names, expected: {}".format(
            [["invalid", "clear", "cloud"], ["invalid", "land", "water"]]
        )
        self.colormaps = {0: COLORS_WORLDFLOODS_INVCLEARCLOUD, 1: COLORS_WORLDFLOODS_INVLANDWATER}

        # zarr dump support (validate_only), identical to parent
        self.val_only = model_params.get("val_only", False)
        self.zarr_save_path = model_params.get("zarr_save_path", None)
        self.zarr_item_count = 0
        self.zarr_initialized = False

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
