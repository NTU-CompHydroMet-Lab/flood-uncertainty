"""Fine-tune a TerraMind EDL model on Sen1Floods11 (optical, SAR, or paired optical + SAR).

``data_params.modality`` selects the model and the loader:

* ``"optical"``: :class:`EDL_TerraMind_ML4FloodsModel` (``task_mode: water_only``), Sen1 optical loader.
* ``"sar"``: :class:`EDL_TerraMind_SAR`, Sen1 SAR loader (official dB normalisation;
  ``backbone_s1_input.units`` must be ``"sen1_scaled_db_zscore"``).
* ``"paired"``: :class:`EDL_TerraMind_ML4FloodsModel` with ``backbone_modalities: ["S2L1C", "S1GRD"]``
  (one encoder, both modalities), paired loader from :mod:`flood_uncertainty.data.sen1floods11_paired`.

``model_params.pretrained_path`` (optional) initialises the model from a stage-1 checkpoint
(WorldFloods optical or KuroSiwo SAR). The input re-normalisation buffers (``network.backbone.renorm.*``)
are not loaded: they describe the stage-1 loader and the new model builds them from this config. A
cloud + water optical head (4 channels) is reduced to its water channels. Any other missing or
mismatched weight is an error.

Logging, checkpointing and early stopping are the same as ``train_edl_terramind_sar.py``.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from flood_uncertainty.data.sen1floods11 import create_sen1floods11_loaders
from flood_uncertainty.data.sen1floods11_paired import create_paired_sen1floods11_loaders
from flood_uncertainty.models.edl_terramind import EDL_TerraMind_ML4FloodsModel, EDL_TerraMind_SAR
from flood_uncertainty.utils.config_loader import load_mode_config

SEN1_SAR_MEAN, SEN1_SAR_STD = [0.6851, 0.5235], [0.0820, 0.1102]
RENORM_PREFIX = "network.backbone.renorm."
HEAD_KEYS = ("network.conv_last.weight", "network.conv_last.bias")


def load_stage1_weights(model: torch.nn.Module, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    current = model.state_dict()
    loaded, mismatched = {}, []
    for key, value in state_dict.items():
        if key not in current or key.startswith(RENORM_PREFIX):
            continue
        if value.shape == current[key].shape:
            loaded[key] = value
        elif key in HEAD_KEYS and value.shape[0] == 4 and value[2:4].shape == current[key].shape:
            loaded[key] = value[2:4]  # cloud + water head -> water channels
        else:
            mismatched.append(key)
    missing = [key for key in current if key not in loaded and not key.startswith(RENORM_PREFIX)]
    if mismatched or missing:
        raise ValueError(f"Stage-1 checkpoint does not fit: mismatched={mismatched} missing={missing}")
    model.load_state_dict({**current, **loaded})
    print(f"Loaded {len(loaded)}/{len(current)} weights from {path} (renorm buffers kept from config)")


def check_config(config) -> None:
    hp, dp = config.model_params.hyperparameters, config.data_params
    modality = dp.get("modality")
    if dp.get("dataset_type") != "sen1floods11":
        raise ValueError("This entry requires a Sen1Floods11 configuration")
    if hp.get("model_type") != "edl_terramind":
        raise ValueError("This entry trains TerraMind models only")
    modalities = list(hp.get("backbone_modalities", []))
    if modality in ("sar", "paired"):
        s1 = hp.get("backbone_s1_input") or {}
        if (s1.get("units") != "sen1_scaled_db_zscore" or list(s1.get("mean", [])) != SEN1_SAR_MEAN
                or list(s1.get("std", [])) != SEN1_SAR_STD):
            raise ValueError("backbone_s1_input must describe the Sen1 official SAR normalisation")
    if modality == "sar":
        if modalities != ["S1GRD"]:
            raise ValueError("SAR fine-tune expects backbone_modalities ['S1GRD']")
    elif modality in ("optical", "paired"):
        if hp.get("task_mode") != "water_only":
            raise ValueError("Sen1 labels supervise only the water task")
        if hp.get("channel_configuration") != (dp.get("optical_channel_configuration") if modality == "paired"
                                                else dp.get("channel_configuration")):
            raise ValueError("Model and loader optical channel_configuration differ")
        expected = ["S2L1C"] if modality == "optical" else ["S2L1C", "S1GRD"]
        if modalities != expected:
            raise ValueError(f"{modality} fine-tune expects backbone_modalities {expected}")
    else:
        raise ValueError(f"Unknown modality: {modality}")


def build(config):
    modality = config.data_params.modality
    if modality == "paired":
        loaders = create_paired_sen1floods11_loaders(config.data_params)
    else:
        loaders = create_sen1floods11_loaders(config.data_params)
    if modality == "sar":
        model = EDL_TerraMind_SAR(config.model_params, normalized_data=False)
    else:
        model = EDL_TerraMind_ML4FloodsModel(config.model_params, normalized_data=True)
    return model, loaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "validate_only"], default="train")
    parser.add_argument("--resume_ckpt", default=None)
    args = parser.parse_args()

    config = load_mode_config(args.config, mode=args.mode)
    seed_everything(config.seed)
    check_config(config)
    model, (train_loader, val_loader, _) = build(config)
    print(f"model: {type(model).__name__}; modality={config.data_params.modality}")

    pretrained_path = config.model_params.get("pretrained_path")
    if pretrained_path and not args.resume_ckpt:
        load_stage1_weights(model, pretrained_path)

    experiment_path = os.path.join(config.model_params.model_folder, config.experiment_name)
    monitor = config.model_params.hyperparameters.metric_monitor

    loggers = []
    wandb_mode = os.environ.get("WANDB_MODE") or config.get("wandb_mode", "offline")
    if wandb_mode != "disabled":
        os.makedirs(experiment_path, exist_ok=True)
        wandb_logger = WandbLogger(
            name=config.experiment_name,
            project=config.wandb_project,
            entity=config.get("wandb_entity"),
            save_dir=experiment_path,
            offline=(wandb_mode == "offline"),
            log_model=config.get("wandb_log_model", False),
        )
        wandb_logger.log_hyperparams(dict(config))
        loggers.append(wandb_logger)  # first: EDL_*.log_images checks isinstance(self.logger, WandbLogger)
    loggers.append(CSVLogger(save_dir=experiment_path, name="lightning_logs"))

    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(experiment_path, "checkpoint"), save_top_k=20, save_last=True,
                        monitor=monitor, mode="min"),
        EarlyStopping(monitor=monitor, patience=config.model_params.hyperparameters.early_stopping_patience, mode="min"),
    ]
    trainer = Trainer(
        default_root_dir=experiment_path,
        callbacks=callbacks,
        accelerator="gpu" if config.gpus is not None else "cpu",
        devices=[int(config.gpus)] if config.gpus is not None else "auto",
        max_epochs=config.model_params.hyperparameters.max_epochs,
        check_val_every_n_epoch=config.model_params.hyperparameters.val_every,
        logger=loggers,
    )

    if args.mode == "validate_only":
        trainer.validate(model, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    main()
