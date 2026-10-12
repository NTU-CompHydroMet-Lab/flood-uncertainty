"""Train or validate a SAR EDL model (UNet or TerraMind encoder) on KuroSiwo / Sen1Floods11.

Same CLI, data loading, checkpointing and early stopping as ``train_edl_sar.py``. Differences:

* ``model_params.hyperparameters.model_type == "edl_terramind"`` builds :class:`EDL_TerraMind_SAR`
  (TerraMind S1 encoder); anything else builds :class:`EDL_SAR_Unet`, so the UNet comparison run
  goes through the identical script.
* Logging: W&B (``shared.wandb_mode``: ``"offline"`` default | ``"online"`` | ``"disabled"``; env
  ``WANDB_MODE`` overrides) plus an always-on CSVLogger, so ``lightning_logs/version_0/metrics.csv``
  exists for training curves and checkpoint selection.
* For TerraMind, ``backbone_s1_input`` (how the loader normalised VV/VH) must match the loader's
  ``data_mean`` / ``data_std``; a mismatch would silently feed wrong dB values to the encoder.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from flood_uncertainty.data.kurosiwo import create_kurosiwo_loaders
from flood_uncertainty.data.sen1floods11 import create_sen1floods11_loaders
from flood_uncertainty.models.edl import EDL_SAR_Unet
from flood_uncertainty.models.edl_terramind import EDL_TerraMind_SAR
from flood_uncertainty.utils.config_loader import load_mode_config


def load_matching_weights(model: torch.nn.Module, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    current = model.state_dict()
    mismatched = [key for key, value in state_dict.items()
                  if key in current and value.shape != current[key].shape]
    if mismatched:
        raise ValueError(f"Checkpoint architecture mismatch: {mismatched}")
    compatible = {key: value for key, value in state_dict.items() if key in current}
    model.load_state_dict({**current, **compatible})
    print(f"Loaded {len(compatible)}/{len(state_dict)} compatible weights from {path}")


def check_s1_input_matches_loader(config) -> None:
    hp = config.model_params.hyperparameters
    s1 = hp.get("backbone_s1_input")
    dp = config.data_params
    if not s1 or s1.get("units") != "linear_zscore":
        return
    if dp.get("dataset_type", "kurosiwo") != "kurosiwo" or dp.get("scale_input") != "normalize":
        raise ValueError("backbone_s1_input units 'linear_zscore' expects the KuroSiwo loader with scale_input='normalize'")
    if list(s1["mean"]) != list(dp.data_mean) or list(s1["std"]) != list(dp.data_std):
        raise ValueError(
            f"backbone_s1_input mean/std {s1['mean']}/{s1['std']} != loader data_mean/data_std "
            f"{list(dp.data_mean)}/{list(dp.data_std)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configurations/terramind_sar/tm_s1grd.json")
    parser.add_argument("--mode", choices=["train", "validate_only"], default="train")
    parser.add_argument("--resume_ckpt", default=None)
    args = parser.parse_args()

    config = load_mode_config(args.config, mode=args.mode)
    seed_everything(config.seed)
    is_terramind = config.model_params.hyperparameters.get("model_type") == "edl_terramind"
    if is_terramind:
        check_s1_input_matches_loader(config)

    dataset_type = config.data_params.get("dataset_type", "kurosiwo")
    if dataset_type == "sen1floods11":
        if config.data_params.get("modality", "sar") != "sar":
            raise ValueError("SAR training requires modality=sar")
        train_loader, val_loader, test_loader = create_sen1floods11_loaders(config.data_params)
    elif dataset_type == "kurosiwo":
        train_loader, val_loader, test_loader = create_kurosiwo_loaders(config.data_params)
    else:
        raise ValueError(f"Unknown SAR dataset_type: {dataset_type}")

    model_cls = EDL_TerraMind_SAR if is_terramind else EDL_SAR_Unet
    model = model_cls(config.model_params, normalized_data=False)
    print(f"model: {model_cls.__name__}")

    pretrained_path = config.model_params.get("pretrained_path")
    if pretrained_path and not args.resume_ckpt:
        load_matching_weights(model, pretrained_path)

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
            save_dir=experiment_path,  # -> {experiment_path}/wandb/
            offline=(wandb_mode == "offline"),
            log_model=config.get("wandb_log_model", False),
        )
        wandb_logger.log_hyperparams(dict(config))
        loggers.append(wandb_logger)  # first: EDL_*.log_images checks isinstance(self.logger, WandbLogger)
    loggers.append(CSVLogger(save_dir=experiment_path, name="lightning_logs"))
    print(f"wandb_mode={wandb_mode}; loggers={[type(l).__name__ for l in loggers]}")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(experiment_path, "checkpoint"),
            save_top_k=20,
            save_last=True,
            monitor=monitor,
            mode="min",
        ),
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
