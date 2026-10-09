"""Fine-tune the optical water-only EDL model on paired Sen1 splits."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from flood_uncertainty.data.sen1floods11 import create_sen1floods11_loaders
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
from flood_uncertainty.utils.config_loader import load_mode_config


def load_matching_weights(model: torch.nn.Module, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    missing = model.load_pretrained_weights(state_dict)
    if missing:
        raise ValueError(f"Incomplete optical checkpoint: {missing}")
    print(f"Loaded all {len(model.state_dict())} optical weights from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configurations/edl_optical_sen1.json")
    parser.add_argument("--mode", choices=["train", "validate_only"], default="train")
    parser.add_argument("--resume_ckpt", default=None)
    args = parser.parse_args()

    config = load_mode_config(args.config, mode=args.mode)
    seed_everything(config.seed)
    if args.resume_ckpt:
        saved = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        saved_h = saved.get("hyper_parameters", {}).get("model_params", {}).get("hyperparameters", {})
        for key in ("task_mode", "num_channels", "num_classes", "channel_configuration", "model_type"):
            if saved_h.get(key) != config.model_params.hyperparameters.get(key):
                raise ValueError(f"Resume architecture mismatch: {key}")
    if (config.data_params.get("dataset_type") != "sen1floods11"
            or config.data_params.get("modality") != "optical"):
        raise ValueError("This entry requires the Sen1 optical configuration")
    if config.model_params.hyperparameters.get("task_mode") != "water_only":
        raise ValueError("Sen1 labels supervise only the water task")
    train_loader, val_loader, test_loader = create_sen1floods11_loaders(config.data_params)
    model = EDL_ML4FloodsModel(config.model_params, normalized_data=True)

    pretrained_path = config.model_params.get("pretrained_path")
    if pretrained_path and not args.resume_ckpt:
        load_matching_weights(model, pretrained_path)

    experiment_path = os.path.join(config.model_params.model_folder, config.experiment_name)
    monitor = config.model_params.hyperparameters.metric_monitor
    os.makedirs(experiment_path, exist_ok=True)
    wandb_logger = False
    if config.get("wandb_enabled", False):
        wandb_logger = WandbLogger(
            name=config.experiment_name,
            project=config.wandb_project,
            entity=config.get("wandb_entity"),
            save_dir=experiment_path,
            log_model=config.get("wandb_log_model", False),
        )
        wandb_logger.log_hyperparams(dict(config))

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
        logger=wandb_logger,
    )

    if args.mode == "validate_only":
        trainer.validate(model, val_loader, ckpt_path=args.resume_ckpt,
                         weights_only=False if args.resume_ckpt else None)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt,
                    weights_only=False if args.resume_ckpt else None)


if __name__ == "__main__":
    main()
