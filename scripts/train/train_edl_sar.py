"""Train or validate the KuroSiwo SAR EDL model."""

import argparse
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from flood_uncertainty.data.kurosiwo import create_kurosiwo_loaders
from flood_uncertainty.models.edl import EDL_SAR_Unet
from flood_uncertainty.utils.config_loader import load_mode_config


def load_matching_weights(model: torch.nn.Module, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and value.shape == current[key].shape
    }
    model.load_state_dict({**current, **compatible})
    print(f"Loaded {len(compatible)}/{len(state_dict)} compatible weights from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configurations/edl_sar.json")
    parser.add_argument("--mode", choices=["train", "validate_only"], default="train")
    parser.add_argument("--resume_ckpt", default=None)
    args = parser.parse_args()

    config = load_mode_config(args.config, mode=args.mode)
    seed_everything(config.seed)
    train_loader, val_loader, test_loader = create_kurosiwo_loaders(config.data_params)
    model = EDL_SAR_Unet(config.model_params, normalized_data=False)

    pretrained_path = config.model_params.get("pretrained_path")
    if pretrained_path:
        load_matching_weights(model, pretrained_path)

    experiment_path = os.path.join(config.model_params.model_folder, config.experiment_name)
    monitor = config.model_params.hyperparameters.metric_monitor
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
        logger=False,
    )

    if args.mode == "validate_only":
        trainer.validate(model, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    main()
