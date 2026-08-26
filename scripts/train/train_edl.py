import sys
import os
import argparse
# 計算到專案根目錄的相對路徑
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
print(f"專案根目錄: {project_root}")
import json
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from typing import Any, Dict
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.models.dataset_setup import get_dataset
# from ml4floods.models.model_setup import get_model
from ml4floods.models import worldfloods_model
from ml4floods.data.worldfloods import configs
from ml4floods.visualization import plot_utils
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
import torch



# =============================================================================
# Configuration Setup
# =============================================================================
DEFAULT_CONFIG_PATH = os.path.join(project_root, "configurations", "edl.json")
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
parser.add_argument("--mode", default="train", choices=["train", "validate_only"])
parser.add_argument("--data_root", default=None)
parser.add_argument(
    "--resume_ckpt",
    default=None,
    help="Lightning checkpoint to resume from (restores epoch, optimizer, and scheduler).",
)
args, _ = parser.parse_known_args()
config = load_mode_config(args.config, mode=args.mode)
data_root = args.data_root or config.data_params.path_to_splits

# Set this to the path of the metadata CSV from huggingface
CSV_PATH = os.path.join(data_root, "dataset_metadata.csv")
# Point this to the root of the dataset on the mounted bucket
JSON_PATH = os.path.join(data_root, "train_test_split_from_csv.json")

# Seed
seed_everything(config.seed)

# =============================================================================
# WorldFloodsV2 torch Dataset
# 
# * **Last Modified**: 31-07-2025
# * **Authors**: Gonzalo Mateo-García (from Ruben Cartuyvels feedback)
#
# Reference:
# E. Portalés-Julià, G. Mateo-García, C. Purcell, and L. Gómez-Chova 
# "Global flood extent segmentation in optical satellite images"
# Scientific Reports 13, 20316 (2023). DOI: 10.1038/s41598-023-47595-7.
#
# Dataset Download:
# The WorldFloods v2 dataset is stored in Hugging-Face: isp-uv-es/WorldFloodsv2
# To download the full dataset (~76GB) run:
# huggingface-cli download --cache-dir /path/to/cachedir --local-dir /path/to/localdir/WorldFloodsv2 --repo-type dataset isp-uv-es/WorldFloodsv2
# =============================================================================

def convert_metadata_csv_to_json() -> None:
    out: Dict[str, Any] = {}
    modalities = ["S2", "gt"]
    csv = pd.read_csv(CSV_PATH)

    for split in csv.split.unique():
        out[split] = {}
        files = csv[csv.split == split]["event id"]
        for mod in modalities:
            out[split][mod] = [
                os.path.join(data_root, split, mod, f"{fn}.tif")
                for fn in files.to_list()
            ]

    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)


convert_metadata_csv_to_json()

# Setup data parameters
config.data_params.loader_type = "local"
config.data_params.bucket_id = None
config.data_params.path_to_splits = data_root
config.data_params.train_test_split_file = JSON_PATH

# Load dataset
dm = get_dataset(config.data_params)
dm.prepare_data()
train_dl = dm.train_dataloader()
val_dl = dm.val_dataloader()

# Get sample batch
for batch in train_dl:
    break

# =============================================================================
# Inspect the batch
# 
# The batch is a dictionary with the S2 13 band image normalized and the mask.
# The mask has 2 channels:
# * **Cloud channel**: 0 invalid, 1 clear, 2 cloud
# * **Water channel**: 0 invalid, 1 land, 2 water
# =============================================================================
print(batch.keys())
print(f"Image shape: {batch['image'].shape}, Mask shape: {batch['mask'].shape}")

# =============================================================================
# Plot the batch
# =============================================================================
n_images = 6
fig, axs = plt.subplots(4, n_images, figsize=(18, 12), tight_layout=True, sharex=True, sharey=True)

worldfloods_model.plot_batch(
    batch["image"][:n_images],
    channel_configuration=config.data_params.channel_configuration,
    axs=axs[0],
    max_clip_val=3500.
)

worldfloods_model.plot_batch(
    batch["image"][:n_images],
    channel_configuration=config.data_params.channel_configuration,
    bands_show=["B11", "B8", "B4"],
    axs=axs[1],
    max_clip_val=4500.
)

cmap_preds_clouds, norm_preds_clouds, patches_preds_clouds = plot_utils.get_cmap_norm_colors(
    configs.COLORS_WORLDFLOODS_INVCLEARCLOUD,
    ["invalid", "clear", "cloud"]
)

for _i, (xi, ax) in enumerate(zip(batch["mask"][:n_images, 0], axs[2])):
    ax.imshow(xi, cmap=cmap_preds_clouds, norm=norm_preds_clouds, interpolation='nearest')
    ax.axis("off")
    if _i == (len(batch["image"][:n_images, 0]) - 1):
        ax.legend(handles=patches_preds_clouds, loc='upper right')

cmap_preds_water, norm_preds_water, patches_preds_water = plot_utils.get_cmap_norm_colors(
    configs.COLORS_WORLDFLOODS_INVLANDWATER,
    ["invalid", "land", "water"]
)

for _i, (xi, ax) in enumerate(zip(batch["mask"][:n_images, 1], axs[3])):
    ax.imshow(xi, cmap=cmap_preds_water, norm=norm_preds_water, interpolation='nearest')
    ax.axis("off")
    if _i == (len(batch["mask"][:n_images, 0]) - 1):
        ax.legend(handles=patches_preds_water, loc='upper right')

fig.savefig("batch_plot.png")

# =============================================================================
# Setup Model
# =============================================================================
# Use EDL_ML4FloodsModel for EDL uncertainty training
model = EDL_ML4FloodsModel(config.model_params, normalized_data=True)

if config.model_params.get("pretrained_path", None):
    pretrained_path = config.model_params.get("pretrained_path", None)
    loaded = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    if pretrained_path.endswith('.ckpt'):
        pretrained_dict = loaded.get('state_dict', loaded.get('model_state_dict', loaded))
    else:
        pretrained_dict = loaded
    
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() 
                     if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    skipped = len(pretrained_dict) - len(filtered_dict)
    print(f"Loaded model weights: {pretrained_path}")
    print(f"  Loaded: {len(filtered_dict)}/{len(pretrained_dict)} weights")
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} weights due to shape mismatch (will use random initialization)")
else:
    print("No pretrained model weights provided")

# Setup callbacks
experiment_path = f"{config.model_params.model_folder}/{config.experiment_name}"

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{experiment_path}/checkpoint",
    save_top_k=20,
    save_last=True,
    verbose=True,
    monitor=config.model_params.hyperparameters.metric_monitor,
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor=config.model_params.hyperparameters.metric_monitor,
    patience=config.model_params.hyperparameters.early_stopping_patience,
    strict=False,
    verbose=False,
    mode='min'
)

callbacks = [checkpoint_callback, early_stop_callback]

print(f"The trained model will be stored in {config.model_params.model_folder}/{config.experiment_name}")

# =============================================================================
# Setup Weights and Biases Logger
# =============================================================================
setup_weights_and_biases = False
if setup_weights_and_biases:
    # UNCOMMENT ON FIRST RUN TO LOGIN TO Weights and Biases (only needs to be done once)
    # wandb.login()
    wandb_logger = WandbLogger(
        name=config.experiment_name,
        project=config.wandb_project,
    )
else:
    wandb_logger = None

# =============================================================================
# Setup Trainer
# =============================================================================
use_gpu = config.gpus is not None

trainer = Trainer(
    fast_dev_run=False,
    logger=wandb_logger,
    callbacks=callbacks,
    default_root_dir=f"{config.model_params.model_folder}/{config.experiment_name}",
    accumulate_grad_batches=1,
    gradient_clip_val=0.0,
    benchmark=False,
    accelerator='gpu' if use_gpu else 'cpu',
    devices=[int(config.gpus)] if use_gpu else 'auto',
    max_epochs=config.model_params.hyperparameters.max_epochs,
    check_val_every_n_epoch=config.model_params.hyperparameters.val_every,
)

# =============================================================================
# Training
# =============================================================================
if not config.model_params.get("val_only", False):
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=args.resume_ckpt,
        # This is a trusted checkpoint produced by this training script. Lightning
        # checkpoints contain optimizer/config objects in addition to tensor weights.
        weights_only=False if args.resume_ckpt else None,
    )
else:
    trainer.validate(model, val_dl)
