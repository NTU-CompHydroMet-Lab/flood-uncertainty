import sys
import os
import argparse
import shutil
# 計算到專案根目錄的相對路徑
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
print(f"專案根目錄: {project_root}")
import json
import pandas as pd
import wandb
from typing import Any, Dict
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.models.dataset_setup import get_dataset
from flood_uncertainty.models.edl import EDL_ML4FloodsModel
import torch



# =============================================================================
# Configuration Setup
# =============================================================================
DEFAULT_CONFIG_PATH = os.path.join(project_root, "configurations", "edl.json")
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
parser.add_argument("--data_root", default=None)
args, _ = parser.parse_known_args()
config = load_mode_config(args.config, mode="train")
data_root = args.data_root or config.data_params.path_to_splits
experiment_path = f"{config.model_params.model_folder}/{config.experiment_name}"
os.makedirs(experiment_path, exist_ok=True)

# Set this to the path of the metadata CSV from huggingface
CSV_PATH = os.path.join(data_root, "dataset_metadata.csv")
# Point this to the root of the dataset on the mounted bucket
JSON_PATH = os.path.join(experiment_path, "train_test_split_from_csv.json")

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

for _batch in train_dl:
    break

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

original_config_copy_path = os.path.join(experiment_path, "original_config.json")
shutil.copyfile(args.config, original_config_copy_path)

run_meta_path = os.path.join(experiment_path, "run_meta.json")
with open(run_meta_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "config_path": os.path.abspath(args.config),
            "mode": "train",
            "data_root_override": args.data_root,
            "resolved_data_root": data_root,
        },
        f,
        indent=2,
    )

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
setup_weights_and_biases = True
if setup_weights_and_biases:
    # UNCOMMENT ON FIRST RUN TO LOGIN TO Weights and Biases (only needs to be done once)
    # wandb.login()
    wandb_logger = WandbLogger(
        name=config.experiment_name,
        project=config.wandb_project,
        entity=getattr(config, "wandb_entity", None),
    )
else:
    wandb_logger = None

# =============================================================================
# Setup Trainer
# =============================================================================
use_gpu = config.gpus is not None
resume_ckpt = config.resume_from_checkpoint or None

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
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=resume_ckpt)
