import sys
import os
import argparse
# 取得當前 notebook 的目錄
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
# 計算到專案根目錄的相對路徑
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
sys.path.insert(0, project_root)
print(f"專案根目錄: {project_root}")
from flood_uncertainty.utils.config_loader import load_mode_config
from ml4floods.models.dataset_setup import get_dataset
from typing import Any, Dict
import pandas as pd
import json
from ml4floods.models.model_setup import get_model
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings("ignore")

DEFAULT_CONFIG_PATH = os.path.join(project_root, "configurations", "dropout.json")
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
parser.add_argument("--mode", default="train", choices=["train", "validate_only"])
parser.add_argument("--data_root", default=None)
args, _ = parser.parse_known_args()
config = load_mode_config(args.config, mode=args.mode)
if args.mode == "validate_only":
    raise NotImplementedError("validate_only is currently supported in scripts/train/train_edl.py")
data_root = args.data_root or config.data_params.path_to_splits
# Set this to the path of the metadata CSV from huggingface
CSV_PATH = os.path.join(data_root, "dataset_metadata.csv")
# Point this to the root of the dataset on the mounted bucket
JSON_PATH = os.path.join(data_root, "train_test_split_from_csv.json")
# Seed
seed_everything(config.seed)

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

# set data params
config.data_params.loader_type = "local"
config.data_params.bucket_id = None
config.data_params.path_to_splits = data_root
config.data_params["download"] = {
    "train": False,
    "val": False,
    "test": False,
}
config.data_params.train_test_split_file = JSON_PATH
dm = get_dataset(config.data_params)
dm.prepare_data()
train_dl = dm.train_dataloader()
val_dl = dm.val_dataloader()

 

experiment_path = f"{config.model_params.model_folder}/{config.experiment_name}"

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{experiment_path}/checkpoint",
    save_top_k=5,
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
wandb_logger = WandbLogger(
    name=f"{config.experiment_name}",
    project=config.wandb_project, 
)

trainer = Trainer(
    fast_dev_run=False,
    logger=wandb_logger,
    callbacks=callbacks,
    default_root_dir=experiment_path,
    accumulate_grad_batches=1,
    gradient_clip_val=0.0,
    benchmark=False,
    accelerator='gpu' if config.gpus else 'cpu', 
    devices=[int(config.gpus)] if config.gpus else 'auto',  
    max_epochs=config.model_params.hyperparameters.max_epochs,
    check_val_every_n_epoch=config.model_params.hyperparameters.val_every,
)

model = get_model(config.model_params)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
wandb.finish()
