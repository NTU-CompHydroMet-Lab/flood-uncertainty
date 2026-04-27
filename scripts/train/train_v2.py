# %%
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

# %%
# from ml4floods.models.config_setup import get_default_config
# import pkg_resources

# # Set filepath to configuration files
# # config_fp = 'path/to/worldfloods_template.json'
# config_fp = pkg_resources.resource_filename("ml4floods","models/configurations/worldfloods_template_v2.json")

# config = get_default_config(config_fp)
# config

# Change accordingly!
from flood_uncertainty.utils.config_loader import load_mode_config

DEFAULT_CONFIG_PATH = os.path.join(project_root, "configurations", "v2.json")
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
parser.add_argument("--data_root", default=None)
args, _ = parser.parse_known_args()
config = load_mode_config(args.config, mode="train")
data_root = args.data_root or config.data_params.path_to_splits

# Set this to the path of the metadata CSV from huggingface
CSV_PATH = os.path.join(data_root, "dataset_metadata.csv")

# Point this to the root of the dataset on the mounted bucket
JSON_PATH = os.path.join(data_root, "train_test_split_from_csv.json")



# %%
from pytorch_lightning import seed_everything
# Seed
seed_everything(config.seed)

# %% [markdown]
# # WorldFloodsV2 torch Dataset
# 
# * **Last Modified**: 31-07-2025
# * **Authors**: Gonzalo Mateo-García (from Ruben Cartuyvels feedback)
# ---
# 
# > E. Portalés-Julià, G. Mateo-García, C. Purcell, and L. Gómez-Chova [Global flood extent segmentation in optical satellite images](https://www.nature.com/articles/s41598-023-47595-7). _Scientific Reports 13, 20316_ (2023). DOI: 10.1038/s41598-023-47595-7.
# 
# 
# This example shows how to load the *WorldFloodsv2* dataset with a pytorch Dataset for training or inference purposes.
# 

# %% [markdown]
# 
# 
# ## Step 1: Download the v2 data from Hugging-Face 🤗
# 
# The WorldFloods v2 dataset is stored in Hugging-Face in the repository: [isp-uv-es/WorldFloodsv2](https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2/). 
# 
# To download the full dataset (~76GB) run:
# 
# ```
# huggingface-cli download --cache-dir /path/to/cachedir --local-dir /path/to/localdir/WorldFloodsv2 --repo-type dataset isp-uv-es/WorldFloodsv2
# ```
# 
# ## Step 2: Load the data with the `Dataset` on `ml4floods`

# %%
from ml4floods.models.dataset_setup import get_dataset
from typing import Any, Dict
import pandas as pd
import json
import os






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
for batch in train_dl:
    # print(batch)
    break


# %% [markdown]
# ## Step 3: Inspect the batch
# 
# The batch is a dictionary with the S2 13 band image normalized and the mask. 
# 
# The mask has 2 channels:
# * **Cloud channel**. With values 0 invalid, 1 clear and 2 cloud
# * **Water channel**. With values 0 invalid, 1 land and 2 water

# %%
print(batch.keys())
batch["image"].shape, batch["mask"].shape

# %% [markdown]
# ## Step 4: Plot the batch

# %%
from ml4floods.models import worldfloods_model
import matplotlib.pyplot as plt
from ml4floods.data.worldfloods import configs
from ml4floods.visualization import plot_utils


n_images=6
fig, axs = plt.subplots(4,n_images, figsize=(18,12),tight_layout=True,sharex=True,sharey=True)
# worldfloods_model.plot_batch(batch["image"][:n_images],axs=axs[0],max_clip_val=3500.)
# worldfloods_model.plot_batch(batch["image"][:n_images],bands_show=["B11","B8", "B4"],
#                              axs=axs[1],max_clip_val=4500.)
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
# worldfloods_model.plot_batch_output_v1(batch["mask"][:n_images, 0],axs=axs[2], show_axis=True)

cmap_preds_clouds, norm_preds_clouds, patches_preds_clouds = plot_utils.get_cmap_norm_colors(configs.COLORS_WORLDFLOODS_INVCLEARCLOUD,
                                                                        ["invalid","clear","cloud"])

for _i, (xi, ax) in enumerate(zip(batch["mask"][:n_images,0], axs[2])):
    ax.imshow(xi, cmap=cmap_preds_clouds, norm=norm_preds_clouds,
              interpolation='nearest')
    ax.axis("off")

    if _i == (len(batch["image"][:n_images,0])-1):
        ax.legend(handles=patches_preds_clouds,
                  loc='upper right')


cmap_preds_water, norm_preds_water, patches_preds_water = plot_utils.get_cmap_norm_colors(configs.COLORS_WORLDFLOODS_INVLANDWATER,
                                                                        ["invalid","land","water"])

for _i, (xi, ax) in enumerate(zip(batch["mask"][:n_images,1], axs[3])):
    ax.imshow(xi, cmap=cmap_preds_water, norm=norm_preds_water,
              interpolation='nearest')
    ax.axis("off")

    if _i == (len(batch["mask"][:n_images,0])-1):
        ax.legend(handles=patches_preds_water,
                  loc='upper right')

# %% [markdown]
# # Setup Model

# %%
config.model_params

# %%
from ml4floods.models.model_setup import get_model
# config.model_params.model_folder = "models"
# os.makedirs("models", exist_ok=True)
# config.model_params.test = False
# config.model_params.train = True
model = get_model(config.model_params)
# model

# %%
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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

print(f"The trained model will be stored in {config.model_params.model_folder}/{config.experiment_name}")

# %%
setup_weights_and_biases = True
if setup_weights_and_biases:
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    # UNCOMMENT ON FIRST RUN TO LOGIN TO Weights and Biases (only needs to be done once)
    # wandb.login()
    # run = wandb.init()

    # Specifies who is logging the experiment to wandb
    # config['wandb_entity'] = 'ml4floods'
    # Specifies which wandb project to log to, multiple runs can exist in the same project
    # config['wandb_project'] = 'worldfloods-project'

    

    wandb_logger = WandbLogger(
        name=config.experiment_name,
        project=config.wandb_project, 
        # entity=config.wandb_entity
    )
else:
    wandb_logger = None

# %%
from pytorch_lightning import Trainer

# config.gpus = '0'  # which gpu to use
config.gpus = config.gpus
# config.gpus = None # to not use GPU

# config.model_params.hyperparameters.max_epochs = 3 # train for maximum 4 epochs

use_gpu = config.gpus is not None

trainer = Trainer(
    fast_dev_run=False,
    logger=wandb_logger,
    callbacks=callbacks,
    default_root_dir=f"{config.model_params.model_folder}/{config.experiment_name}",
    accumulate_grad_batches=1,
    gradient_clip_val=0.0,
    # auto_lr_find=False,
    benchmark=False,
    accelerator='gpu' if use_gpu else 'cpu',  # 改用 accelerator
    devices=[int(config.gpus)] if use_gpu else 'auto',  # 改用 devices
    # gpus=config.gpus,
    max_epochs=config.model_params.hyperparameters.max_epochs,
    check_val_every_n_epoch=config.model_params.hyperparameters.val_every,
    # log_gpu_memory=None,
    # resume_from_checkpoint=None
)

# %%
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

# %%
