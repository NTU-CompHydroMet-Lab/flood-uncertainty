# TerraMind encoder integration into the EDL model: implementation log

> Branch: `feat/terramind-edl-backbone` (made from `dev`)
> Basis: [`foundation_model_integration.md`](./foundation_model_integration.md) (in this document, "the plan")
> Goal: Replace the UNet **encoder** of `EDL_ML4FloodsModel` with TerraMind v1 base.
> Do not change the decoder / EDL head / loss / data pipeline / Trainer. Train and test with the pattern of `train_edl.py`.

This document records each step: what we did, why we did it, and how we verified it. §4 lists the differences from the plan.

---

## 1. Environment check (before the work)

| Item | Result |
|---|---|
| Python / torch / lightning | 3.10 / 2.10.0 / pytorch-lightning 2.6.0 (locked in `pyproject.toml`) |
| GPU | 1× RTX 3080 10 GB |
| TerraMind weights | The HuggingFace cache already has `ibm-esa-geospatial/TerraMind-1.0-base/TerraMind_v1_base.pt` |
| Network | Connection to huggingface.co is available |
| Data | The `chlunchen-10030` path in `configurations/edl.json` does **not exist** on this machine. The train split of the `cjchen-10025` copy has **no S2 images**. `cynthia-10040/Data/worldfloods_v2_smoke_source` has the full train S2 (475 images; val/test are symlinks to cjchen-10025), but the directory is read-only. The training script writes `train_test_split_from_csv.json` into the data root. Thus we made a writable symlink root in `/home/NAS/house/ycchen-10014/data/worldfloods_v2` (train/val/test each point to the sources above, and `dataset_metadata.csv` is copied). The config points to this root |
| terratorch | Not installed. The latest version on PyPI is 1.2.13. It requires `numpy>=2.2`, which conflicts with the project `numpy==1.26.4` (plan risk #1 occurred) |

---

## 2. List of changes

### New files

| File | Content | Plan section |
|---|---|---|
| `flood_uncertainty/models/backbones/__init__.py` | Entry point of the subpackage | §4 |
| `flood_uncertainty/models/backbones/terramind_vendored/` | Source code of the TerraMind ViT encoder from terratorch 1.2.11 (Apache-2.0, with LICENSE and a provenance header in each file): `tm_utils.py`, `encoder_embeddings.py` (word for word), `modality_info.py`, `modality_embeddings.py` (reduced), `terramind_vit.py` (tokenizer removed), `factory.py` (`build_terramind_vit` = the equivalent of `BACKBONE_REGISTRY.build`) | §6 (§3.7 gives the reason for vendoring) |
| `flood_uncertainty/models/backbones/terramind_backbone.py` | `TerraMindBackbone`: dict in, pyramid list out. It exposes `out_channels` / `downsample_ratios`. It has three pure `nn.Module` helpers (`SelectIndices`, `ReshapeTokensToImage`, `LearnedInterpolateToPyramidal`), `InputRenormalizer`, and the band name mapping | §3.1, §5, §6 |
| `flood_uncertainty/models/terramind_unet.py` | `TerraMindUNet`: TerraMind encoder + a decoder with the same structure as the ml4floods `UNet` (`layer_factory.double_conv`) + a 1×1 head. `forward(Tensor) -> Tensor`, the same interface as `UNet` | §0, §5 |
| `flood_uncertainty/models/edl_terramind.py` | `EDL_TerraMind_ML4FloodsModel(EDL_ML4FloodsModel)`: It overrides only `__init__` (builds `self.network`) and `configure_optimizers` (smaller lr for the backbone, skips frozen parameters) | §9 |
| `configurations/edl_terramind.json` | Based on `edl.json`, with added `backbone_*` / `decoder_channels` fields | §7 |
| `scripts/train/train_edl_terramind.py` | A word-for-word copy of `train_edl.py`. Only the imports, the model construction and the default config are different | §8 |
| `scripts/smoke/smoke_train_edl_terramind.sh` | Wraps `smoke_train.sh`. If the train split has no S2, it uses one val tile as train (only for the pipeline check) | §8 |
| `tests/test_terramind_backbone.py` | Band mapping, token→image coordinate order, pyramid stride, re-normalization values, `out_channels`/`downsample_ratios` agree with the real output, freeze, an offline ckpt does not start an HF download | F2, F3 |
| `tests/test_edl_terramind_smoke.py` | Forward + backward with random initialization, output shape `(B, 4, H, W)`, range of EDL probability/uncertainty, optimizer groups | F4 |
| `docs/terramind_edl_integration_log.md` | This document | — |

### Changed files (minimum changes)

| File | Change | Reason |
|---|---|---|
| `scripts/smoke/smoke_train.sh` | Added an `EDL_TERRAMIND)` branch (4 lines) and a comment to `case` | Plan §8 specifies to use the existing dispatch |
| `flood_uncertainty/inference/infer.py` | Added a `model_type == "EDL-TERRAMIND"` branch to `load_model` (+import) | Lets the inference entry point in README 4.3 load the TerraMind checkpoint |
| `scripts/inference/run_inference.py` | Added `EDL-TERRAMIND` to `--model_type`, added `--config_edl_terramind`, and the `used_EDL` check includes the new type | Same as above |
| `scripts/eval/eval_metrics.py` | Added `EDL-TERRAMIND` to `--model_type`, and added the related config to `MODEL_CONFIG_PATHS` | Evaluation entry point in README 4.4 |
| `flood_uncertainty/metrics/segmentation.py` | The band dispatch of `plot_spatial_confusion_matrix` treats `EDL-TERRAMIND` the same as `EDL` (1 line) | The output band layout is the same as EDL |
| `scripts/smoke/smoke_infer_eval.sh` | Added `EDL-TERRAMIND)` to `case`. Added an optional `SMOKE_INFER_CHECKPOINT`, which goes to `--checkpoint` | Agrees with the smoke conventions |
| `README.md` | Added the EDL-TERRAMIND commands for training / inference / evaluation / smoke to each section. Added a §7 introduction | Keeps the documentation up to date |

### Files that did not change

`pyproject.toml`, `uv.lock` (**no new dependencies**, see §3.7), `flood_uncertainty/models/edl.py`, `losses/`, `metrics/`, `utils/config_loader.py`,
`ml4floods/**`, `scripts/train/train_edl.py` and the other three training scripts, and the four existing configs.

---

## 3. Design decisions and reasons

### 3.1 Why we use "a subclass that overrides `__init__`" and do not change `configure_architecture`

`configure_architecture` is in `ml4floods/models/worldfloods_model.py`, which is vendored upstream code.
If we add `elif architecture == 'edl_terramind'` there, the FM dependency goes into ml4floods. The project already has an example:
`EDL_SAR_Unet` uses the same method (it rewrites `__init__` and keeps the other methods). Thus we did the same. The cost is that part of `__init__` repeats the parent class.
The benefit is that `edl.py` has zero changes, and the EDL loss / metrics / zarr / wandb logic is fully shared.

### 3.2 Why `TerraMindUNet.forward` accepts a Tensor and not a dict

The parent class `training_step` / `validation_step` / `forward` all call `self.network(x)`, where x is `(B, C, H, W)`.
`TerraMindUNet` keeps the Tensor interface, and the dict translation (`{"S2L2A": x}`) occurs in one line inside it.
Thus we do not change any line of the parent class. This is the "translation is only one line" of plan §5.

### 3.3 Input re-normalization (`backbone_renormalize_input`, default `true`)

The ml4floods dataloader already applies a z-score with `SENTINEL2_NORMALIZATION` (WorldFloods statistics).
TerraMind pretraining used its own per-band mean/std (terratorch `v1_pretraining_mean/std`).
The two scales are near but not the same. `InputRenormalizer` first reverses the ml4floods z-score and then applies the TerraMind statistics.
Thus the pretrained patch embedding gets the distribution that it knows. We copied the statistics into this file (to prevent a dependency on private terratorch symbols).
A unit test compares them with values calculated by hand. If you turn off this switch, you can do an ablation (plan risk #5b).

### 3.4 Band subset

`bgriswirs` = B2, B3, B4, B8, B11, B12 → TerraMind `BLUE, GREEN, RED, NIR_BROAD, SWIR_1, SWIR_2`.
The `bands=` parameter of the terratorch factory takes the 6 related weight rows from the 12-band pretrained patch embedding.
It is not necessary to cut the weights manually. If the config contains B10 (L2A does not have it), the construction gives an error immediately.

### 3.5 Pyramid and decoder width

TerraMind is ViT-B/16: 12 layers, all at stride 16, 768 dimensions. We take layers `[2, 5, 8, 11]` (0-based).
`LearnedInterpolateToPyramidal` changes them to stride `[4, 8, 16, 32]`. Then a 1×1 lateral decreases them to
`[64, 128, 256, 512]`. These are the same skip widths as the ml4floods `UNet`, thus the decoder capacity agrees with the baseline.
The first skip of the CNN UNet is at stride 1, but the shallowest skip of the ViT version is at stride 4. Thus the end of the decoder has an added
bilinear ×4 + `double_conv(64, 64)` step to go back to full resolution.

### 3.5b Input of any size (inference tiling)

The ml4floods inference tiling applies padding to a multiple of 8 only for `unet` / `unet_dropout` in `SUBSAMPLE_MODULE`.
It does not know the new model_type, thus the tile size from a full image is not always a multiple of 16. `TerraMindUNet.forward`
applies its own reflection pad to the patch grid, and then crops the output back to the original size. The decoder upsampling now aligns to the real size of the skip
(the stride-32 layer is a max-pool, and an odd grid uses floor, thus ×2 does not match). A unit test checks `45×70 → 45×70`.
`ml4floods/**` did not change.

### 3.6 Optimizer

Adam with two param groups: decoder/head use `lr`, and the encoder uses `lr * backbone_lr_mult` (default 0.1).
If `backbone_freeze=true`, the encoder parameters are not in the optimizer, and the encoder stays in eval mode during `train()`.
The scheduler / monitor are the same as the parent class (`ReduceLROnPlateau` on `val_bce_land_water`).

### 3.8 Training log: W&B offline + CSV (added 2026-09-28)

`train_edl.py` hard-codes `setup_weights_and_biases = False`. `Trainer(logger=None)` keeps only the Lightning default
`CSVLogger` (`lightning_logs/version_N/metrics.csv`). The images from `self.log_images` are never sent, and the
`wandb_project` field of the config has no effect. `train_edl_terramind.py` changes this as follows:

- `config.shared.wandb_mode`: `"offline"` (default) | `"online"` | `"disabled"`. The env `WANDB_MODE` can override it.
- `offline` does not need a login or a network connection. The run is written to `{model_folder}/{experiment_name}/wandb/offline-run-*`
  (`save_dir=experiment_path`, so it no longer adds files to `wandb/` in the repo root). Later, use `wandb sync <that directory>` to upload it.
- For all W&B modes, a `CSVLogger` is also attached, so `metrics.csv` is still written. The order of `logger=[wandb, csv]` is fixed with wandb first.
  The reason is that `EDL_ML4FloodsModel` uses `isinstance(self.logger, WandbLogger)` to decide whether to log images, and `self.logger`
  returns the first logger when there are many loggers.
- `train_edl.py` did not change. To make the baseline the same, you only move the same code section into it.

### 3.7 Dependencies: why we finally **vendor** TerraMind and do not use `uv add terratorch`

Plan §6 specifies "add terratorch as a uv dependency and call only its backbone factory". Risk #1 requires a dry-run before the work. Test results:

| Attempt | Result |
|---|---|
| `uv pip install --dry-run terratorch` | Resolved to terratorch 1.2.11. It needs numpy 1.26.4 → 2.2.6, safetensors 0.7 → 0.8, and 35 more packages (torchgeo, lightning, diffusers, lightly, tensorboard…). torch / pytorch-lightning / timm do not change |
| `uv add terratorch==1.2.11` + `uv sync` after the pin change | The installation was successful, but `import terratorch` failed: `AttributeError: ResNet50_Weights.SENTINEL2_ALL_SOFTCON`. The cause: it got torchgeo 0.6.2 (the highest version for Python 3.10), but the code needs torchgeo ≥ 0.7 |
| Check the requirements of each version on PyPI | Except terratorch 1.0 (no TerraMind), all versions with TerraMind (1.0.1 ~ 1.2.13) require torchgeo ≥ 0.7.0. torchgeo 0.7+ requires **Python ≥ 3.11**, but the project has `requires-python = ">=3.10,<3.11"` |

Thus, if we do not upgrade Python, the terratorch method has no solution. There are two options:

1. Upgrade the project to Python 3.11 and resolve the 180 pins again. This has an effect on all existing pipelines. It is against "do not change the project structure" and plan §4 "zero risk to existing pipelines".
2. **Vendor** the TerraMind encoder source code (approximately 1.6k lines, Apache-2.0) into `backbones/terramind_vendored/`. The weights use the existing project `huggingface_hub` to download or read the cache.

We selected option 2. The dependency files did not change at all (we reverted the earlier numpy/safetensors changes and `uv.lock` with `git checkout`, and `uv sync` set the environment back to the locked state).
`terramind_backbone._build_terramind_encoder` is the only location that uses the factory. If the project later upgrades to 3.11 and you want to go back to terratorch, change only this function.

Differences between the vendored content and upstream: `terramind_vit.py` has no tokenizer (only tokenized modalities such as LULC/NDVI use it). `modality_info.py` keeps only untokenized image modalities.
All other files are word for word. Each file starts with a provenance comment, and changes have the `# [vendored]` mark.

---

## 4. Differences from the plan

| Plan | Implementation | Reason |
|---|---|---|
| The first case is `FMBaselineSegModel` (single task, CE loss) | The first case connects directly to **EDL** (two tasks, EDL loss) | User requirement: train and test with the pattern of `train_edl.py` |
| `out_channels` example `[96, 192, 384, 768]` | Real value `[768, 768, 768, 768]` | The example is a Swin-type backbone. TerraMind is a plain ViT, and all layers have the same width |
| No mention of re-normalization | Added `backbone_renormalize_input` | See §3.3 |
| `smoke_train_fm.sh` | `smoke_train_edl_terramind.sh` | The name agrees with the real model type. It also handles the data copy whose train split has no S2 |
| terratorch as a uv dependency | Vendor the encoder source code, zero new dependencies | Python 3.10 is not compatible with any terratorch version that has TerraMind (§3.7) |

---

## 5. Verification log

| # | Item | Command | Result |
|---|---|---|---|
| 1 | Syntax | `python -m py_compile` on all new files | Pass |
| 2 | Existing modules can be imported in the locked environment | `uv run python -c "import ml4floods.models.dataset_setup, flood_uncertainty.models.edl"` | Pass |
| 3 | Unit tests (including the existing `test_edl_annealing.py`) | `uv run --with pytest python -m pytest tests/ -q` | **21 passed** (helper 9, backbone 7, EDL smoke 5) |
| 4 | Loading of real weights (HF cache, `HF_HUB_OFFLINE=1`) | See below | 113/113 encoder keys come from the checkpoint. `encoder.0.attn.qkv.weight` is exactly equal to the ckpt. The 6-band subset of the patch embedding = ckpt columns `[1,2,3,7,10,11]`. 256×256 input → pyramid `(768,64²),(768,32²),(768,16²),(768,8²)`. Trainable parameters: 93.2M |
| 5 | Smoke train (GPU, 1 epoch, 1 tile/split, batch 2) | `SMOKE_TRAIN_GPUS=0 SMOKE_TRAIN_BATCH_SIZE=2 bash scripts/smoke/smoke_train_edl_terramind.sh` | **SMOKE TRAIN PASS**: 600 steps / 32 s (19 it/s, RTX 3080, 2.6 GB GPU memory). `epoch=0-step=600.ckpt` and `last.ckpt` are written to `artifacts/models/smoke/edl_terramind_v1_base_bgriswirs_smoke_20260928_140050/`. After training on 1 tile, land/water recall is 1.0 and precision is 0.018 (only a pipeline check) |
| 6 | `--mode validate_only` (with the `last.ckpt` from item 5 as `pretrained_path`) | `uv run python scripts/train/train_edl_terramind.py --config <smoke cfg> --mode validate_only --data_root artifacts/smoke/train/data_min` | 154/154 weights loaded, and validation completed. **The zarr write failed**: `ModuleNotFoundError: No module named 'zarr'`. `zarr` is not in `pyproject.toml`. `train_edl.py` uses the same inherited code and fails in the same way → an existing problem, not caused by this change (see §6) |
| 8 | Inference + evaluation smoke (README 4.3/4.4 procedure, with the `last.ckpt` from item 5) | `SMOKE_INFER_MODEL_TYPE=EDL-TERRAMIND SMOKE_INFER_CHECKPOINT=<last.ckpt> SMOKE_INFER_DATA_ROOT=artifacts/smoke/train/data_min bash scripts/smoke/smoke_infer_eval.sh` | **SMOKE PASS**: Inference on a full val tile (the tile is not a multiple of 16; the padding of §3.5b handles it). `*_output_EDL-TERRAMIND.tif` and `metrics_EDL-TERRAMIND.csv` are written. The first run stopped at the tiling size and then at the model_type dispatch in `segmentation.py`. After the fixes, it passed |
| 9 | Regression of the existing EDL inference + evaluation (changed infer/eval scripts) | `SMOKE_INFER_MODEL_TYPE=EDL ...` | SMOKE PASS |
| 10 | Training log (§3.8): W&B offline + CSV | `bash scripts/smoke/smoke_train_edl_terramind.sh` | **SMOKE TRAIN PASS** (600 steps / 31 s). Both `{exp}/wandb/offline-run-20260928_172421-*/` (contains `media/`, images are logged) and `{exp}/lightning_logs/version_0/metrics.csv` are made. No new run in `wandb/` in the repo root |
| 7 | Regression of the existing EDL pipeline (same environment, same mini dataset) | `uv run python scripts/train/train_edl.py --config <edl smoke cfg, pretrained_path=null> --mode train --data_root artifacts/smoke/train/data_min` | Pass: 600 steps / 22 s, exit 0. The existing UNet-EDL pipeline operates correctly in the same environment and on the same mini dataset (the environment and dependencies did not change, so this is expected) |

Script for item 4: Build `TerraMindBackbone(channel_configuration="bgriswirs", pretrained=True)`, and
compare it with the state dict from `hf_hub_download("ibm-esa-geospatial/TerraMind-1.0-base", "TerraMind_v1_base.pt")`.


---

## 6. Known problems and next steps

| Item | Description | Recommendation |
|---|---|---|
| The `zarr` package is missing | The zarr dump of the `validate_only` mode (`EDL_ML4FloodsModel._append_batch_to_zarr`; the TerraMind version inherits the same code) needs `zarr`, but it is not in `pyproject.toml`. The training and validation metrics are not affected | `uv add zarr` (independent of this change; the version needs a separate decision) |
| `configurations/edl.json` paths | `pretrained_path` and `data_params` point to `chlunchen-10030`, which does not exist on this machine. If you run `SMOKE_TRAIN_MODEL=EDL` directly, you get `FileNotFoundError` | This is an existing problem: the config is tied to one machine. This change did not change it. The baseline UNet weights are available at `/home/NAS/homes/cjchen-10025/flood-uncertainty/artifacts/models/WF2_unetv2_bgriswirs/model.pt` |
| Full training (F6) completed on 2026-09-29 | Two groups with a frozen encoder (random decoder / WF2 initialization), 30 epochs each. We compared them with the EDL baseline on val/test with full-image inference + retention/PAvPU. `edl_vs_terramind_comparison.md` has the results and conclusions. The unfrozen encoder (group D) is not run yet | `uv run python scripts/train/train_edl_terramind.py --mode train` (config default: 30 epochs, batch 16). We recommend that you first try one group with `backbone_freeze: true` and one with `false` |
| `batch_size` / precision | The estimate for ViT-B/16 at 256² with fp32 and batch 16 is < 10 GB. If OOM occurs, decrease the batch or add `precision="bf16-mixed"` to the Trainer (the script stays the same as `train_edl.py` on purpose and does not add this parameter) | Adjust the config for the real GPU memory |
| `scripts/analysis/*` not connected | Each analysis script (`analysis_S2.py`, `compute_pavpu.py`, each `plot_*_compare.py`) has its own model_type dictionary (name, color, file name suffix, uncertainty band list). `EDL-TERRAMIND` is not added | Add it to each script when full training results are available for comparison plots. The output bands are the same as EDL, so you can use the EDL band list |
| `BackboneProtocol` | Not made yet, as plan §3.2 says. `TerraMindBackbone` already agrees with its shape | Add it when there is a second backbone |

### How to run

```bash
# 單元測試
uv run --with pytest python -m pytest tests/ -q

# Smoke（1 epoch、1 tile/split）
SMOKE_TRAIN_GPUS=0 bash scripts/smoke/smoke_train_edl_terramind.sh
# 或走既有 dispatch（需 source root 有 train/S2）
SMOKE_TRAIN_MODEL=EDL_TERRAMIND SMOKE_TRAIN_SOURCE_DATA_ROOT=/home/NAS/house/ycchen-10014/data/worldfloods_v2 bash scripts/smoke/smoke_train.sh

# 完整訓練 / 驗證
uv run python scripts/train/train_edl_terramind.py --config configurations/edl_terramind.json --mode train
uv run python scripts/train/train_edl_terramind.py --config configurations/edl_terramind.json --mode validate_only
```
