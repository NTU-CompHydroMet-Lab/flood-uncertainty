# Sen1Floods11 SAR fine-tuning

`flood_uncertainty/data/sen1floods11.py` uses a shared S1/S2/label manifest.
The supplied root is resolved to its `v1.1` directory. Existing hand-labeled
splits contain 252 train, 89 validation, and 90 test IDs, with no ID overlap
and no missing S1/S2/label files. Bolivia holdout is not merged into these splits.

The supplied 512x512 tiles are split into four aligned 256x256 windows,
without resizing or padding. Train/val/test contain 1008/356/360 windows.
The original sample IDs define the split, preventing cross-split crop leakage.
Training shuffles samples; validation/test do not. Augmentation is optional
and disabled in the initial configuration.

SAR reads S1GRDHand in VV/VH order and follows the official Train.ipynb:
fill NaN with zero, clip dB to [-50, 1], scale as (x + 50) / 51, then
standardize using mean=[0.6851, 0.5235], std=[0.0820, 0.1102]. No conversion
to linear power or KuroSiwo clamp/statistics is used. Invalid image/label
pixels remain excluded by mask=0; -1/0/1 labels become 0/1/2.
These normalization constants are fixed in the loader; sar_units must be db.

Sen1 does not provide KuroSiwo's two pre-event acquisitions: a six-channel
checkpoint is incompatible with the two-channel input, and mismatched
checkpoint tensor shapes are rejected rather than silently skipped.

KuroSiwo is Sentinel-1 GRD (or its separate SLC variant), not Sentinel-2 L1C.
The local `KuroSiwo/configs/grd_preprocessing.xml` includes calibration,
Lee Sigma speckle filtering and terrain correction. Sen1 is also GRD but
uses the Earth Engine preprocessing pipeline; numeric conversion does not
make these upstream pipelines identical. Sen1 optical defaults to S2L1CHand,
selects the existing optical band configuration and normalization, and returns
a dummy invalid cloud channel plus the water channel for `water_only` models.
The optical entry point is `scripts/train/train_edl_optical_sen1.py`.

Sources:
- https://github.com/cloudtostreet/Sen1Floods11/blob/master/README.md
- https://github.com/cloudtostreet/Sen1Floods11/blob/master/Train.ipynb
- https://papers.nips.cc/paper_files/paper/2024/file/43612b0662cb6a4986edf859fd6ebafe-Paper-Datasets_and_Benchmarks_Track.pdf

Set `shared.model_params.pretrained_path` in `configurations/edl_sar_sen1.json`
to the desired two-channel SAR checkpoint, then run:

```sh
python scripts/train/train_edl_sar.py --config configurations/edl_sar_sen1.json
```

This starts a new fine-tuning optimizer/scheduler and epoch count using the
checkpoint weights. For full-state continuation use `--resume_ckpt path.ckpt`
with a compatible model and a `max_epochs` above the saved epoch. The SAR configuration selects the existing two-channel KuroSiwo checkpoint.

## Optical fine-tuning

```sh
uv run python scripts/train/train_edl_optical_sen1.py --config configurations/edl_optical_sen1.json --mode train
```

The independent Sen1 optical configuration uses S2L1CHand, six bands
(B2/B3/B4/B8/B11/B12), 256 windows, existing paired splits, batch size 4,
30 epochs and W&B. It loads `artifacts/models/edl_optical/checkpoint/last.ckpt`
into the water-only optical EDL model, resetting optimizer/scheduler and epoch.
Optical input uses the existing pretrained WorldFloods S2 statistics on
TOA reflectance scaled by 10000. The official Sen1 Train.ipynb normalization
is SAR-specific and is not applied to optical bands.

Use `--mode validate_only --resume_ckpt <optical-sen1.ckpt>` for validation,
or `--resume_ckpt <optical-sen1.ckpt>` for complete training continuation.
The output directory is `artifacts/models/edl_optical_sen1floods11`.

## Epoch metrics

Optical and SAR log valid-pixel aggregate train/validation metrics to W&B:
`train_Global_*_land_water` and `val_Global_*_land_water` include Accuracy,
Precision, Recall, F1, mIoU, per-class IoU, TP/FP/FN/TN and valid pixel count.
`train_loss_epoch` averages training loss weighted by valid pixel count.
Validation prints the accumulated epoch confusion matrix (rows=prediction,
columns=GT). Original `val_bce_land_water` checkpoint monitoring is retained.
Training metrics use predictions as the model updates during the epoch.
