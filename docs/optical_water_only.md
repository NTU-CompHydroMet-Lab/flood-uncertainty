# Optical EDL water-only

Use the same model class and scripts for both optical modes. Existing configurations
without `task_mode` default to `cloud_water`; parameter names and the four-output
architecture are preserved for old checkpoints.

| Setting | cloud_water | water_only |
| --- | --- | --- |
| num_classes (task count) | 2 | 1 |
| Evidence channels | clear, cloud, land, water | land, water |
| Probability channels | cloud, water | water |
| GT on disk | cloud, water | cloud, water |

The model selects the existing GT water channel in training and validation. Its
invalid pixels remain excluded, including clouds already excluded in the GT.
Do not change GT files or relabel cloud pixels as land.

## Train

```sh
python scripts/train/train_edl.py --config configurations/edl_water.json
```

The new config starts from random weights (`pretrained_path: null`) and writes to
the separate `edl_water_only` experiment. Change the experiment name for each run.
To initialize from an old four-evidence EDL UNet checkpoint, set `pretrained_path`
to that checkpoint. Shared weights load unchanged; `network.conv_last` weights
and bias copy the old channels 2:4 (land/water). Other shape mismatches are reported
as randomly initialized parameters. A standard two-output non-EDL v2 model does
not have the same evidence semantics and is not a water-evidence conversion source.

Use `--resume_ckpt` only for a checkpoint with the same task mode and architecture.
Initialization from four evidence to two evidence is a new training run, not an
optimizer/epoch resume. For `--mode validate_only`, set `pretrained_path` to the
water-only checkpoint. Water-only Zarr exports contain water variables without
cloud variables; existing double-task Zarr exports retain their original layout.

## Infer

```sh
python scripts/inference/run_inference.py \
  --model_type EDL --config_edl configurations/edl_water.json \
  --checkpoint artifacts/models/edl_water_only/checkpoint/last.ckpt \
  --output_dir artifacts/results/edl_water_only
```

For Lightning EDL checkpoints, saved model hyperparameters determine the architecture
and task mode; the external config supplies runtime/data settings. Pure state dicts
require the matching external config and are loaded strictly.

Water-only classification is `0=invalid, 1=land, 2=water`, with water probability
strictly greater than `th_water`. Cloud and flood-trace postprocessing are disabled.
Cloud-covered pixels still receive predictions; excluded GT pixels remain excluded
from evaluation.

The seven EDL TIFF bands retain their existing order and descriptions:
classification, water DST uncertainty, water probability, land evidence, water
evidence, water aleatoric uncertainty, water epistemic uncertainty. Existing
`eval_metrics.py` can read these files without changes. Classification metadata
identifies the reduced class set. The generic plotting palette remains shared.

This mode applies to optical EDL. The sigmoid ensemble/MC-dropout workflows and
SAR model configuration remain separate and unchanged.

## Numerical regression

The water-only numerical tests pin
`artifacts/models/edl_optical/checkpoint/epoch=8-step=9225.ckpt` and compare against
`tests/fixtures/optical_water_only_epoch8_regression.pt`. This is an initial snapshot
of the current water-only implementation, not a comparison with the old cloud/water
model. Checkpoint SHA-256, epoch, step, and two-evidence architecture are checked.

```sh
uv run python -m pytest -c pyproject.toml -q tests/test_optical_water_only_regression.py tests/test_optical_water_only.py
```

Tests compare classification exactly and probability, evidence, three uncertainty
outputs, loss, the first parameter gradient, and post-update logits with
`rtol=1e-5, atol=1e-6`. Training uses the checkpoint optimizer state and epoch 8
annealing, selecting the water channel from unchanged two-channel GT. CPU threads
are fixed to one for both generation and testing. The test restores the previous
thread count afterward. Missing checkpoints skip the tests; missing baselines fail.

The baseline records PyTorch version and CPU thread count. Other hardware or
backend versions can still introduce floating-point differences.

To create a baseline when none exists:

```sh
uv run python scripts/tests/generate_optical_water_only_regression_baseline.py
```

The generator refuses to overwrite an existing baseline. Only replace the saved
fixture after reviewing and intentionally accepting a model behavior change.
