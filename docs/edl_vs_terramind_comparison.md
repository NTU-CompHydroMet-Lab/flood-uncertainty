# EDL with a UNet encoder and EDL with a TerraMind encoder: comparison

This document compares EDL flood segmentation models. The models use the same EDL head, loss, data and training hyperparameters.
The models are different only in the encoder and in the input modality.
For the implementation and the design decisions, refer to `terramind_edl_integration_log.md`.

The document uses ASD-STE100 (Simplified Technical English) where possible. It keeps technical names (for example IoU, checkpoint, bootstrap).

**Direction of the experiments**: We fine-tune the **full** TerraMind encoder (no frozen layers).
The TerraMind baseline is `tm_l1c13`. Each new experiment changes one variable of `tm_l1c13`.
The target to exceed is `unet_baseline`.
We use two test sets: WorldFloods v2 test (same dataset as the training data) and Sen1Floods11 (a different dataset; no model uses it for training).
Appendix A gives the results of the frozen encoder.
`tm_frozen_l2a6_wf2dec` and `tm_l2a6_wf2dec` showed that a WF2 decoder initialization does not give a constant improvement.
Thus, this document does not compare them. Their values are in this document at commit `e9cc0ba`.

## 0. Summary

All differences are differences of water IoU (ΔIoU). The interval after each difference is the 95% confidence interval (CI) of a paired bootstrap.

- **WorldFloods test (same dataset)**: All TerraMind models have a lower IoU than `unet_baseline`.
  The nearest model is `tm_l1c13_tim`: ΔIoU OVERALL −0.0106 [−0.0331, −0.0019].
- **Sen1Floods11 (different dataset)**: Most TerraMind models have a higher IoU than `unet_baseline`.
  On Sen1Floods11 test, `tm_l1c13_tim` gives +0.0390 [+0.0078, +0.0748]. This is the first model with a significant increase.
  `tm_l2a6` and `tm_l1c13_inskip` give +0.0368 and +0.0342, but their CIs include zero.
- **A possible cause**: The encoder and the decoder of `unet_baseline` start from WF2, a UNet that was trained on WorldFloods.
  Thus, the UNet has an advantage on its own dataset. On a different dataset, the general pre-training of TerraMind gives better generalization.
- **Changes to `tm_l1c13`**: TiM (`tm_l1c13_tim`) gives +0.0280 [+0.0045, +0.0512] on Sen1Floods11 test.
  The full-resolution input skip (`tm_l1c13_inskip`) gives +0.0232 [+0.0020, +0.0422]. On WorldFloods, the differences are not significant.
- **Uncertainty**: On WorldFloods test and on Sen1Floods11 test, `unet_baseline` has the highest retention AUC (or the same as the highest).
  TerraMind predicts more correctly on Sen1Floods11. But its uncertainty does not identify its errors better.
- **UNet retraining (§3.8)**: We trained the UNet again on our data and machine. `unet_bs64@ep8` gives the same result as `unet_baseline`
  (WorldFloods test −0.0002 [−0.0034, +0.0011]). With batch size 16, `unet_bs16@ep9` is worse:
  WorldFloods test −0.0108 [−0.0243, −0.0004], Sen1Floods11 test −0.0663 [−0.1021, −0.0305].
  With the same batch size 16, TerraMind is not significantly different on WorldFloods test (`tm_l1c13_tim` +0.0003 [−0.0211, +0.0125]).
  On Sen1Floods11 test, all TerraMind models are significantly better (+0.0808 to +0.1087).
- **Overfitting (§3.9)**: The WorldFloods test IoU of `unet_bs16` decreases from 0.8351 at epoch 5 to 0.8034 at epoch 29.
  The IoU of `tm_l1c13` stays between 0.8183 and 0.8218 from epoch 10 to epoch 25. It does not decrease late in the training.
  But the checkpoint that the validation set selects (epoch 7) has only 0.8082.
- **SAR (Kuro Siwo, §8)**: On Kuro Siwo test, `tm_s1grd@ep23` minus `unet_s1@ep17` is +0.0224.
  The event bootstrap CI is [−0.0015, +0.0423]. `tm_s1grd` is better on 7 of 10 events.
  On Bolivia, the difference is +0.3550 [+0.2142, +0.4857]. TiM makes the SAR model significantly worse on Kuro Siwo test (−0.0146 [−0.0221, −0.0013]).
- **Sen1Floods11 late fusion (§9)**: BEF (optical and SAR TerraMind with stage-1 training, fine-tuned on Sen1Floods11, DEUF sum)
  is not better than one multi-modal TerraMind: test −0.0222 [−0.0619, +0.0035], Bolivia −0.0519 [−0.0872, −0.0263].
  In all three late-fusion systems, the fused output is worse than the optical branch alone.
  The main cause is the SAR branch: when only the SAR branch is sure, it is correct on only 45.30% of the pixels, and DEUF follows it.

## 1. Experiments

**Names**: `<encoder>[_frozen]_<input>[_<variable>]`.
- `tm` = TerraMind v1 base.
- `l2a6` = the 6 `bgriswirs` bands through the S2L2A embedding.
- `l1c13` = the 13 bands through the S2L1C embedding.
- A name without `_frozen` has a fully trainable encoder.
- A checkpoint is `<name>@ep<epoch>`.

| Name | Configuration | Variable compared with | Status |
|---|---|---|---|
| **`unet_baseline`** | ml4floods UNet (`EDL_ML4FloodsModel`). Encoder and decoder start from WF2. Batch size 64. | Target | Complete |
| **`tm_l2a6`** | Trainable TerraMind. `bgriswirs` 6 bands → S2L2A embedding (band subset). | Ablation: compare with `tm_l1c13` for the effect of 13 L1C bands | Complete |
| **`tm_l1c13`** | Trainable TerraMind. `all` 13 bands → **S2L1C** embedding. | TerraMind baseline | Complete |
| **`tm_l1c13_tim`** | `tm_l1c13` + TiM (generates LULC tokens from S2L1C; argmax decoding; frozen generator) | `tm_l1c13` | Complete |
| **`tm_l1c13_inskip`** | `tm_l1c13` + full-resolution input skip (`decoder_input_skip_channels: 32`) | `tm_l1c13` | Complete |
| `unet_bs64` | `unet_baseline` trained again on our data and machine (same settings, batch size 64) | `unet_baseline` (reproduction) | Complete |
| `unet_bs16` | Same as `unet_bs64`, batch size 16 (same as the TerraMind models) | `unet_bs64` (batch size) | Complete |

**Settings of all TerraMind models**
- Model class: `EDL_TerraMind_ML4FloodsModel`.
- Encoder: TerraMind v1 base (ViT-B/16, HF `ibm-esa-geospatial/TerraMind-1.0-base`).
- Features: layers 2, 5, 8 and 11 → 1×1 lateral convolutions → a pyramid with stride 4, 8, 16 and 32.
- Decoder: the UNet decoder structure + `dconv_full` (stride 4 → full resolution). Random initialization.
- Learning rate: decoder and head 1e-4. Encoder 1e-4 × `backbone_lr_mult` 0.1 = 1e-5.
- Batch size 16 (4,099 steps per epoch), `num_workers` 16, fp32.

**Reason for `tm_l1c13`**: The Sentinel-2 images of WorldFloods v2 are **L1C** (13 bands).
`tm_l2a6` uses the TerraMind S2L2A embedding (surface reflectance) and changes the normalization to align the data.
The TerraMind weights also contain `untok_sen2l1c@224` (13 bands). `tm_l1c13` uses the modality that agrees with the data.
TiM also needs this: the TiM generator accepts only a full pre-trained modality.

**Reason for `tm_l1c13_inskip`**: The finest features of ViT-B/16 have stride 16.
A `double_conv` with stride 1 processes the raw input. The result goes into the decoder before `dconv_full` (the same method as UNETR).
Thus, the last decoder block gets the spectral data of each pixel. The aim is to decrease the false water near water bodies (§3.7).
The skip adds approximately 35,000 parameters.

| Name | Configuration file | Experiment directory (`artifacts/models/`) | Result tag |
|---|---|---|---|
| `unet_baseline` | `configurations/edl.json` (trained on the machine of chlunchen) | `edl_EpochKL_20260819` (symlink) | `EDL` |
| `tm_l2a6` | `configurations/terramind/tm_l2a6.json` | `edl_terramind_v1_base_bgriswirs_nofreeze` | `EDL-TERRAMIND_nofreeze_ep*` |
| `tm_l1c13` | `configurations/terramind/tm_l1c13.json` | `edl_terramind_v1_base_s2l1c_all_nofreeze` | `EDL-TERRAMIND_s2l1c_ep*` |
| `tm_l1c13_tim` | `configurations/terramind/tm_l1c13_tim.json` | `edl_terramind_v1_base_s2l1c_all_nofreeze_tim_lulc` | `EDL-TERRAMIND_tim_ep*` |
| `tm_l1c13_inskip` | `configurations/terramind/tm_l1c13_inskip.json` | `edl_terramind_v1_base_s2l1c_all_nofreeze_inskip32` | `EDL-TERRAMIND_inskip_ep*` |
| `unet_bs64` | `configurations/terramind/unet_bs64.json` | `edl_unet_bgriswirs_bs64` | `EDL_unet_bs64_ep*` |
| `unet_bs16` | `configurations/terramind/unet_bs16.json` | `edl_unet_bgriswirs_bs16` | `EDL_unet_bs16_ep*` |

**Common conditions**
- Data: WorldFloods v2. Train 475, **validation 16**, test 18 Sentinel-2 images. Each epoch has approximately 65,600 windows.
- `max_tile_size` 256. `filter_windows` with `threshold_clouds` 0.8.
- Optimizer: Adam. Scheduler: ReduceLROnPlateau (factor 0.5, patience 2) on `val_bce_land_water`.
- 30 epochs (all runs complete the 30 epochs).
- Same `pos_weight` and `weight_problem`. KL annealing: linear, coefficient 0.5, step 10.

## 2. Evaluation rules and metrics

**Checkpoint rule (set before we looked at the test results)**: For each run, we report two checkpoints.
**The conclusions use only the first checkpoint.**

1. **Best val**: the checkpoint with the lowest `val_bce_land_water`.
2. **Last**: the last epoch (epoch 29). This is supplementary data only.

`unet_baseline` uses `epoch=8-step=9225`, which the lab selected. This checkpoint is also its best val checkpoint.
The script `artifacts/runs/compare/pick_ckpts.py` selects the checkpoints. It reads only `metrics.csv`.

| Name | Best val (for conclusions) | Last (supplementary) |
|---|---|---|
| `unet_baseline` | ep8 | — |
| `tm_l2a6` | ep20 | ep29 (not evaluated) |
| `tm_l1c13` | ep7 | ep29 (1024 tile results only) |
| `tm_l1c13_tim` | ep16 | ep29 |
| `tm_l1c13_inskip` | ep13 | ep29 |
| `unet_bs64` | ep8 | ep29 |
| `unet_bs16` | ep9 | ep29 |

**Inference tile size**: The training uses 256 × 256 tiles. **All main results use 256 × 256 tiles for inference.**
We set this rule before the results. We did not select it from the results.
At first, the wrapper script `artifacts/runs/compare/run_inference_tile.py` set the tile size.
Now `run_inference.py --max_tile_size` sets it. For `EDL-TERRAMIND`, the default is 256. For other model types, the default is 1024.
The 1024 tile results are a sensitivity analysis (§3.4).

**Pipeline**: `artifacts/runs/compare/run_pipeline_tile.sh` runs these steps:
1. `run_inference.py` (full images).
2. `eval_metrics.py`.
3. `analysis_S2.py` (retention curve).
4. `compute_pavpu.py` (PAvPU).

`report.py` and `make_reports.sh` make the reports (`artifacts/runs/compare/report_*_tile256.md`).
`eval_val_iou.py` calculates the pooled IoU on the validation set (§3.6).

**Metrics**

- **Segmentation**: For each image, we round the EDL probability to water or not water. We compare it with the ground truth (GT).
  This gives TP, FP, FN and TN. Invalid pixels are not used. IoU water = TP / (TP + FP + FN).
  - **OVERALL**: we add the pixels of all images, then calculate one value. Large events have a large weight.
  - **AVERAGE**: we calculate one value for each image, then calculate the mean. Each event has the same weight.
  - **Use both values.** Many Sen1Floods11 chips have little water. Their IoU can be near 0. Thus, AVERAGE is less useful on Sen1Floods11.
- **Retention curve**: We sort the pixels by uncertainty, from low to high. We keep the X% most certain pixels and calculate their IoU.
  We report the normalized area under the curve (AUC).
- **PAvPU**: For each 3 × 3 patch, we find if the patch is accurate (patch IoU ≥ threshold).
  We also find if it is uncertain (mean uncertainty ≥ threshold).
  PAvPU = (accurate and certain + inaccurate and uncertain) / all patches.
  - `compute_pavpu.py` takes the two thresholds from the retention 90% point of each model.
  - The IoU of a 3 × 3 patch has only discrete values. The two values nearest to 1 are 1 and 8/9 (0.889).
  - The accuracy thresholds of almost all models are in (0.889, 1]. Thus, the rule is the same for all models: "accurate" means that all pixels of the patch are correct.
  - Thus, **a comparison between models is valid**. We calculated PAvPU again with the threshold of `unet_baseline` (`pavpu_fixed.py`).
    Most differences are less than 0.01. The largest difference is 0.0131 (`tm_l1c13@ep7` on Bolivia).
- **Significance**: We do a paired bootstrap of the test events or chips (10,000 resamples). We report the difference and the 95% CI.
  The CI includes only the variation from the selection of events. It does not include the variation from training again (seed).

**Test on a different dataset: Sen1Floods11 v1.1**

- Source: `/home/NAS/house/ycchen-10014/data/sen1floods11_v1.1/v1.1`, hand-labeled 512 × 512 chips.
- Input: `S2L1CHand` (13 bands; same band order and scale as WorldFloods).
- Labels: `LabelHand` (−1 no data, 0 not water, 1 water). We change them to the WorldFloods GT format.
- `artifacts/runs/compare/prepare_sen1floods11.py` writes the data to `/home/NAS/house/ycchen-10014/data/sen1floods11_wf/<group>/test/{S2,gt}`.
  The pipeline does not need changes.
- Groups:
  - **`test`**: the official test split (90 chips) without the 6 Spain chips. 84 chips from 9 countries.
  - **`bolivia`**: 15 chips. Sen1Floods11 does not use this event for training.
- **Data leak**: The 6 Spain test chips overlap EMSR388 in the WorldFloods **train** split (the same flood, September 2019).
  Thus, we do not use them (they are in `test_spain_leak`). The chips from the other countries do not overlap.

## 3. Results (256 tile inference)

Full reports: `artifacts/runs/compare/report_worldfloods_tile256.md`, `report_sen1floods11_test_tile256.md`, `report_sen1floods11_bolivia_tile256.md`.

### 3.1 WorldFloods v2

**IoU water** (validation 16 events, test 18 events)

| Model | Val OVERALL | Val AVERAGE | Test OVERALL | Test AVERAGE | Test precision | Test recall |
|---|---|---|---|---|---|---|
| **`unet_baseline@ep8`** | 0.8744 | 0.8012 | **0.8285** | **0.8302** | 0.8896 | 0.9235 |
| `tm_l2a6@ep20` | 0.8885 | 0.7826 | 0.8090 | 0.8095 | 0.9013 | 0.8876 |
| `tm_l1c13@ep7` | 0.8934 | 0.7938 | 0.8082 | 0.8126 | 0.8818 | 0.9064 |
| **`tm_l1c13_tim@ep16`** | 0.8932 | 0.7899 | **0.8179** | 0.8132 | 0.8835 | 0.9167 |
| `tm_l1c13_inskip@ep13` | 0.8943 | 0.7946 | 0.8000 | 0.8080 | 0.8978 | 0.8802 |
| `tm_l1c13_inskip@ep29` (last) | 0.8955 | 0.7938 | 0.7987 | 0.8024 | 0.8975 | 0.8789 |

**Test differences (paired bootstrap, 95% CI)**

| Model | vs `unet_baseline`: OVERALL | vs `unet_baseline`: AVERAGE | vs `tm_l1c13`: OVERALL | vs `tm_l1c13`: AVERAGE |
|---|---|---|---|---|
| `tm_l2a6@ep20` | **−0.0196 [−0.0350, −0.0038]** | −0.0207 [−0.0515, +0.0038] | +0.0007 [−0.0102, +0.0191] | −0.0031 [−0.0221, +0.0129] |
| `tm_l1c13@ep7` | **−0.0203 [−0.0423, −0.0061]** | −0.0176 [−0.0568, +0.0103] | — | — |
| `tm_l1c13_tim@ep16` | **−0.0106 [−0.0331, −0.0019]** | −0.0170 [−0.0416, +0.0017] | +0.0097 [−0.0046, +0.0186] | +0.0005 [−0.0200, +0.0200] |
| `tm_l1c13_inskip@ep13` | **−0.0285 [−0.0498, −0.0046]** | **−0.0222 [−0.0555, −0.0001]** | −0.0082 [−0.0245, +0.0109] | −0.0046 [−0.0182, +0.0065] |

### 3.2 Sen1Floods11 (different dataset)

**Test** (84 chips, without Spain). The last-epoch rows are supplementary. We do not give their differences.

| Model | IoU OVERALL | IoU AVERAGE | Precision | Recall | vs `unet_baseline`: OVERALL | vs `tm_l1c13`: OVERALL |
|---|---|---|---|---|---|---|
| `unet_baseline@ep8` | 0.6366 | 0.4131 | 0.6744 | 0.9191 | — | — |
| `tm_l2a6@ep20` | 0.6735 | 0.4218 | 0.7112 | 0.9270 | +0.0368 [−0.0022, +0.0795] | +0.0258 [−0.0056, +0.0622] |
| `tm_l1c13@ep7` | 0.6476 | 0.4265 | 0.6784 | 0.9345 | +0.0110 [−0.0204, +0.0463] | — |
| **`tm_l1c13_tim@ep16`** | **0.6756** | **0.4333** | 0.7116 | 0.9304 | **+0.0390 [+0.0078, +0.0748]** | **+0.0280 [+0.0045, +0.0512]** |
| `tm_l1c13_tim@ep29` (last) | 0.6867 | 0.4317 | 0.7265 | 0.9261 | — | — |
| `tm_l1c13_inskip@ep13` | 0.6708 | 0.4284 | 0.7042 | 0.9340 | +0.0342 [−0.0034, +0.0722] | **+0.0232 [+0.0020, +0.0422]** |
| `tm_l1c13_inskip@ep29` (last) | 0.6637 | 0.4224 | 0.6983 | 0.9305 | — | — |

**Bolivia** (15 chips. Sen1Floods11 does not use this event for training. The sample is small, thus the CIs are wide.)

| Model | IoU OVERALL | IoU AVERAGE | Precision | Recall | vs `unet_baseline`: OVERALL | vs `tm_l1c13`: OVERALL |
|---|---|---|---|---|---|---|
| `unet_baseline@ep8` | 0.5416 | 0.3997 | 0.5500 | 0.9726 | — | — |
| `tm_l2a6@ep20` | 0.6498 | 0.4518 | 0.6680 | 0.9597 | +0.1082 [−0.0100, +0.2318] | +0.1201 [+0.0588, +0.1875] |
| `tm_l1c13@ep7` | 0.5298 | 0.3784 | 0.5350 | 0.9820 | −0.0118 [−0.0780, +0.0577] | — |
| `tm_l1c13_tim@ep16` | 0.5962 | 0.4059 | 0.6052 | 0.9758 | +0.0546 [−0.0317, +0.1412] | +0.0665 [+0.0363, +0.0895] |
| `tm_l1c13_tim@ep29` (last) | 0.6231 | 0.4331 | 0.6359 | 0.9687 | — | — |
| `tm_l1c13_inskip@ep13` | 0.5832 | 0.4096 | 0.5906 | 0.9788 | +0.0416 [−0.0381, +0.1413] | +0.0534 [+0.0218, +0.0878] |
| `tm_l1c13_inskip@ep29` (last) | 0.6012 | 0.4209 | 0.6102 | 0.9761 | — | — |

On Sen1Floods11, the precision of all models is low (test 0.6744 to 0.7265; Bolivia 0.5350 to 0.6680). The recall is high (0.9191 or more).
Thus, all models predict too much water. A possible cause is a difference in the labeling rules of the two datasets
(for example, the Sen1Floods11 hand labels can be more conservative). This difference has the same effect on all models.

### 3.3 Uncertainty (DST; the four uncertainty types give almost the same values)

| Model | WF val: retention AUC / PAvPU | WF test: retention AUC / PAvPU | S1F11 test: retention AUC / PAvPU | Bolivia: retention AUC / PAvPU |
|---|---|---|---|---|
| `unet_baseline@ep8` | 0.9909 / 0.9010 | **0.9688** / **0.8394** | **0.9427** / 0.8295 | 0.8901 / 0.6938 |
| `tm_l2a6@ep20` | 0.9914 / 0.9037 | 0.9498 / 0.7953 | 0.9244 / 0.8313 | **0.9544** / **0.7958** |
| `tm_l1c13@ep7` | 0.9927 / 0.8946 | 0.9648 / 0.8266 | 0.9355 / 0.8355 | 0.9190 / 0.7204 |
| `tm_l1c13_tim@ep16` | 0.9903 / 0.8982 | 0.9685 / 0.8273 | 0.9318 / **0.8442** | 0.8308 / 0.7789 |
| `tm_l1c13_inskip@ep13` | 0.9912 / 0.9007 | 0.9431 / 0.8051 | 0.9311 / 0.8390 | 0.8267 / 0.7713 |

- On WorldFloods test and on Sen1Floods11 test, `unet_baseline` has the highest retention AUC. On WorldFloods test, `tm_l1c13_tim` is almost the same.
  TerraMind segments more correctly on Sen1Floods11. But it does not sort its uncertainty better.
- On Sen1Floods11 test, the PAvPU of the TerraMind models is slightly higher (0.8313 to 0.8442; `unet_baseline` 0.8295).
  Bolivia has only 15 chips. Thus, its values change much.

### 3.4 Sensitivity to the inference tile size (WorldFloods test, 1024 → 256)

| Model | OVERALL 1024 → 256 | AVERAGE 1024 → 256 | Precision 1024 → 256 |
|---|---|---|---|
| `unet_baseline@ep8` | 0.8285 → 0.8285 | 0.8303 → 0.8302 | 0.8894 → 0.8896 |
| `tm_l2a6@ep20` | 0.8086 → 0.8090 | 0.8029 → 0.8095 | 0.8930 → 0.9013 |
| `tm_l1c13@ep7` | 0.8070 → 0.8082 | 0.8063 → 0.8126 | 0.8508 → 0.8818 |

- The tile size has no effect on the UNet (control model, as expected).
- With 256 tiles, the OVERALL IoU of TerraMind changes by 0.0012 or less. The AVERAGE IoU increases by 0.0063 to 0.0066.
  The precision of `tm_l1c13` increases by 0.0310 (less false water).
- The tile size **does not change the conclusions**. On WorldFloods test, TerraMind is still lower than `unet_baseline`.
  The 1024 tile values are in this document at commit `af11788`.

### 3.5 Training curves (`lightning_logs/version_0/metrics.csv`)

Figure: `artifacts/runs/compare/curves_ABDF.png`.
`val_iou_land_water water` is not a correct estimate of IoU, and it changes with the batch size (§3.6). Use it only for trends in one run.

| Model | Best val_bce: epoch / val_bce | Final (ep29) val_bce | Epoch 0 val_bce |
|---|---|---|---|
| `unet_baseline` (batch size 64) | 8 / 0.0621 | 0.0777 | 0.24 |
| `tm_l2a6` | 20 / 0.0536 | 0.0543 | 0.96 |
| `tm_l1c13` | 7 / 0.0514 | 0.0544 | 0.29 |
| `tm_l1c13_tim` | 16 / 0.0535 | 0.0563 | 0.53 |
| `tm_l1c13_inskip` | 13 / 0.0522 | 0.0538 | 0.50 |

For the trainable TerraMind models, val_bce at epoch 0 is high. The encoder has no warmup. From epoch 1, the value is normal.
After epoch 10, all TerraMind models stay between 0.052 and 0.056. This is lower than `unet_baseline`.

### 3.6 Error in the validation IoU during training, and the pooled IoU

`EDL_ML4FloodsModel.validation_step` uses the ml4floods method.
It calculates one IoU for each validation batch with `metrics.calculate_iou`. Lightning then calculates a mean weighted by the batch size.
This method has two problems:

1. `calculate_iou` (`ml4floods/models/utils/metrics.py`) adds 1e-6 to TP. It calculates FP and FN as the sum minus TP.
   Thus, **if a batch has no water**, a fully correct batch gives −1. One false water pixel gives 0.
2. A batch with little water gets a low IoU from a small error. But it has the same weight as a batch with much water.
   Thus, the value **changes with the batch size**.
   The real validation set has 3,524 tiles (1,817 tiles with water).
   On this set, a perfect model gets 0.80 with batch size 16 (22 of 221 batches have no water).
   It gets 0.93 with batch size 64 (2 of 56 batches).

`artifacts/runs/compare/eval_val_iou.py` calculates the values again with the same validation data loader. It does not change shared code.
The `logged_*` values are the same as the values in the training logs.

| Model | Logged, batch 16 | Logged, batch 64 | **Pooled IoU** | Precision | Recall |
|---|---|---|---|---|---|
| `unet_baseline@ep8` | 0.539 | 0.703 | 0.873 | 0.888 | 0.982 |
| `tm_l2a6@ep8` | 0.531 | 0.698 | 0.888 | 0.911 | 0.973 |
| `tm_l2a6@ep20` | — | — | 0.887 | 0.908 | 0.975 |
| `tm_l1c13@ep7` | — | — | 0.892 | 0.914 | 0.974 |

The same `unet_baseline` model gives 0.539 with batch size 16 and 0.703 with batch size 64.
Thus, the training-curve gap "UNet 0.70 and TerraMind 0.51" comes from the batch size.
(An earlier version of this document gave "edge effects of 256 tiles on the ViT" as the cause. **That cause is not correct.**)
`EDL_SAR_Unet.on_validation_epoch_end` has a correct epoch-level `val_Global_iou_*`. The optical EDL base class does not have it.
Before you change the shared `edl.py`, coordinate with the SAR work.

### 3.7 Distance of the errors from the GT water boundary (WorldFloods test, 256 tiles)

We put each valid pixel into a band by its distance to the nearest GT boundary between water and land (1 px = 10 m).
We compare the error rate of each band.
Script: `artifacts/runs/compare/boundary_error_analysis.py`. Output: `boundary_error_test_tile256.txt` (1024 tile version: `boundary_error_test.txt`).

| Distance from the boundary | Pixel fraction | `unet_baseline@ep8` | `tm_l1c13@ep7` | `tm_l2a6@ep20` | Fraction of the extra errors of `tm_l1c13@ep7` |
|---|---|---|---|---|---|
| 1–2 px (at the boundary) | 1.8% | 0.3966 | 0.3984 | 0.4025 | 0.5% |
| 2–4 px | 2.6% | 0.2640 | 0.2762 | 0.2758 | 4.3% |
| 4–8 px | 3.7% | 0.1515 | 0.1704 | 0.1637 | 9.7% |
| 8–16 px (1 patch) | 5.5% | 0.0997 | 0.1204 | 0.1119 | 15.7% |
| 16–32 px (1 to 2 patches) | 8.2% | 0.0719 | 0.0932 | 0.0845 | 23.9% |
| ≥ 32 px (far from the boundary) | 78.2% | 0.0349 | 0.0392 | 0.0388 | 46.0% |

`tm_l1c13@ep7` has 1,218,750 more error pixels than `unet_baseline@ep8`.

- **At the boundary, TerraMind is as good as the UNet** (error rate at 1–2 px: 0.3984 and 0.3966).
  Thus, the hypothesis "the ViT cannot find the detail of the boundary" is not correct.
- 53.6% of the extra errors of `tm_l1c13@ep7` are 2 to 32 px from the water (1 to 2 ViT patches).
  In these bands, the error rate is 4.6% to 29.6% higher. Most extra errors are false water:
  at 8–32 px, FP increases by 327,956 and FN increases by 154,162.
  With 1024 tiles, the false water in these bands is larger. Thus, a part of it comes from the large inference tile.
- 46.0% of the extra errors are far from water (≥ 32 px). With 256 tiles, **most of these are missed water**:
  FN increases by 825,436 and FP decreases by 265,019. Some areas inside large water bodies are not found.
  `tm_l2a6@ep20` has the same pattern (FN +1,205,915).
- `tm_l1c13_inskip` was made to decrease the false water near water bodies. On WorldFloods test, it does not improve the result
  (−0.0082 [−0.0245, +0.0109] vs `tm_l1c13`). Its recall decreases (0.8802 and 0.9064).

### 3.8 UNet retraining and batch size

`unet_bs64` and `unet_bs16` start from the WF2 pre-trained weights (`/home/NAS/homes/cjchen-10025/.../WF2_unetv2_bgriswirs/model.pt`).
They use the same training data and machine as the TerraMind models. All settings except the batch size are the same as `configurations/edl.json`.
The checkpoints follow the rule of §2 (best val: `unet_bs64` ep8, `unet_bs16` ep9).

**IoU water (OVERALL / AVERAGE)**

| Model | WF val | WF test | Sen1Floods11 test | Bolivia |
|---|---|---|---|---|
| `unet_baseline@ep8` | 0.8744 / 0.8012 | 0.8285 / 0.8302 | 0.6366 / 0.4131 | 0.5416 / 0.3997 |
| `unet_bs64@ep8` | 0.8755 / 0.7999 | 0.8283 / 0.8268 | 0.6332 / 0.4132 | 0.5457 / 0.4056 |
| `unet_bs64@ep29` (last) | 0.8475 / 0.7726 | 0.8013 / 0.7846 | 0.5672 / 0.3570 | 0.4821 / 0.3646 |
| `unet_bs16@ep9` | 0.8566 / 0.7672 | 0.8176 / 0.8127 | 0.5669 / 0.3808 | 0.5669 / 0.4081 |
| `unet_bs16@ep29` (last) | 0.8534 / 0.7760 | 0.8034 / 0.7947 | 0.5249 / 0.3638 | 0.4615 / 0.3517 |

- **Reproduction**: `unet_bs64@ep8` minus `unet_baseline@ep8` (ΔIoU OVERALL): WorldFloods test −0.0002 [−0.0034, +0.0011];
  Sen1Floods11 test −0.0034 [−0.0126, +0.0049]. There is no difference. The training data of `unet_baseline` does not change the results (§6, item 1).
- **Batch size**: `unet_bs16@ep9` minus `unet_bs64@ep8`: WorldFloods test −0.0108 [−0.0243, −0.0004]; Sen1Floods11 test −0.0663 [−0.1021, −0.0305].
- **Comparison with TerraMind (ΔIoU OVERALL)**

| Model | vs `unet_bs16@ep9`: WF test | vs `unet_bs16@ep9`: Sen1Floods11 test | vs `unet_bs64@ep8`: WF test | vs `unet_bs64@ep8`: Sen1Floods11 test |
|---|---|---|---|---|
| `tm_l2a6@ep20` | −0.0086 [−0.0259, +0.0138] | +0.1066 [+0.0546, +0.1707] | — | — |
| `tm_l1c13@ep7` | −0.0094 [−0.0300, +0.0084] | +0.0808 [+0.0243, +0.1469] | −0.0201 [−0.0397, −0.0052] | +0.0144 [−0.0198, +0.0536] |
| `tm_l1c13_tim@ep16` | +0.0003 [−0.0211, +0.0125] | +0.1087 [+0.0579, +0.1712] | −0.0105 [−0.0304, −0.0021] | +0.0424 [+0.0095, +0.0817] |
| `tm_l1c13_inskip@ep13` | −0.0176 [−0.0406, +0.0117] | +0.1039 [+0.0498, +0.1668] | — | — |

All TerraMind models use batch size 16. We did not test TerraMind with batch size 64.

### 3.9 Overfitting

**WorldFloods test IoU OVERALL at each epoch** (256 tiles; `artifacts/results/tile256_traj`)

| Model | ep5 | ep7 | ep9 | ep10 | ep15 | ep20 | ep25 | ep29 |
|---|---|---|---|---|---|---|---|---|
| `tm_l1c13` | — | 0.8082 (best val) | — | 0.8198 | 0.8218 | 0.8189 | 0.8183 | — |
| `unet_bs16` | 0.8351 | — | 0.8176 (best val) | 0.8095 | — | — | 0.8060 | 0.8034 |

- The test IoU of `unet_bs16` decreases after epoch 5. This agrees with the increase of val_bce
  (for `unet_baseline`, val_bce increases by 25.1% from the best epoch to epoch 29).
- After epoch 10, the test IoU of `tm_l1c13` stays stable. It does not decrease late in the training.
  The checkpoint that the 16 validation images select (epoch 7) has the lowest measured value.
- **Fit to the training data** (30 random training images, seed 0; IoU OVERALL):
  `unet_baseline@ep8` 0.7390, `unet_bs16@ep9` 0.7174, `tm_l1c13@ep7` 0.7759, `tm_l1c13@ep29` 0.8117, `tm_l1c13_tim@ep16` 0.8109.
  TerraMind fits the training data better. But its WorldFloods test IoU does not increase by the same quantity.
  The 30 sampled images are more difficult than the test images. Thus, use the difference between training and test IoU only to compare models.

### 3.10 Parameters and computation

We measured one forward pass of one tile with `torch.utils.flop_counter` (optical 256 × 256, SAR 224 × 224).

| Model | Total parameters | Trainable | Frozen | Encoder | Pyramid + lateral | Decoder | GFLOPs per tile |
|---|---|---|---|---|---|---|---|
| `unet_*` (optical) | 7.78M | 7.78M | 0 | 4.69M | — | 3.10M | 85.0 |
| `tm_l2a6` | 97.13M | 97.13M | 0 | 86.13M | 7.82M | 3.17M | 65.4 |
| `tm_l1c13` | 98.50M | 98.50M | 0 | 87.51M | 7.82M | 3.17M | 66.1 |
| `tm_l1c13_inskip` | 98.53M | 98.53M | 0 | 87.51M | 7.82M | 3.20M | 70.2 |
| `tm_l1c13_tim` | 304.05M | 95.95M | 208.11M | 84.95M | 7.82M | 3.17M | 214.4 |
| `unet_s1` (SAR) | 7.78M | 7.78M | 0 | 4.68M | — | 3.10M | 64.9 |
| `tm_s1grd` (SAR) | 96.34M | 96.34M | 0 | 85.35M | 7.82M | 3.17M | 49.8 |

- Most TerraMind parameters are in the encoder. The ViT operates only on tokens with stride 16.
  Thus, it uses less computation per tile than the UNet (TiM is an exception).
- **The input embedding of TiM is frozen.** The TiM S2L1C patch embedding (2.56M) and the LULC token embedding (3.36M)
  use the same weights as the frozen generator. Thus, `tm_l1c13_tim` and `tm_l1c13` have two differences:
  TiM, and a frozen patch embedding (TiM) or a trainable patch embedding (`tm_l1c13`). The SAR model `tm_s1grd_tim` has the same condition.

## 4. Conclusions

- **The same dataset and a different dataset give opposite results.**
  On WorldFloods test, all TerraMind models are lower than `unet_baseline` (all OVERALL differences are significant; −0.0285 to −0.0106).
  On Sen1Floods11, most TerraMind models are higher. `tm_l1c13_tim@ep16` gives a significant +0.0390 [+0.0078, +0.0748].
  The most possible cause: the full `unet_baseline` network starts from WF2, which was trained on WorldFloods. This gives an advantage on WorldFloods.
- **`tm_l1c13_tim` (TiM) is the best TerraMind model now.** On WorldFloods test, it is the nearest to `unet_baseline` (−0.0106).
  Compared with `tm_l1c13`, it gives +0.0280 [+0.0045, +0.0512] on Sen1Floods11 test and +0.0665 [+0.0363, +0.0895] on Bolivia.
  The generated land-cover data helps generalization to a different dataset.
- **`tm_l1c13_inskip` (full-resolution input skip)**: On WorldFloods, it does not help (−0.0082 vs `tm_l1c13`).
  On Sen1Floods11 test, it gives +0.0232 [+0.0020, +0.0422]. On Bolivia, it gives +0.0534 [+0.0218, +0.0878].
  It does not decrease the false water near water bodies on WorldFloods. We cannot explain the increase on the different dataset yet.
- **`tm_l2a6` and `tm_l1c13` (13 L1C bands)**: On WorldFloods, they are almost the same (+0.0007).
  On Sen1Floods11 test, `tm_l2a6` is higher by 0.0258 (the CI includes zero). On Bolivia, it is higher by 0.1201.
  **The 13 L1C bands do not give a constant increase.** The 6-band subset can be more stable on a different dataset.
  (A possible cause: it uses fewer bands with a large distribution difference between the datasets. We did not test this.)
- **The validation set cannot select the checkpoint reliably.** The best-val checkpoint of `tm_l2a6` (ep20) has a lower test AVERAGE than ep8.
  The best-val checkpoint of `tm_l1c13` (ep7) has a lower test OVERALL than ep29 (1024 tiles).
  On Sen1Floods11, `tm_l1c13_tim@ep29` is better than ep16. The `val_bce` of the 16 validation images almost does not predict the test result.
- **Uncertainty**: On Sen1Floods11, TerraMind predicts more correctly. But its retention AUC is lower than `unet_baseline`.
  The quality of the EDL uncertainty does not increase with the encoder.
- **The batch size is an important variable for the UNet.** With the same batch size 16, TerraMind and the UNet are not significantly different on WorldFloods test.
  On Sen1Floods11 test, TerraMind is significantly better.
  Thus, a part of the gap on WorldFloods test comes from the more favorable batch size 64 of the UNet. We did not test TerraMind with batch size 64.
- **The effect of TiM changes with the dataset and the modality.** For optical data, TiM helps on Sen1Floods11.
  For SAR data, TiM makes the result significantly worse on Kuro Siwo test. On Sen1Floods11 S1, the direction is not constant (§8).
  The input embedding of the TiM models is frozen (§3.10). This difference can have an effect. We did not separate the two effects.

## 5. Open items

- PAvPU with a common threshold for the WorldFloods 256 tile results (`make_reports.sh` calculates it).
- Train TerraMind with batch size 64 (`tm_l1c13`). Then compare it with the best UNet under the same conditions.
- Control runs with a frozen patch embedding (`tm_l1c13` and `tm_s1grd`). These runs separate the effect of TiM from the effect of the frozen embedding.

## 6. Known limitations

1. **The selection method of the `unet_baseline` checkpoint is not recorded.** Its data directory does not exist now.
   But our retrained `unet_bs64@ep8` (selected with the rule of §2) gives the same result (§3.8).
   Thus, the training data and the checkpoint selection do not change the conclusions.
2. **One seed, one training run for each model, different machines.** `unet_baseline` was trained on the machine of chlunchen.
   The TerraMind models were trained on an RTX 4090. The bootstrap CI does not include the training variation. The real uncertainty is larger.
3. **The validation set has only 16 images.** The checkpoint selection, the LR scheduler (patience 2) and early stopping use it.
4. **Different training budgets.** `unet_baseline` uses batch size 64 (1,025 steps per epoch). TerraMind uses batch size 16 (4,099 steps per epoch).
   The learning rate does not change with the batch size. The UNet is sensitive to the batch size (§3.8). We did not train TerraMind with batch size 64.
5. **Dataset differences.** The labeling rules of Sen1Floods11 and WorldFloods are different (§3.2). The precision of all models is low.
   For the different dataset, compare only the relative differences between models.
6. The training data is a copy made from symlinks (train comes from `cynthia-10040/.../worldfloods_v2_smoke_source`).
   The number of files agrees with the metadata. We did not compare the contents of each file.
7. **No hyperparameter search.** TerraMind uses the optimizer settings of the UNet
   (Adam, no warmup, ReduceLROnPlateau on 16 validation images). We tested only `backbone_lr_mult` 0.1.

## 7. Execution notes

- The TerraMind models were trained on an RTX 4090 (24 GB), one run at a time.
  30 epochs took 5.5 hours (`tm_l2a6`), 7.5 hours (`tm_l1c13`) and 6.3 hours (`tm_l1c13_inskip`). GPU memory was approximately 8 GB.
  `tm_l1c13_tim` took 10.6 hours and approximately 11 GB. The frozen TiM generator is also in each checkpoint. Each checkpoint is approximately 2 GB.
- During full-image inference on WorldFloods, the non-square tiles at the image edges caused an error in the TiM token embedding
  (`Input tokens are not squared`). Now `TerraMindUNet.forward` pads the input to a square when TiM is on, then crops the output.
  The training tiles are square, thus training has no change. The Sen1Floods11 chips are square, thus they had no error.
- In this container, `UV_PROJECT_ENVIRONMENT` points to `/opt/venv` of the image by default. This is not the project environment.
  Use the project `.venv` (Python 3.10).
  `compose.yml` had `shm_sizes` (incorrect) instead of `shm_size`. Thus, `/dev/shm` had only 64 MB. After the correction, the DataLoader can use many workers.
- The output paths of `analysis_S2.py` and `compute_pavpu.py` contain the model type but not the subset. Thus, validation and test overwrite each other.
  We use `analysis_S2/{val,test}/<tag>/` to keep them separate.
  `eval_metrics.py`, `analysis_S2.py` and `compute_pavpu.py` accept only `--subset val|test`. Thus, each Sen1Floods11 group is a separate data root.
- If two machines train on the same NAS, the experiment directories must have different names.
  (Two frozen runs once mixed checkpoints and logs because a different machine started the same configuration.)
- The inference and analysis scripts are in `artifacts/runs/compare/`. They are not in git:
  `run_pipeline_tile.sh`, `run_inference_tile.py`, `report.py`, `make_reports.sh`, `pick_ckpts.py`, `eval_val_iou.py`, `pavpu_fixed.py`,
  `boundary_error_analysis.py`, `prepare_sen1floods11.py`, `summarize.py`, `plot_curves.py`, `bench_train_step.py`,
  and for SAR: `eval_sar_tiles.py`, `eval_sar_sen1.py`, `sar_report.py`.

## 8. SAR (Kuro Siwo)

**Settings**
- Training script: `scripts/train/train_edl_terramind_sar.py`. Configuration files: `configurations/terramind_sar/`.
- Input: post-event VV and VH only (`use_pre_event: false`). Tile 224. Batch size 16. 30 epochs.
- The data pipeline is the same as `configurations/edl_sar.json` (with `clamp_input` 0.15 and the Kuro Siwo z-score).
- Training set: 48,768 grids (`KuroV2_grid_dict_test_0_100.gz`, filtered with `train_acts`). The train, validation and test events do not overlap.
- The TerraMind model changes the input back to linear power, converts it to dB, then applies the TerraMind S1GRD pre-training statistics
  (`S1InputRenormalizer`).
- Checkpoint rule: the same as §2 (best val: `unet_s1` ep17, `tm_s1grd` ep23, `tm_s1grd_tim` ep19).

| Name | Configuration | Experiment directory |
|---|---|---|
| `unet_s1` | `EDL_SAR_Unet` (ml4floods UNet, 2 channels) | `edl_sar_kurosiwo_unet_s1_bs16` |
| `tm_s1grd` | `EDL_TerraMind_SAR`, trainable TerraMind S1GRD | `edl_terramind_sar_kurosiwo_s1grd` |
| `tm_s1grd_tim` | `tm_s1grd` + TiM (generates LULC from S1GRD) | `edl_terramind_sar_kurosiwo_s1grd_tim_lulc` |

**Kuro Siwo test** (15,896 tiles, 10 events; `artifacts/runs/compare/report_sar_kurosiwo_test*.md`)

| Model | Water IoU | Precision | Recall |
|---|---|---|---|
| `unet_s1@ep17` | 0.6734 | 0.7761 | 0.8357 |
| `tm_s1grd@ep23` | **0.6957** | 0.7914 | 0.8519 |
| `tm_s1grd_tim@ep19` | 0.6811 | 0.7893 | 0.8325 |
| `unet_s1@ep29` (last) | 0.6622 | 0.7615 | 0.8355 |
| `tm_s1grd@ep29` (last) | 0.6951 | 0.7950 | 0.8469 |

- `tm_s1grd@ep23` minus `unet_s1@ep17`: ΔIoU +0.0224. Event bootstrap (10 events): [−0.0015, +0.0423]. `tm_s1grd` is better on 7 of 10 events.
- `tm_s1grd_tim@ep19` minus `tm_s1grd@ep23`: −0.0146 [−0.0221, −0.0013]. This is significantly worse.
- From epoch 17 to epoch 29, `unet_s1` decreases by 0.0112. `tm_s1grd` decreases by 0.0006.

**Sen1Floods11 S1 (different dataset; `artifacts/runs/compare/report_sar_sen1floods11_*.md`)**
- The input follows the Kuro Siwo training pipeline: S1Hand dB → linear → NaN set to 0.15 → clamp to 0.15 → Kuro Siwo z-score.
- Inference uses 224 tiles.
- The Sen1Floods11 test chips and the Bolivia chips do not overlap the Kuro Siwo training grids.
- The bootstrap resamples chips (2,000 resamples).

| Model | Test (84 chips): IoU / P / R | Bolivia (15 chips): IoU / P / R |
|---|---|---|
| `unet_s1@ep17` | 0.4738 / 0.7797 / 0.5470 | 0.3395 / 0.8731 / 0.3571 |
| `tm_s1grd@ep23` | 0.5188 / 0.6136 / 0.7704 | **0.6945** / 0.7651 / 0.8827 |
| `tm_s1grd_tim@ep19` | **0.5440** / 0.6685 / 0.7449 | 0.5673 / 0.8097 / 0.6545 |

| Comparison | Test (84 chips) | Bolivia |
|---|---|---|
| `tm_s1grd` minus `unet_s1` | +0.0450 [−0.0869, +0.1997] | +0.3550 [+0.2142, +0.4857] |
| `tm_s1grd_tim` minus `unet_s1` | +0.0701 [−0.0590, +0.2136] | +0.2277 [+0.1024, +0.3108] |
| `tm_s1grd_tim` minus `tm_s1grd` | +0.0252 [−0.0027, +0.0549] | −0.1272 [−0.3360, +0.0239] |

**SAR conclusions**
- On Kuro Siwo test, TerraMind S1GRD is higher than the UNet. With the event bootstrap, the difference is not significant.
  On Bolivia, TerraMind is significantly better.
- On Sen1Floods11, the UNet misses much water (recall 0.5470 on test, 0.3571 on Bolivia).
  A possible cause is the different SAR processing of the two datasets.
  Kuro Siwo uses a Lee filter and terrain correction. Sen1Floods11 uses Earth Engine processing.
- TiM makes the SAR model worse on the same dataset. On Sen1Floods11, the direction is not constant. This is different from the optical results.

**SAR notes**
- `clamp_input` 0.15 (approximately −8.24 dB) clips 29.304% of the valid VV pixels. This has the same effect on `unet_s1` and `tm_s1grd`.
- `edl_sar_kurosiwo_epochKL@ep14` (an existing lab checkpoint) has 6 channels (with pre-event images).
  Thus, it is not comparable with our post-event 2-channel models.
  `configurations/edl_sar_sen1.json` uses it as a 2-channel checkpoint. The new `load_matching_weights` stops with a shape error.
  Its steps per epoch (335 steps × batch size 64) agree with the 21,462 training grids of `KuroV2_grid_dict.gz`.
  The current configuration uses 48,768 grids.
- `tests/test_sar_structure.py` expects 2 channels, but `configurations/edl_sar.json` has 6 channels. Thus, the test fails now (the upstream settings do not agree).
- For SAR, the retention AUC from water IoU is low. The most certain pixels are mostly land, thus the IoU is defined only at a high retention.
  We do not use this value for conclusions.

## 9. Sen1Floods11 late fusion (BEF / DEUF)

**Question**: Is the late fusion of an optical model and a SAR model better than one multi-modal model?
All models are fine-tuned on Sen1Floods11 train. All models use the full TerraMind encoder (no frozen layers), except the UNet reference.

**Settings**
- Training script: `scripts/train/train_edl_terramind_sen1.py`. Configuration files: `configurations/terramind_sen1/`.
  Experiment directories: `artifacts/models/terramind_sen1/`.
- Batch size 4. 30 epochs. Learning rate 1e-4 (encoder × 0.1). Tile 256. `pos_weight` [1, 6]. No augmentation.
  KL annealing step: 10 for optical and multi-modal, 30 for SAR (the same values as the teammate's Sen1Floods11 configurations).
- Labels: water only (LabelHand). Sen1Floods11 data: 252 / 89 / 90 source images (train / val / test), 1,008 / 356 / 360 windows of 256 × 256.
- Stage-1 initialization: the optical model loads `tm_l1c13@ep7` (the 4-channel cloud + water head is reduced to the water channels).
  The SAR model loads `tm_s1grd@ep23`. The input re-normalization buffers come from the Sen1Floods11 configuration, not from the checkpoint.
- Checkpoint rule: lowest val BCE (the same rule as §2).
- Fusion (DEUF, `flood_uncertainty/fusion/evidence_sum.py`): e_f = e_SAR + e_optical; α = e_f + 1; water when α_water / S ≥ 0.5.
  This section uses only this original rule.
- Evaluation: `scripts/analysis/eval_sen1_fusion_compare.py`. All entries use the same windows and the same pixel mask
  (SAR valid, optical valid and GT valid). The paired bootstrap resamples the source images (10,000 resamples).
  Bolivia: the 15 images of `splits/flood_bolivia_data.txt` (`--splits-path`). No model uses them for training or validation.

| Name | Optical branch (best epoch) | SAR branch (best epoch) | Stage-1 data |
|---|---|---|---|
| BEF | `tm_l1c13_wf_s1f11` (ep12) | `tm_s1grd_kuro_s1f11` (ep15) | WorldFloods / Kuro Siwo |
| direct (baseline b) | `tm_l1c13_s1f11` (ep25) | `tm_s1grd_s1f11` (ep16) | none (TerraMind pre-training only) |
| multimodal (baseline a) | `tm_l1c13_s1grd_s1f11` (ep10): one encoder, S2L1C + S1GRD input | — | none |
| UNet DEUF (reference) | teammate's `edl_optical_sen1floods11` (ep9, 6 bands) | teammate's `edl_sar_sen1floods11` (ep17) | WorldFloods UNet |

**Results** (water IoU; the retention AUC is for the fused or multi-modal output)

| System | Test SAR | Test optical | Test fused | Test retention AUC | Bolivia SAR | Bolivia optical | Bolivia fused | Bolivia retention AUC |
|---|---|---|---|---|---|---|---|---|
| BEF | 0.6696 | 0.8171 | 0.7845 | 0.8034 | 0.5905 | 0.7934 | 0.7545 | 0.8574 |
| direct | **0.6814** | 0.8059 | 0.7848 | 0.7173 | **0.6384** | 0.7756 | 0.7745 | 0.7546 |
| multimodal | — | — | 0.8067 | 0.7312 | — | — | **0.8064** | 0.6223 |
| UNet DEUF | 0.6534 | **0.8392** | **0.8127** | **0.9535** | 0.5851 | **0.8173** | 0.7578 | **0.9493** |

Test: 360 windows (90 images). Bolivia: 60 windows (15 images). The UNet DEUF test values agree with the teammate's evaluation
(0.6534 / 0.8392 / 0.8126; the fused value is 0.8127 on GPU).

| Comparison (row minus column) | Test | Bolivia |
|---|---|---|
| BEF fused − multimodal | −0.0222 [−0.0619, +0.0035] | −0.0519 [−0.0872, −0.0263] |
| BEF fused − direct fused | −0.0003 [−0.0079, +0.0075] | −0.0201 [−0.0481, −0.0012] |
| BEF fused − UNet DEUF fused | −0.0282 [−0.0536, −0.0092] | −0.0033 [−0.0287, +0.0335] |
| multimodal − UNet DEUF fused | −0.0059 [−0.0240, +0.0137] | +0.0486 [+0.0155, +0.1033] |
| BEF fused − BEF optical | −0.0326 [−0.0705, −0.0074] | −0.0389 [−0.0699, −0.0089] |
| direct fused − direct optical | −0.0212 [−0.0540, +0.0001] | −0.0011 [−0.0184, +0.0226] |
| UNet DEUF fused − UNet optical | −0.0265 [−0.0515, −0.0072] | −0.0595 [−0.1107, −0.0300] |
| BEF optical − direct optical | +0.0111 [+0.0042, +0.0177] | +0.0178 [−0.0142, +0.0419] |
| BEF SAR − direct SAR | −0.0119 [−0.0210, −0.0041] | −0.0479 [−0.1193, +0.0115] |
| BEF optical − UNet optical | −0.0221 [−0.0357, −0.0104] | −0.0239 [−0.0555, −0.0084] |
| BEF SAR − UNet SAR | +0.0161 [−0.0101, +0.0435] | +0.0054 [−0.0390, +0.0639] |

**Result conclusions**
- BEF is not better than the two baselines. On test, BEF and direct are equal. On Bolivia, BEF is significantly worse than both.
- The fused output is worse than the optical branch alone in all three late-fusion systems (significant for BEF and for the UNet).
- Stage-1 training helps the optical branch (WorldFloods: +0.0111 on test, significant). It makes the SAR branch worse (Kuro Siwo: −0.0119 on test, significant).
  After fusion, the two effects cancel.
- The UNet DEUF reference is the best on test. On Bolivia, the multi-modal TerraMind is the best and significantly better than the UNet DEUF.
- The UNet branches have a much higher retention AUC. Their uncertainty identifies their errors better.

### 9.1 Agreement and confidence of the two branches

Script: `scripts/analysis/sen1_conflict_stats.py` (per pixel, BEF pair, original DEUF sum).
A branch is "sure" when its DST uncertainty u = 2 / S ≤ 0.2 (S ≥ 10). We selected this threshold to describe the results, not to tune a method.
The per-image values are in `conflict_stats_images.csv`.

| Branches | Confidence | Test pixels | Test: SAR / optical / fused correct | Bolivia pixels | Bolivia: SAR / optical / fused correct |
|---|---|---|---|---|---|
| Agree | both sure | 88.48% | 99.54% | 79.20% | 99.64% |
| Agree | not both sure | 6.94% | 83.80% | 12.97% | 86.67% |
| Disagree | only optical sure | 1.39% | 3.25% / 96.75% / 96.75% | 2.61% | 5.33% / 94.67% / 94.67% |
| Disagree | only SAR sure | 1.19% | 45.30% / 54.70% / **45.30%** | 1.88% | 43.92% / 56.08% / **43.92%** |
| Disagree | both sure | 0.69% | 3.84% / 96.16% / 68.39% | 1.07% | 5.67% / 94.33% / 73.21% |
| Disagree | neither sure | 1.31% | 34.40% / 65.60% / 57.64% | 2.26% | 35.06% / 64.94% / 58.32% |

- When only one branch is sure, DEUF follows that branch. This is the expected behaviour of the evidence sum.
- When only the optical branch is sure, it is correct on 96.75% of the pixels. Thus, the fusion is correct there.
- When only the SAR branch is sure, it is correct on only 45.30% of the pixels (test). Thus, the fusion is wrong on more than half of these pixels.
  The other systems show the same problem: fused correct 51.83% (direct) and 38.48% (UNet DEUF).
- When both branches are sure and disagree, the optical branch is correct on 96.16% of the pixels, but the fusion is correct on only 68.39%.
- Conclusion: the fusion rule operates correctly. The SAR uncertainty is not reliable: the SAR branch is often sure when it is wrong.

### 9.2 Conditions of the disagreement

Scripts: `sen1_window_features.py` (cloud and SAR features per window), `sen1_bef_sample_analysis.py` (per image and per condition),
`sen1_fusion_evidence_analysis.py` (per pixel evidence), `sen1_divergence_cases.py` (case figures, `artifacts/results/sen1_divergence_cases/`).
Cloud: `S2CloudlessMask` (uint8 0 to 255, cloud when the probability ≥ 50%).

- **The labels exclude most thick clouds.** Of the LabelHand −1 (no label) pixels, 45.6% are cloud. Of the evaluated pixels, only 10.9% are cloud (test, mean of the windows).
  Thus, the test does not contain the condition where the SAR branch must replace the optical branch.
- **In the cloudy windows, the fusion becomes worse.** Windows with more than 20% cloud in the evaluation mask (48 test windows):
  BEF fused 0.5735, BEF optical 0.6787, multimodal 0.7005.
  The SAR recall also decreases (from 0.874 in cloud-free windows to 0.581). Thus, the cause is not the cloud itself.
  The S1 and S2 acquisitions are 0 to 2 days apart (`Sen1Floods11_Metadata.geojson`), thus the time difference is not the main cause.
- **Disagreement cases** (the 8 test images with the largest disagreement share; they have 43.8% of all disagreement pixels):
  - The SAR branch misses shallow water and flooded fields (`India_747992`, `Pakistan_849790`, `Somalia_685158`).
    The VH median of these pixels is −15.9 to −20.6 dB, which is not dark. A VH Otsu threshold also fails on the same pixels.
  - The SAR branch finds water on dark land (VH < −23 dB; `USA_595451`, `India_900498`). The GT and the MNDWI show land.
    In `USA_595451`, the median u of the wrong SAR pixels is 0.063 (very sure).
  - In `Sri-Lanka_649970`, the VH Otsu mask (78%) and the MNDWI Otsu mask (47%) show water where the GT shows land.
    The GT can be incorrect there. The annotators drew the GT on the S2 image.
  - In `Pakistan_849790`, the optical uncertainty increases under thin cloud (median u 0.326). But the wrong SAR branch is more sure (median u 0.178).
    Thus, the fusion is correct on only 43.8% of the disagreement pixels.
- **Spain leakage check** (6 Spain test images overlap WorldFloods EMSR388): BEF optical minus direct optical is +0.0184 on Spain.
  This is approximately the same as on Somalia (+0.0183) and USA (+0.0182). Thus, we do not see an effect of the leakage.

### 9.3 Limitations of §9

- One seed for each model.
- The UNet reference models are from a teammate. They use 6 optical bands, and their initialization and pipeline are different.
- The MNDWI and the GT are both from S2. Thus, their agreement is partly circular. An independent reference (for example a Copernicus EMS map) is necessary to decide when the SAR branch is correct.
- Late-fusion rules other than the original sum (SAR weight, cloud discount, Dempster combination) are in `scripts/analysis/sen1_fusion_rules.py`.
  This document does not report them.

## Appendix A: frozen encoder (`tm_frozen_l2a6`, 1024 tiles)

`tm_frozen_l2a6`: frozen TerraMind, random decoder, `bgriswirs` 6 bands → S2L2A, batch size 16. Checkpoint ep3 (best val).
Configuration `configurations/terramind/tm_frozen_l2a6.json`. Experiment directory `edl_terramind_v1_base_bgriswirs_freeze`. Result tag `EDL-TERRAMIND_freeze`.

| Model | Val OVERALL | Val AVERAGE | Test OVERALL | Test AVERAGE | Test retention AUC |
|---|---|---|---|---|---|
| `unet_baseline@ep8` | 0.8745 | 0.8016 | 0.8285 | 0.8303 | 0.9688 |
| `tm_frozen_l2a6@ep3` | 0.8474 | 0.7631 | 0.8171 | 0.8211 | 0.9665 |

Test difference from `unet_baseline`: OVERALL −0.0113 [−0.0201, +0.0051], AVERAGE −0.0092 [−0.0246, +0.0083].
Pooled validation IoU 0.864 (full image 0.847).
A frozen TerraMind with a decoder trained from zero gets 0.9690 (validation) and 0.9862 (test) of the `unet_baseline` IoU OVERALL.
The uncertainty quality is almost the same. When the encoder is trainable (`tm_frozen_l2a6` → `tm_l2a6`), the validation result increases. The test result does not.

`tm_frozen_l2a6_wf2dec` (frozen + WF2 decoder) and `tm_l2a6_wf2dec` (trainable + WF2 decoder):
the WF2 decoder loads only partially (12 of 30 tensors). Compared with a random initialization, the results are sometimes better and sometimes worse.
The differences are 0.01 or less. There is no constant improvement.
For the full values, refer to this document at commit `e9cc0ba` and `artifacts/runs/compare/tables_ABCDF.md`.
