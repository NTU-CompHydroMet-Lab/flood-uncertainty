# Sen1 paired evidence evaluation

The supplied hand-labeled test split has 90 paired 512x512 source images.
Each yields four 256x256 windows: 360 evaluated tiles. These are crops,
not 360 independent scenes/events. Metrics pool all valid pixels, rather
than averaging tile scores. Statistical resampling should group crops by
source image, and preferably by event if event metadata are available.

## Implemented workflow

`flood_uncertainty/fusion/evidence_sum.py` implements cumulative raw evidence
fusion. This formula previously appeared only in `BEF_integration_plan.md`.
`evaluate_sen1_paired.py` now validates shared split IDs, windows, image
shape/CRS/transform, and GT labels. Models use their existing preprocessing,
network inference and best-loss Sen1 checkpoints without retraining.

```sh
python scripts/analysis/evaluate_sen1_paired.py
# Optional full raw products, or validation statistics:
python scripts/analysis/evaluate_sen1_paired.py --save-products
python scripts/analysis/evaluate_sen1_paired.py --split val --output artifacts/results/sen1_paired_val
```

Use `--sar-weights` / `--optical-weights` to override checkpoints;
`--preview-index` changes the displayed paired window. Defaults read the
checkpoint paths from the existing individual `metrics_best_loss.json`.
The preview defaults to tile 16: Ghana_319168, row=0, col=0.

## Formula and interpretation

Class order is [non-water, water]. For every aligned pixel:

```
e_f[k]     = e_sar[k] + e_optical[k]
alpha_f[k] = e_f[k] + 1 = alpha_sar[k] + alpha_optical[k] - 1
S          = sum_k alpha_f[k]
p_water    = alpha_f[water] / S
DST_u      = 2 / S
e_total    = sum_k e_f[k] = S - 2
```

The common Dirichlet(1,1) prior is included once. Raw evidence is fused
before any display normalization. The +1 produces concentration alpha,
not raw evidence. All three previews contain prediction, probability,
DST uncertainty, each class's raw evidence and total evidence. Class
color scales are shared across modalities and fusion; display uses a
log scale with values below 0.01 in its lowest color. White is the common
invalid mask. NPZ preview products preserve the raw values and alpha.

A two-class Dirichlet is a Beta distribution. The code also follows the
project's existing binary variance decomposition:

```
epistemic = p_water * (1 - p_water) / (S + 1)
aleatoric = p_water * (1 - p_water) - epistemic
```

These are Var(p_water) and E[p_water(1-p_water)] respectively. They are
Dirichlet-based quantities, not variation across independently sampled
network weights. The water target includes water generally; these models
do not separately distinguish permanent water from newly flooded water.

Neural evidence is learned pseudo-counts. Cumulative addition is an
explicit fusion assumption, not proof that SAR/optical likelihoods are
independent or calibrated. Different evidence scales and shared errors
can affect fusion. Vacuity 2/S decreases when evidence is added, even
when modalities conflict: [100,0] + [0,100] gives p=0.5 and u=2/202.
Thus lower DST_u alone does not demonstrate better predictions.

Sources for Dirichlet EDL definitions:
https://proceedings.neurips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty.pdf

## Full paired test result

The shared mask is the intersection of valid SAR, optical and label pixels:
20,420,561 pixels across all 90 source images / 360 windows. Individual
previous evaluations had their own masks, so those numbers differ slightly.

| Model | Water IoU | mIoU | F1 | Brier |
|---|---:|---:|---:|---:|
| SAR | 0.653405 | 0.796258 | 0.790375 | 0.041290 |
| Optical | 0.839163 | 0.906906 | 0.912549 | 0.016879 |
| Evidence sum | 0.812635 | 0.891689 | 0.896634 | 0.020011 |

Fusion improves on SAR but is below optical on Water IoU, F1 and Brier in
this run. These are point estimates for fixed checkpoints, not significance
tests, confidence intervals or evidence of superiority across seeds.

## Retention and PAvPU

The run exports nine CSVs: aleatoric, epistemic and DST rankings for each
of SAR, optical and fusion. `retention_iou.png` shows their Water IoU versus
the actual fraction of valid pixels retained. Pixels with lower uncertainty
are retained first. All pixels tied at the selected threshold are retained
together, so actual retention can exceed requested retention. Full-retention
IoU matches the corresponding aggregate metric. No test threshold tuning is
performed. Undefined Water IoU is stored as empty, not zero.

Pretraining/fine-tuning does not prevent these analyses. They measure how
well the final checkpoint ranks errors on Sen1 test data. They do not
isolate the effect of pretraining; that would need a controlled comparison.

`patch_statistics.csv` exports non-overlapping 16x16 patch pixel accuracy,
Water IoU and mean aleatoric/epistemic/DST uncertainty for all three modes.
Patches with some invalid pixels use valid pixels only, with valid count
and fraction recorded; empty patches are omitted. All-land patches have
well-defined accuracy but undefined Water IoU (empty field). There is no
final PAvPU score yet because its thresholds/protocol have not been set.

The existing `compute_pavpu.py` uses patch Water IoU, excludes all-TN and
any partially invalid patches, and derives thresholds from retention CSVs.
Its CLI expects WorldFloods prediction/confusion paths, so it cannot consume
Sen1 outputs unchanged. Its counting helper can be reused under that same
IoU-based protocol. The original paper uses pixel accuracy within a patch
and proposes validation-based uncertainty thresholds:
https://arxiv.org/html/1811.12709

For a pixel-accuracy protocol choose patch size, minimum valid fraction,
accuracy threshold and uncertainty measure first; estimate the uncertainty
threshold from validation, freeze it, then compute on test:

```
PAvPU = (n_accurate_certain + n_inaccurate_uncertain) / n_patches
```

The exported test patch statistics already contain the necessary quantities.
Validation statistics can be produced with the same script; an IoU-based
variant should be named explicitly and keep its all-land policy consistent.

The three retention curves are genuinely close in this run, rather than
missing series. Across the 50 requested retention points, maximum absolute
Water IoU differences from DST ranking are:

| Model | Aleatoric vs DST | Epistemic vs DST |
|---|---:|---:|
| SAR | 0.008384 | 0.002326 |
| Optical | 0.000013 | 0.000017 |
| Fused | 0.015124 | 0.012790 |

All three uncertainty quantities derive from the same binary Dirichlet
parameters; they are not independent estimators. Similar rankings therefore
need not imply lack of pretraining benefit. Measuring that benefit requires
a separate pretrained-versus-from-scratch experimental design.

## Input / prediction / GT context figures

```sh
python scripts/analysis/plot_sen1_paired_case.py --tile-index 16
```

This reuses saved NPZ products without rerunning inference and checks them
against the raw label crop. It writes two figures: fused total evidence / fused DST
uncertainty / fused prediction / GT, and SAR CF / optical CF / fused CF
above raw VV / raw VH / optical RGB inputs. CF is spatial confusion, with
TN/FP/FN/TP colors matching existing visualization conventions. Per-tile
IoU/F1 beneath the CF maps are not the aggregate test scores. Input images
retain pixels outside the GT-valid region; gray output pixels are excluded.
SAR is displayed in original dB with limits [-30, 0], optical RGB in B4/B3/B2
order clipped at 3500. Display transforms never change model outputs.

## Two-row fused comparison without inputs

```sh
python scripts/analysis/plot_sen1_fused_examples.py --tile-indices 44 108
```

`fused_two_examples.png` has two different Sen1 sources (India_900498 and
Mekong_1443339) and four columns: raw total fused evidence, DST uncertainty,
fused prediction and GT. The evidence scale is logarithmic and shared.
Total evidence is S-2, so it is inversely related to DST uncertainty; it
measures strength without indicating which class is supported. Selection
used GT water coverage and label completeness, not prediction performance.
Missing raw products are generated for selected tiles with the same models
and preprocessing; existing products are reused after checking source GT.

`retention_tail_audit.json` checks the 90%-to-100% DST retention interval.
Fused Water IoU is 0.938501 at 90% retained pixels and 0.812635 at 100%.
The last 10% added pixels have accuracy 0.797557. A drop near 100% is
consistent with low-to-high uncertainty inclusion, not by itself a plotting
error: 100% must recover the full aggregate IoU. IoU need not be monotonic,
and changes in class composition also affect it. A random-ranking baseline
would help quantify ranking benefit beyond this consistency check.

## DEUF: one command for every test tile

From the project directory, in a Python environment with the project's
PyTorch and plotting dependencies:

```sh
python scripts/analysis/infer_sen1_deuf.py
```

The default writes to `artifacts/results/sen1_paired_test/DEUF/`. It processes
all 360 test windows (90 original 512x512 paired samples, four 256x256 crops
per sample). Each tile produces exactly five figures, for 1800 PNGs total:

1. `sar_evidence`: prediction, water probability, DST uncertainty, class
   evidence and total evidence (2x3).
2. `optical_evidence`: same layout for optical.
3. `fused_evidence`: same layout after cumulative evidence fusion.
4. `fused_context`: total fused evidence, DST uncertainty, prediction, GT.
5. `confusion_inputs`: SAR/optical/fused spatial CF above VV/VH/optical RGB.

Each tile has its own folder, for example:

```text
DEUF/Ghana_319168__tile_000016__r0000_c0000/
```

The five filenames inside that folder are:

```
Ghana_319168__tile_000016__r0000_c0000__sar_evidence.png
Ghana_319168__tile_000016__r0000_c0000__optical_evidence.png
Ghana_319168__tile_000016__r0000_c0000__fused_evidence.png
Ghana_319168__tile_000016__r0000_c0000__fused_context.png
Ghana_319168__tile_000016__r0000_c0000__confusion_inputs.png
```

Sample ID, global test tile index and native crop row/column distinguish all
outputs and tile folders. The root DEUF folder holds the range manifest and
run record; PNGs are stored only inside tile folders. Images and GT use verified paired grids. The common valid mask is
used for all result maps. All-land and all-invalid crops are still exported;
undefined Water IoU/F1 is labeled accordingly. Gray marks excluded pixels.
Evidence uses shared scales within each tile's figures, not across the full
test set. All tiles have fixed [0,1] probability/uncertainty scales.

Models are loaded once and inference is batched. This command does not
recompute retention or PAvPU. By default checkpoints are read from paired
`metrics.json`, falling back to individual `metrics_best_loss.json`; it never
silently chooses the initial pretraining checkpoint. Overrides are supported:

```sh
python scripts/analysis/infer_sen1_deuf.py \
  --sar-weights /path/to/sar_sen1.ckpt \
  --optical-weights /path/to/optical_sen1.ckpt \
  --batch-size 4 --dpi 160
```

An optional short run or restart range:

```sh
python scripts/analysis/infer_sen1_deuf.py --start-tile 16 --max-tiles 2
```

No output cap or GT water filter applies unless `--max-tiles` is explicitly
provided. A range manifest such as `manifest_000000_000359.csv` maps every
tile to its five files and valid-pixel count. A `.partial` manifest is retained
if interrupted; the final CSV appears only after that range is complete.
`run_000000_000359.json` records checkpoint/config paths, requested range,
mask/fusion settings and completion status. Progress is printed after each
completed tile. A rerun of the same range overwrites the same named PNGs.
Use `--save-products` if raw NPZ evidence/alpha/uncertainty arrays are also
needed; they are written under `DEUF/<tile folder>/products/{sar,optical,fused}/`.

The command was checked using synthetic paired data, including all-land and
all-invalid cases. The complete real test export was intentionally left for
manual execution at the user's request.

## GT water coverage index

`DEUF/water_gt20_tiles.csv` records tiles with strictly more than 20% GT
water among their common valid SAR/optical/GT pixels. It retains sample ID,
tile index, crop coordinates, valid count, water fraction and the five PNG
paths from the manifest, and adds `gt_water_pixels`. Exactly 20% and
zero-valid-pixel tiles are excluded. This index does not filter inference or
figure export: all requested test tiles still produce five figures.

The unified command updates this CSV after completing its requested range.
When `--start-tile` / `--max-tiles` selects a subset, the CSV describes that
range. The current full-test CSV was generated from the existing 360-tile
manifest without rerunning inference or plots.
