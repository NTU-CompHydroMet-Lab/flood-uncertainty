# BEF Fusion Method Integration Plan

> Scope: This plan puts the Bayesian Evidential Framework (BEF) into the flood-uncertainty project.
> BEF fuses the Dirichlet posteriors of two single-modality EDL models into one joint posterior.
> It uses a uniform prior and the conditional independence assumption, with `α_fused = α_s1 + α_s2 − 1`.
>
> **This document does not select a backbone**. The backbone comes in through an interface contract (see §4).
> You can use it with an `nn.Module` backbone that the project already has.
> You can also use it with the foundation model backbone that a different document describes.
> The two tasks can progress independently.

---

## 0. Quick Glossary

| Term | Meaning in plain words | Description |
|---|---|---|
| Bayesian Evidential Framework (BEF) | Bayesian evidence fusion | Fuses the Dirichlet concentrations from two single-modality EDL models with `α_f = α_s1 + α_s2 − 1`, under a uniform prior |
| Evidential Deep Learning (EDL) | Evidential deep learning | Models the class posterior with a Dirichlet distribution. It gives α as output, not softmax probabilities |
| Vacuity | Vacuity (missing evidence) | `K / S`, where K is the number of classes and S is Σα. It is the total uncertainty in the Dempster-Shafer sense |
| Aleatoric / Epistemic | Data uncertainty / knowledge uncertainty | Separated with the law of total variance (LoTV) |
| KL Annealing | KL annealing | Keeps the KL regularization weight low at the start of training. This prevents an early collapse of α to 1 |
| Sentinel-1 (S1) / Sentinel-2 (S2) | Keep the original names | The SAR remote sensing satellite and the optical remote sensing satellite |

---

## 1. Decision Summary

| Decision point | Selection | Reason |
|---|---|---|
| Fusion level | Late fusion (evidence-sum) | The core BEF formula fuses at the posterior level. Other variants are extensions |
| Data modality | S1 + S2 paired input | The value of BEF comes from cross-modality Bayesian fusion |
| Task structure | Write `pl.LightningModule` directly | Follows the convention of the current `EDL_ML4FloodsModel` |
| Backbone supply | **The caller injects an `nn.Module`**. This document does not specify it | Removes the dependency on the foundation model selection. You can use the current UNet or an FM backbone that comes later |
| Configuration file | Add `configurations/bef.json`. Keep the four sections `shared / train / infer / validate_only` | Lets you use `utils/config_loader.load_mode_config` again |
| Label compatibility | The cloud head continues to use sigmoid. The water head uses EDL Dirichlet | The project mask has a `(B, 2, H, W)` multi-task structure. BEF has a meaning only for water |

> **Key principle**: This document describes only the math layer and the training layer. **Representation layer topics** are not in the scope of this document. These include the backbone source, the pretrained weights and the multi-modality tokenizer.

---

## 2. Directories and File Locations

```
flood_uncertainty/
├── models/
│   └── bef.py                    # BEF_ML4FloodsModel（雙塔 + evidence-sum）
├── losses/
│   ├── edl_dirichlet.py          # alpha_from_logits / dirichlet_stats / kl / edl_seg_loss / annealing
│   └── edl_loss.py               # 既有 sigmoid-binary EDL，不動
├── fusion/                       # 融合規則
│   ├── __init__.py
│   ├── evidence_sum.py           # α_f = α_s1 + α_s2 − 1（BEF 本體）
│   ├── prob_avg.py               # 對照組
│   └── logit_avg.py              # 對照組
└── data/
    └── paired_s1s2_dataset.py    # S1/S2 tile 對齊之 dataset 包裝

scripts/
├── train/train_bef.py            # 複製 train_edl.py 為樣板
├── inference/run_inference.py    # 修改：--model_type 新增 "BEF"
├── eval/eval_metrics.py          # 修改：--model_type 新增 "BEF"
└── smoke/smoke_train_bef.sh      # 新增

configurations/
└── bef.json                      # 新增：與 edl.json 同 schema

tests/
├── test_bef_fusion.py            # 純 CPU 單元測試
├── test_paired_dataset.py        # S1/S2 對齊測試
└── test_bef_smoke.py             # mock backbone 的整合 smoke
```

**Not in this document**: `models/backbones/*`. The caller supplies the backbone. See §4.3.

---

## 3. Math Layer (`losses/edl_dirichlet.py` + `fusion/evidence_sum.py`)

### 3.1 Core Formula

Under a shared uniform prior `Dir(1, ..., 1)` and the conditional independence assumption:

```
α_fused_k = e_s1_k + e_s2_k + 1     （以 evidence 表示）
          = α_s1_k + α_s2_k − 1     （以 concentration 表示）
```

`k = 1..K` is the class index. `e_i = α_i − 1`.

### 3.2 Public Interface of `losses/edl_dirichlet.py`

```
alpha_from_logits(logits, activation="softplus", clamp_min=1e-6) -> α (B,K,H,W)
dirichlet_stats(α) -> {S: (B,H,W), P: (B,K,H,W), vacuity: (B,H,W)}
total_variance(α) -> (aleatoric: (B,H,W), epistemic: (B,H,W))
kl_dirichlet(α_hat) -> (B, H, W)      # 對 Dir(1,...,1) 的封閉解
annealing_lambda(epoch, annealing_step, mode="linear",
                 fixed_coefficient=1.0) -> float
edl_seg_loss(logits, target, epoch, num_classes,
             variant="digamma",       # "digamma" | "log" | "mse"
             annealing_step=10,
             annealing_mode="linear",
             annealing_coefficient=1.0,
             ignore_index=-1) -> Tensor
```

### 3.3 Three Data Term Losses

- `digamma` (default): `L_data = Σ_k y_k · (ψ(S) − ψ(α_k))`
- `log`: `L_data = Σ_k y_k · (log S − log α_k)`
- `mse` (Sensoy Eq. 5): `L_data = ||y − P||² + Σ_k α_k(S − α_k) / (S²(S+1))`

KL regularization term: `α_hat = (α − 1)·(1 − y_onehot) + 1`, `L_kl = KL(Dir(α_hat) || Dir(1,...,1))`

Total loss (per pixel): `L_pix = L_data + λ(epoch) · L_kl`

### 3.4 KL Annealing: Two Knob Sets in One

| Mode | Formula | Use |
|---|---|---|
| `linear` | `λ = min(1, (epoch + 1) / annealing_step)` | Keeps the KL weight low at the start of training |
| `fixed` | `λ = annealing_coefficient` | Ablation experiments, fixed KL strength |

`annealing_mode` selects the mode. This design keeps the same knob names as the current `losses/edl_loss.py` (commit `e848e5e`).

### 3.5 `fusion/evidence_sum.py`

```
fuse_evidence_sum(alpha_s1, alpha_s2) -> alpha_fused
```

This is one function: `α_s1 + α_s2 − 1`. **Only the unit tests check** the precondition (`α ≥ 1`). There is no runtime check. This follows the flood-uncertainty convention: "do not add defensive checks at internal function boundaries".

---

## 4. Model Contract: `models/bef.py`

### 4.1 Class Signature

```
class BEF_ML4FloodsModel(pl.LightningModule):
    def __init__(self,
                 model_params,               # 沿用現有 config schema
                 backbone_s1: nn.Module,     # 由呼叫端注入
                 backbone_s2: nn.Module,     # 由呼叫端注入
                 ...): ...
```

**The two backbones come in as constructor parameters**. This module does not know or care which type of backbone they are. This is the key design that removes the dependency between BEF and the backbone selection.

### 4.2 I/O Contract

```
batch input:
    image_s1: (B, 2, H, W)          # VV, VH
    image_s2: (B, C_s2, H, W)       # 依 channel_configuration
    mask:     (B, 2, H, W)          # (cloud task, water task)，{0=invalid, 1=neg, 2=pos}

backbone 介面契約（呼叫端須遵守）:
    backbone_s{i}(image_dict: dict[str, Tensor]) -> list[Tensor]   # pyramid features

forward(batch) -> {
    "alpha_water_s1"   : (B, 2, H, W),
    "alpha_water_s2"   : (B, 2, H, W),
    "alpha_water_fused": (B, 2, H, W),   # fuse_evidence_sum(...)
    "logits_cloud"     : (B, 2, H, W),
}
```

### 4.3 Backbone Interface Contract (the Only Interface Between This Document and External Work)

The `backbone_s{i}` that the caller injects must agree with these rules:

- **Input**: `dict[str, Tensor]`. The key is the modality name (`"S1GRD"` or `"S2L2A"`). The value shape is `(B, C, H, W)`.
- **Output**: `list[Tensor]`, with 4 levels of pyramid features. The backbone sets the number of channels and the spatial resolution.
- **Stateless**: The caller does the `.eval()` / `.train()` switch and loads the weights.

After the backbone, the BEF model has: `decoder_s{i}: UNetDecoder(...)` → `head_water_s{i}: Conv1x1 → K=2`. For the UNet decoder, you can use the decoder of the current `ml4floods` backbone in the project. You can also use a simple 4-level upsampling conv. This document does not set a limit.

### 4.4 Training Loss

```
L =   w_s1    · edl_seg_loss(logits_water_s1,    y_water, epoch, ...)
    + w_s2    · edl_seg_loss(logits_water_s2,    y_water, epoch, ...)
    + w_fused · edl_seg_loss(logits_water_fused, y_water, epoch, ...)
    + w_cloud · BCE_masked(logits_cloud, y_cloud)
```

The three water losses run in parallel as "multi-branch training". Thus each single-modality tower learns a stable α before fusion. The `branch_weights` field in the configuration file gives the weights `w_*`.

### 4.5 Label Alignment

The project labels use `{0, 1, 2}`, and `0` means invalid. The conversion occurs at the entry of `edl_seg_loss`: `invalid = (label == 0)`. The water task one-hot uses `label − 1` to get K=2. **Do not change the upstream data layer**.

---

## 5. Configuration File: `configurations/bef.json`

Keep the four sections `shared / train / infer / validate_only`. Add these fields to `shared.model_params.hyperparameters`:

| Field | Type | Use |
|---|---|---|
| `bef_fusion_rule` | `"evidence_sum" \| "prob_avg" \| "logit_avg"` | Selects the fusion rule |
| `evidence_activation` | `"softplus" \| "relu" \| "exp"` | Logits → evidence conversion |
| `edl_variant` | `"digamma" \| "log" \| "mse"` | Data term loss formula |
| `annealing_mode` | `"linear" \| "fixed"` | KL annealing mode |
| `annealing_step` | `int` | T for linear, default 10 |
| `annealing_coefficient` | `float` | λ for fixed, default 1.0 |
| `branch_weights` | `{s1, s2, fused, cloud: float}` | Loss weight of each branch |
| `s1_ckpt_path` / `s2_ckpt_path` | `str \| null` | For offline fusion of two single-modality ckpts (infer only) |

This file **does not contain** backbone-specific fields. The backbone source document defines the backbone hyperparameters.

Add these fields to `shared.data_params`:

- `s1_input_folder` (next to the current `input_folder: "S2"`)
- `s1_channel_configuration` (for example `"vv_vh"`)
- `pair_alignment_strategy` (for example `"filename_match"`)

---

## 6. Data Pipeline: `data/paired_s1s2_dataset.py`

### 6.1 Requirements

BEF needs paired S1 + S2 tiles. The current `WorldFloodsDataset` accepts only one `input_folder`.

### 6.2 Two-Stage Solution

- **Stage A**: Wrap two `WorldFloodsDataset` instances. Align them by file name (the sen1floods11 names have pair relations).
- **Stage B**: Extend `filter_windows` so that it also applies to the S1 image tiles. This prevents empty windows from missing tiles.

### 6.3 Alignment Checks (`tests/test_paired_dataset.py`)

- The S1 and S2 images have the same geographic extent, shape and dtype.
- Read `mask` only one time (do not get it from each of the two sides).
- After `filter_windows`, S1, S2 and mask have the same number of records.

---

## 7. Script Connection Points

### 7.1 `scripts/train/train_bef.py`

Copy `train_edl.py` as the template. Change only these items:

- Use `PairedS1S2Dataset` for the dataset call.
- Change the model to `BEF_ML4FloodsModel(model_params, backbone_s1=..., backbone_s2=...)`.
- **Assemble the backbone source in the script**. You can get it from the current `get_model()` in the project. You can also call the backbone factory that a different document describes. This document does not specify it.

Keep these CLI options: `--config / --mode / --resume_ckpt / --data_root`.

### 7.2 `scripts/inference/run_inference.py`

```
--model_type {EDL, v2, EDL-SAR, BEF}
```

Add a BEF branch in `load_model`. Add `bef_predict(batch)` in `load_inference_function`. It returns:

```
{
    "prob"      : (B, 1, H, W),   # water positive posterior mean
    "evidence"  : (B, 2, H, W),
    "vacuity"   : (B, 1, H, W),
    "aleatoric" : (B, 1, H, W),
    "epistemic" : (B, 1, H, W),
    "cloud_prob": (B, 1, H, W),
}
```

The field names are the same as the current EDL output. Thus `metrics/segmentation.py` and the `analysis/` scripts need zero changes.

### 7.3 `scripts/eval/eval_metrics.py`

Add `BEF` to `--model_type`. It takes the three maps vacuity / aleatoric / epistemic. It makes the retention curve, PAvPU and the epistemic FP/FN comparison plot.

---

## 8. Milestones by Stage

| Milestone | Deliverables | Verification method |
|---|---|---|
| **M1** | `losses/edl_dirichlet.py` + `fusion/evidence_sum.py` + unit tests | `pytest tests/test_bef_fusion.py`: use fake α to verify the formula, KL and annealing |
| **M2** | `models/bef.py` + mock backbone smoke | `pytest tests/test_bef_smoke.py`: one forward pass on fake data + `loss.backward()`. Assert that the two backbones are different objects |
| **M3** | `data/paired_s1s2_dataset.py` + `configurations/bef.json` | Print the batch shape manually. Make sure that the file names align |
| **M4** | `scripts/train/train_bef.py` + `smoke_train_bef.sh` | Run a 1-epoch smoke with **any available backbone** |
| **M5** | BEF branch in `run_inference.py` + `eval_metrics.py` | Full-flow smoke: train → infer → eval |
| **M6** | Retention curve comparison with the baselines (EDL, v2) | Write the output to `artifacts/results/analysis_S2/` |

M1–M2 are pure CPU and have no external dependencies. They are the safe foundation. From M3, the work includes the data layer and real training.

**The milestones do not depend on the backbone selection**. You can complete M1–M3 with a mock backbone (for example, three Conv layers). From M4, you can use the current project backbone or a foundation model backbone that comes later.

---

## 9. Risk Register

| # | Risk | Countermeasure |
|---|---|---|
| 1 | The `(B, 2, H, W)` mask meaning (invalid=0) is different from `ignore_index=-1` in the EDL paper | Do the mapping at the entry of `edl_seg_loss`. The M1 unit tests cover it |
| 2 | The two backbones share weights by accident (this breaks the BEF conditional independence assumption) | `test_bef_smoke.py` asserts `backbone_s1 is not backbone_s2` and `id(w_s1) != id(w_s2)` |
| 3 | The cloud head and the water EDL head share the last layer, and this mixes the gradients | The cloud head splits off before the shared decoder. It does not connect to the final conv of the water head |
| 4 | Incorrect use of the default values of the two KL annealing modes | The M1 tests cover both the linear path and the fixed path |
| 5 | In sen1floods11, some tiles do not have a matching S1 or S2 tile | `paired_s1s2_dataset.py` uses a strict file name intersection. The tests report the missing ratio |

**Not in the risk scope of this document**: backbone weight download failure, GPU memory overflow and foundation model version compatibility. The backbone source document is responsible for these.

---

## 10. Output List

- New: `losses/edl_dirichlet.py`, `fusion/`, `models/bef.py`, `data/paired_s1s2_dataset.py`
- New: `configurations/bef.json`
- Changed: `scripts/inference/run_inference.py`, `scripts/eval/eval_metrics.py`
- New: `scripts/train/train_bef.py`, `scripts/smoke/smoke_train_bef.sh`
- New: `tests/test_bef_fusion.py`, `tests/test_paired_dataset.py`, `tests/test_bef_smoke.py`
- Changed: `metrics/segmentation.py` (add vacuity / aleatoric / epistemic export fields)
- Updated: `README.md` (BEF training/inference/evaluation sections)

**Not in this list**: backbone modules, backbone dependencies (`pyproject.toml`) and backbone configuration fields.

---

## 11. Work Estimate

| Stage | Estimate | Description |
|---|---|---|
| M1 (pure math) | 1 day | Mostly file-level moves. Only the numerical values need verification |
| M2 (model structure + smoke) | 0.5 day | With the M1 foundation, this is mainly connection work |
| M3 (S1/S2 aligned dataset) | 0.5–1 day | You must learn the sen1floods11 naming |
| M4 (training script + smoke) | 0.5 day | Mainly a copy of `train_edl.py` |
| M5 (inference + evaluation branches) | 0.5 day | Uses the current format |
| M6 (analysis, baseline comparison) | Estimated separately | Iterate as necessary |

The total is approximately 3–3.5 days + M6 iterations.

---

## 12. Interface With the Backbone Source

The only interface between this document and external work is the **§4.3 backbone interface contract**.

The BEF model operates if the injected `backbone_s{i}` meets this rule: "input `dict[str, Tensor]`, output pyramid feature `list[Tensor]`". Thus you can complete the BEF implementation first, independently of a specific backbone selection. This includes the current project UNet, a foundation model backbone that comes later, or any `nn.Module` that replaces them in the future.

You can complete M1–M3 with a minimal three-layer Conv mock backbone. You need a real backbone only from M4. At that time, "inject the backbone that you want as a constructor parameter" in `train_bef.py`. All other items stay the same.
