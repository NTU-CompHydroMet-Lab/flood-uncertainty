# Foundation Model Backbone Integration (First Case: TerraMind v1)

> Scope: This document adds a remote sensing foundation model (Foundation Model, FM) to flood-uncertainty as a backbone.
> Any downstream decoder + head can use this backbone. The first case uses IBM/ESA **TerraMind v1**.
> We get the pretrained weights through `terratorch`.
>
> **This document does not include uncertainty methods**. The backbone gives pyramid features through a general interface.
> These features can go to the current EDL, to the baseline segmentation, or to the fusion method in
> [`BEF_integration_plan.md`](./BEF_integration_plan.md). You can do the two tasks independently.

---

## 0. TL;DR

**Use only the backbone factory. Do not use the full framework.** We add `terratorch` as a uv dependency.
We call only one function from it: the backbone factory. We do not inherit Task, and we do not use the registry or the YAML CLI.
We keep the project conventions: `pl.LightningModule` + JSON config + script.

TerraMind adds only "**one backbone, one script, one config**". All other parts stay the same.

```
原本：  [ml4floods backbone] ── decoder ── head
換成：  [TerraMind backbone] ── decoder ── head
                 ▲
                 └── 只有這顆換了；decoder、head、資料、Trainer、config schema 都不動
```

The work takes approximately 2.5–3 days + the time for full training.

**This phase does not add an abstract interface layer**. The config sets model_type.
The model class builds TerraMindBackbone directly. This is the same method as the four current methods (refer to §3).
`BackboneProtocol` is a possible improvement for later (§3.2).
We will do it when a second backbone implementation exists or when the BEF dual-tower model starts.

> **Practical rule**: The important point is not "add TerraMind". The important point is **to add it with the smallest interface**.
> A frequent error is "inherit everything that you use". Then the external framework controls the project structure.
> If you replace the framework, you must change half of the codebase.
> We do not define a Protocol now. But we still write the code in this shape: "input dict, output pyramid list, expose out_channels".
> We remove one file, not the design.

---

## 1. Quick Glossary

| Term | Common name | Description |
|---|---|---|
| Foundation Model (FM) | Foundation model | A general representation model, pretrained on large-scale remote sensing data |
| TerraMind v1 | Original name kept | A multimodal remote sensing foundation model from IBM/ESA |
| terratorch | Original name kept | The training framework that officially supports TerraMind. This project uses only its backbone factory |
| Pyramid features | Feature pyramid | Multi-level feature maps for the UNet decoder |
| Tokenizer | Tokenizer | A module that divides an image into a token sequence. Multimodal FMs frequently use this design |
| L2A / GRD | Original name kept | Sentinel-2 atmospherically corrected product / Sentinel-1 ground range detected product |

---

## 2. Decision Summary

| Decision point | Selection | Reason |
|---|---|---|
| Integration method | Add `terratorch` as a uv dependency, and **call only its backbone factory** | We do not have to rewrite the HuggingFace weight loading logic |
| Parts not used | Do not inherit `SemanticSegmentationTask`. Do not use the registry. Do not use the YAML CLI | Keep the project script + JSON conventions. Do not have two conflicting systems |
| Backbone wrapper form | A plain `nn.Module`. Input is `dict[str, Tensor]`. Output is a pyramid feature list | A general shape. Any downstream decoder can connect to it |
| Abstract interface layer | **Not in this phase**. The model class builds the backbone directly. The config sets it | With only one backbone implementation, a Protocol has nothing to abstract. Refer to the trigger conditions in §3.2 |
| Modality support | In the first phase, `S2L2A` and `S1GRD` each have an independent backbone (no shared weights) | This agrees with the TerraMind pretraining practice. It also works if a downstream task needs independent towers |
| Offline weights | Support the `backbone_local_ckpt` setting | You can use it when the training node has no network |
| Default output layers | `select_indices = [2, 5, 8, 11]` | This agrees with the usual UNet decoder configuration for TerraMind base |

> This document describes only the **representation layer**. The caller assembles the downstream tasks (single-modality segmentation, multi-task, EDL, BEF fusion).

---

## 3. How to Build the Backbone: The Config Sets It, the Model Builds It Directly

**Currently we build the backbone directly**. We do not add an abstract interface layer.
`FMBaselineSegModel.__init__` reads the `backbone_*` fields from the config and calls `TerraMindBackbone(...)` directly:

```python
class FMBaselineSegModel(pl.LightningModule):
    def __init__(self, model_params):
        h = model_params["hyperparameters"]
        self.backbone = TerraMindBackbone(
            variant         = h["backbone"],                 # "terramind_v1_base"
            modalities      = tuple(h["backbone_modalities"]),
            pretrained      = h["backbone_pretrained"],
            local_ckpt_path = h["backbone_local_ckpt"],
            select_layers   = tuple(h["backbone_select_layers"]),
            freeze          = h["backbone_freeze"],
        )
        self.decoder = UNetDecoder(in_channels=self.backbone.out_channels)
        self.head    = Conv1x1(h["num_classes"])
```

The `model_type` in the config sets which backbone to use. The dispatch in the script selects the related model class (refer to §8, §9).
This is fully the same as the current `v2` / `EDL` / `ensemble` / `dropout` method. We add no new mechanism.

### 3.1 The Only Requirement: The Backbone Exposes Two Attributes

We do not define a Protocol. But `TerraMindBackbone` **must still provide**:

```python
out_channels: list[int]        # 每層 pyramid 的 channel 數，例如 [96, 192, 384, 768]
downsample_ratios: list[int]   # 每層相對原圖的下採樣倍數，例如 [4, 8, 16, 32]
```

This is not for a contract. The decoder really needs these attributes.
Without these two attributes, there are only two alternatives: hard-code the channel numbers, or run one dummy forward in `__init__` to find them.
Both alternatives are worse.

People frequently forget `downsample_ratios`. TerraMind is a ViT, so all layers can have the same resolution (`[16,16,16,16]`).
This is different from the `[4,8,16,32]` pattern of a CNN backbone. The decoder needs this attribute to set the correct number of upsampling layers.
This also relates directly to item 4 in the risk register (the token → image reshape order).

### 3.2 Possible Improvement for Later: `BackboneProtocol`

**We do not implement it now**. We record it here as a direction for later. We will do it when one of these conditions occurs:

- A second backbone implementation is necessary (for example, a change to Prithvi / SatMAE, or a change to HRNet to expose its current pyramid)
- The BEF dual-tower model starts, and `models/bef.py` must inject two replaceable backbones
- We want the current `EDL_ML4FloodsModel` to also accept an FM backbone

At that time, we add `flood_uncertainty/models/backbones/base.py`:

```python
class BackboneProtocol(Protocol):
    def __call__(self, image_dict: dict[str, Tensor]) -> list[Tensor]: ...
    out_channels: list[int]
    downsample_ratios: list[int]
```

**The migration cost is almost zero**. `Protocol` uses structural typing, so the implementation does not have to inherit it.
The code must keep the two attributes in §3.1 and the `dict → list[Tensor]` call convention in §5.
If it does, `TerraMindBackbone` satisfies this contract when we finish writing it.
The `base.py` file only lets mypy / pyright do the check.

> **Thus, the rule for this phase**: The implementation can skip the abstraction for now. But **write the call convention in the shape of the contract**.
> We remove one file, not the design.

---

## 4. File Locations

```
flood-uncertainty/
├── flood_uncertainty/
│   ├── models/
│   │   ├── backbones/                       ← 【新增】子套件
│   │   │   ├── __init__.py
│   │   │   └── terramind_backbone.py        ← TerraMind 包裝
│   │   │       （base.py / BackboneProtocol 暫不建立，見 §3.2）
│   │   ├── fm_baseline.py                   ← 【新增】TerraMind + UNet decoder + head
│   │   ├── edl.py                           ← 不動
│   │   └── ...                              ← 不動
│   ├── losses/ metrics/ inference/ utils/   ← 不動（config_loader 已足夠通用）
│
├── scripts/
│   ├── train/train_fm_baseline.py           ← 【新增】以 train_v2.py 為樣板
│   ├── train/（其餘）                        ← 不動
│   ├── inference/                           ← 不動（baseline 沿用 v2 分支）
│   └── smoke/
│       ├── smoke_train_fm.sh                ← 【新增】
│       └── smoke_train.sh                   ← 【修改】dispatch 新增 FM_BASELINE
│
├── configurations/fm_baseline.json          ← 【新增】對齊 v2.json 的 schema
├── tests/
│   ├── test_terramind_backbone.py           ← 【新增】
│   └── test_fm_baseline_smoke.py            ← 【新增】
└── pyproject.toml                           ← 【修改】新增 terratorch 依賴
```

The "unchanged" items are the important point. The current EDL, v2, ensemble, and dropout training pipelines have **zero risk**.
The FM baseline is a fully parallel new branch. If we find a problem later, we remove the full package. This does not affect the current experiment results.

**Not in this list**: the EDL loss, the Dirichlet fusion, and the paired S1/S2 dataset. These are downstream or data-layer work.

---

## 5. Data Flow: The Translation Is Only One Line

The current `WorldFloodsDataset` gives `(B, C, H, W)`. TerraMind needs a dict. The translation is only in
`FMBaselineSegModel.forward()`:

```python
class FMBaselineSegModel(pl.LightningModule):
    def __init__(self, model_params):
        self.backbone = TerraMindBackbone(          # 呼叫 terratorch 拿預訓權重
            variant="terramind_v1_base",
            modalities=("S2L2A",),                  # 單模態，簡化 baseline
        )
        self.decoder = UNetDecoder(
            in_channels=self.backbone.out_channels, # 用骨幹暴露的屬性自動配置
        )
        self.head = Conv1x1(num_classes)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)   # 對齊專案 label 慣例

    def forward(self, batch):
        image_dict = {"S2L2A": batch["image"]}      # ← 唯一翻譯層
        features   = self.backbone(image_dict)
        return self.head(self.decoder(features))
```

`WorldFloodsDataset`, `pl.Trainer`, the config schema, and `utils/config_loader` have **zero changes**.
The Trainer knows only `pl.LightningModule`. It cannot see if TerraMind or ml4floods is inside.

The training data uses the current S2-only batch. It uses only `mask[:, 1]` (the water task) and ignores cloud.
This is a baseline simplification.

---

## 6. TerraMind Backbone Implementation

```python
class TerraMindBackbone(nn.Module):
    def __init__(self,
                 variant: str = "terramind_v1_base",
                 modalities: tuple[str, ...] = ("S2L2A",),
                 pretrained: bool = True,
                 select_layers: tuple[int, ...] = (2, 5, 8, 11),
                 local_ckpt_path: str | None = None,
                 freeze: bool = False): ...
```

**Internal procedure**

1. Call `terratorch.models.backbones.<terramind factory>` to build the encoder.
2. If `local_ckpt_path` has a value, skip the HuggingFace download and read the local `.ckpt`.
3. Forward:
   - The encoder gives a `(B, n_tokens, D)` token sequence for each layer.
   - `SelectIndices(select_layers)` gets the specified layers.
   - `ReshapeTokensToImage`: `(B, n_tokens, D) → (B, D, H', W')`
   - `LearnedInterpolateToPyramidal`: makes a 4-level multi-scale pyramid.
4. Return `list[Tensor]`.

The three helpers are plain `nn.Module` classes in the same file. **They are not in the terratorch registry**.

**Offline weights**: `local_ckpt_path` can point to a manually downloaded
`artifacts/backbones/terramind_v1_base.ckpt`. It can also be a symbolic link from the current HuggingFace cache.

**Freeze**: `freeze=True` sets `.requires_grad` off for all encoder parameters. Use it to train only the decoder + head.

**Dependency**: Add `terratorch` to `pyproject.toml`, then run `uv sync --python 3.10`.
Before you start, run `uv add terratorch --dry-run`. Make sure that it does not conflict with `ml4floods` / `lightning`.
If there is a conflict, first try to upgrade `lightning`.

---

## 7. Config: Keep the Same Four-Section Structure

`configurations/fm_baseline.json` keeps the four sections `shared / train / infer / validate_only`.
`utils/config_loader.load_mode_config()` can read it with **zero changes**.

New fields in `shared.model_params.hyperparameters`:

| Field | Type | Purpose |
|---|---|---|
| `model_type` | `"fm_baseline"` | For the dispatch |
| `backbone` | `"terramind_v1_base"` | TerraMind variant |
| `backbone_modalities` | `["S2L2A"]` or `["S1GRD"]` | Modalities that the backbone accepts |
| `backbone_pretrained` | `bool` | Download from HF or not |
| `backbone_local_ckpt` | `str \| null` | Path of the offline weights |
| `backbone_select_layers` | `[int, int, int, int]` | Encoder layers to get. Default is `[2, 5, 8, 11]` |
| `backbone_freeze` | `bool` | Freeze the backbone or not |
| `decoder_channels` | `[int, int, int, int]` | Channels of each UNet decoder stage. Default is `[512, 256, 128, 64]` |

Current keys such as `channel_configuration` stay the same. The config **does not include** uncertainty, fusion, or multi-task fields.

---

## 8. Script and Smoke

`scripts/train/train_fm_baseline.py` uses `train_v2.py` as a template. **Change only the line that builds the model**:

```python
# train_v2.py：
model = get_model(config.model_params)                # ml4floods 那顆
# train_fm_baseline.py：
model = FMBaselineSegModel(config.model_params)       # 內部組 TerraMind + decoder + head
```

Copy all other logic word for word (`--config` / `--mode` / `--resume_ckpt` / `--data_root`, callbacks, logger,
optimizer, `trainer.fit()`). The CLI is then the same as for the other models:

```bash
uv run python scripts/train/train_fm_baseline.py --config configurations/fm_baseline.json --mode train
```

The smoke test uses the current `SMOKE_TRAIN_MODEL` dispatch. Add only one line to `smoke_train.sh`:

```bash
SMOKE_TRAIN_MODEL=FM_BASELINE bash scripts/smoke/smoke_train.sh
```

**Inference**: In the baseline phase, use the `v2` branch of `run_inference.py` (single output). You do not have to add a new
model_type.

---

## 9. Parallel to the Current Model Types

| model_type | Backbone | Purpose |
|---|---|---|
| `v2` | ml4floods `get_model()` | baseline segmentation |
| `EDL` | ml4floods `get_model()` | Single-model uncertainty |
| `EDL-SAR` | ml4floods `get_model()` (SAR only) | SAR version |
| `ensemble` | ml4floods `get_model()` (multiple models) | Deep ensemble uncertainty |
| `dropout` | ml4floods `get_model()` (with dropout) | MC Dropout |
| **`fm_baseline` (new)** | **TerraMind v1 base** | **Verify the backbone connection and that the representation is usable** |

Each model type has its own config, script, and model class. They are **fully parallel**.
In practice, you can run fm_baseline experiments and EDL training on the same branch at the same time.
They read different configs, run different scripts, and write to different checkpoint directories.

---

## 10. Handshake with BEF

When the BEF side is complete, there will be a `models/bef.py`. Its `__init__` must inject two backbones:

```python
model = BEF_ML4FloodsModel(
    model_params,
    backbone_s1=TerraMindBackbone(modalities=("S1GRD",), ...),
    backbone_s2=TerraMindBackbone(modalities=("S2L2A",), ...),
)
```

**The injection is the handshake**. The FM side only makes an `nn.Module` that accepts a dict, gives a pyramid list, and exposes `out_channels`.
The BEF side only uses it. The two sides finish independently.
They meet only at assembly time in `scripts/train/train_bef.py`.

The BEF dual-tower model is one of the trigger conditions in §3.2. At that time, there are two backbone instances (S1 / S2).
We may also want to replace the backbone type. Then `BackboneProtocol` has a real thing to abstract.

For more information, refer to [`BEF_integration_plan.md`](./BEF_integration_plan.md).

---

## 11. Milestones

```
F1 依賴解析 ─▶ F2 骨幹包裝 ─▶ F3 離線權重接口 ─▶ F4 baseline 模型 smoke
                                                          │
                                                          ▼
                                                F5 baseline train smoke ─▶ F6 完整訓練驗證
```

| Milestone | Deliverable | Verification | Estimate |
|---|---|---|---|
| **F1** | Add terratorch to `pyproject.toml` + `uv sync` is successful | `uv run python -c "import terratorch"` passes | 0.5 days |
| **F2** | `backbones/terramind_backbone.py` | CPU dummy input → correct pyramid shape. `out_channels` / `downsample_ratios` agree with the real output | 1 day |
| **F3** | Offline ckpt path interface | With a fake ckpt path, make sure that the HF download is skipped | 0.5 days |
| **F4** | `models/fm_baseline.py` + smoke | `pytest tests/test_fm_baseline_smoke.py` runs one forward + backward | 0.5 days |
| **F5** | `fm_baseline.json` + `train_fm_baseline.py` + smoke | `SMOKE_TRAIN_MODEL=FM_BASELINE` runs 1 epoch | 0.5 days |
| **F6** | Full training on sen1floods11 train/val. Record the mIoU | Better than or at least equal to the current v2 baseline | Not included |

**F1–F4 are only connection work and do not need a GPU**. F5–F6 include real training.
The milestones **do not depend on the selection of the downstream task**.
F1–F5 do not have to touch uncertainty, fusion, multi-task, or related topics.

The total is approximately 2.5–3 days + the F6 training time.

---

## 12. Risk Register

| # | Risk | Countermeasure |
|---|---|---|
| 1 | `terratorch` has a version conflict with the current `ml4floods` / `lightning` | Run `uv add --dry-run` before F1. If necessary, lock the `lightning` version |
| 2 | The HuggingFace weight download fails on the training node | Define and test the `backbone_local_ckpt` interface in F2 |
| 2b | We do not abstract the interface now. When we add the Protocol later, the call convention has changed | The F2 test asserts that `out_channels` / `downsample_ratios` agree with the real output. This locks the shape |
| 3 | GPU memory pressure from the TerraMind base backbone (especially the BEF dual-tower model) | `freeze=True`, `precision: bf16-mixed`, a small batch, gradient checkpointing |
| 4 | Incorrect coordinate order in the token → image reshape | The F2 dummy input test checks the spatial anchor (top-left pixel ↔ top-left token) |
| 5 | The baseline mIoU is much lower than v2 | Check three items: (a) the band order agrees with the TerraMind L2A expectation (b) the normalization statistics (c) `select_layers` is reasonable |
| 6 | A `terratorch` update breaks the API | Lock the minor version in `pyproject.toml`. Add a smoke test to CI to find problems early |

**Not in the risk scope of this document**: fusion formula errors, multi-branch loss weights, and KL annealing behavior. These are downstream task work.

---

## 13. Related Documents

- [`BEF_integration_plan.md`](./BEF_integration_plan.md): BEF fusion method (downstream, can be done independently)
- [`architecture.md`](./architecture.md): Modules and class diagrams of the current architecture
