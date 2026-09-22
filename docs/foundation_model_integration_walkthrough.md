# Foundation Model 整合怎麼放進 flood-uncertainty 原架構

> 本文件說明**位置與流程**：TerraMind 骨幹以哪個檔案、哪個資料流位置、
> 什麼樣的 config 與 script 樣板整合進 flood-uncertainty 既有架構。
>
> 若要看整合的**決策脈絡、里程碑、風險登記**，請參考
> [`foundation_model_integration_plan.md`](./foundation_model_integration_plan.md)。

---

## 1. 先看 flood-uncertainty 原架構

未引入 FM 之前，任何一個 model 的訓練流程長這樣：

```
                           configurations/v2.json
                                   │  （JSON config，四段：shared / train / infer / validate_only）
                                   ▼
                     scripts/train/train_v2.py
                                   │  （用 utils/config_loader 讀 config）
                                   ▼
              ┌────────────────────────────────────────────┐
              │  pl.LightningModule（例如 EDL_ML4FloodsModel）│
              │                                            │
              │   backbone ── decoder ── head              │
              │   （ml4floods get_model() 產出）             │
              │                                            │
              │   training_step / validation_step ...      │
              └────────────────────────────────────────────┘
                                   ▲
                                   │  batch: {"image": (B,C,H,W), "mask": (B,2,H,W)}
                                   │
                           WorldFloodsDataset
                                   ▲
                                   │
                           NAS 上的 sen1floods11
```

要記得的三個位置：

- **backbone**：在 `pl.LightningModule` 裡面，目前是 ml4floods `get_model()` 產出的那顆
- **script**：一種 model 一支（`train_v2.py` / `train_edl.py` / `train_dropout.py` / `train_ensemble.py`）
- **config**：一種 model 一份 JSON

---

## 2. FM 進來的唯一入口：backbone 這一顆

TerraMind 不是取代整個架構，而是**只替換 backbone 那一顆**：

```
原本：  [ml4floods backbone] ── decoder ── head
換成：  [TerraMind backbone] ── decoder ── head
                 ▲
                 └── 只有這顆換了；decoder、head、資料、Trainer、config schema 都不動
```

其他所有東西（`pl.LightningModule` 骨架、`pl.Trainer`、config 四段結構、`WorldFloodsDataset`、smoke script 慣例）**完全不動**——這是為什麼 FM 整合可以獨立進行、不影響既有 pipeline。

> **設計原則**：常見錯誤方向是「把 terratorch 整套 CLI + registry + YAML 搬進來」，這會讓專案變成雙頭馬車（JSON config vs YAML config、script 呼叫 vs CLI 呼叫）。此處刻意**只借 terratorch 的 backbone factory 這一個函式呼叫**，其他一律用專案原有慣例——這是「引入外部框架時只切最小介面」的實務原則。

---

## 3. 檔案落點：新增 / 修改 / 不動

```
flood-uncertainty/
├── flood_uncertainty/
│   ├── models/
│   │   ├── backbones/                            ← 【新增】子套件
│   │   │   ├── __init__.py
│   │   │   ├── base.py                           ← 【新增】BackboneProtocol 介面契約
│   │   │   └── terramind_backbone.py             ← 【新增】TerraMind 包裝
│   │   ├── fm_baseline.py                        ← 【新增】示範模型：TerraMind + UNet decoder + head
│   │   ├── edl.py                                ← 不動
│   │   └── ...                                   ← 不動
│   ├── losses/                                    ← 不動
│   ├── inference/                                 ← 不動（baseline 沿用 v2 分支即可）
│   ├── metrics/                                   ← 不動
│   └── utils/                                     ← 不動（config_loader 已足夠通用）
│
├── scripts/
│   ├── train/
│   │   ├── train_fm_baseline.py                  ← 【新增】複製 train_v2.py 為樣板
│   │   ├── train_v2.py                           ← 不動
│   │   ├── train_edl.py                          ← 不動
│   │   └── ...                                   ← 不動
│   ├── inference/                                 ← 不動（baseline 沿用 v2 分支）
│   └── smoke/
│       ├── smoke_train_fm.sh                     ← 【新增】
│       └── smoke_train.sh                        ← 【可能修改】dispatch 新增 FM_BASELINE
│
├── configurations/
│   ├── fm_baseline.json                          ← 【新增】對齊 v2.json 的 schema
│   └── ...                                       ← 不動
│
├── tests/
│   ├── test_terramind_backbone.py                ← 【新增】
│   └── test_fm_baseline_smoke.py                 ← 【新增】
│
└── pyproject.toml                                 ← 【修改】新增 terratorch 依賴
```

「不動」欄位是重點：現有 EDL、v2、ensemble、dropout 訓練管線**零風險**——FM baseline 是完全並列的新分支，不會影響任何既有實驗。

---

## 4. 資料流變化：只在 model 內部長出「TerraMind 那一顆」

新增的 `models/fm_baseline.py` 內部結構：

```
class FMBaselineSegModel(pl.LightningModule):
    __init__:
        self.backbone = TerraMindBackbone(          ← 呼叫 terratorch 拿預訓權重
            variant="terramind_v1_base",
            modalities=("S2L2A",),
            pretrained=True 或 local_ckpt_path=...
        )
        self.decoder  = UNetDecoder(
            in_channels=self.backbone.out_channels  ← 用介面契約暴露的屬性自動配置
        )
        self.head     = Conv1x1 → num_classes

    forward(batch):
        image_dict = {"S2L2A": batch["image"]}       ← 唯一需要的資料調整：包成 dict
        features   = self.backbone(image_dict)       ← 拿到 pyramid feature list
        x          = self.decoder(features)
        logits     = self.head(x)
        return logits
```

兩個「無感」設計：

1. **資料集完全不動**：既有 `WorldFloodsDataset` 吐的 `batch["image"]` 是 `(B, C, H, W)`。在 `forward()` 內把它包成 `{"S2L2A": tensor}` 是**唯一需要「翻譯」的地方**，因為 TerraMind 需要 dict 輸入（未來要支援 S1+S2 才會需要更多 key）。
2. **`pl.Trainer` 完全不動**：因為它只認 `pl.LightningModule`，看不到裡面是 TerraMind 還是 ml4floods。

---

## 5. Config 怎麼加：新舊 schema 對照

**既有** `configurations/v2.json`（片段）：

```json
{
  "shared": {
    "model_params": {
      "hyperparameters": {
        "model_type": "unet",
        "channel_configuration": "bgriswirs",
        ...
      }
    }
  },
  "train": {...},
  "infer": {...},
  "validate_only": {...}
}
```

**新增** `configurations/fm_baseline.json`：

```json
{
  "shared": {
    "model_params": {
      "hyperparameters": {
        "model_type": "fm_baseline",              ← 給 dispatch 用
        "backbone": "terramind_v1_base",          ← 新
        "backbone_modalities": ["S2L2A"],         ← 新
        "backbone_pretrained": true,              ← 新
        "backbone_local_ckpt": null,              ← 新（無網路時填路徑）
        "backbone_select_layers": [2, 5, 8, 11],  ← 新
        "backbone_freeze": false,                 ← 新
        "decoder_channels": [512, 256, 128, 64],  ← 新
        "channel_configuration": "all",           ← 沿用既有 key
        ...
      }
    }
  },
  "train": {...},
  "infer": {...},
  "validate_only": {...}
}
```

四段結構（`shared / train / infer / validate_only`）完全一樣——這代表 `utils/config_loader.load_mode_config()` **零改動**就能讀新 config。

---

## 6. Script 怎麼加：複製 `train_v2.py` 為樣板

`scripts/train/train_fm_baseline.py` 幾乎就是 `train_v2.py` 的複製品，**只換一行 model 建構**：

```python
# 原本 train_v2.py：
model = get_model(config.model_params)                # ml4floods 那顆

# 新增 train_fm_baseline.py：
model = FMBaselineSegModel(config.model_params)       # 內部組 TerraMind + decoder + head
```

其他所有邏輯（`--config` / `--mode` / `--resume_ckpt` / `--data_root`、callbacks、logger、optimizer、`trainer.fit()`）逐字複製。使用者 CLI 體驗與其他 model 一致：

```bash
uv run python scripts/train/train_fm_baseline.py --config configurations/fm_baseline.json --mode train
```

---

## 7. Smoke 怎麼加：沿用既有 `SMOKE_TRAIN_MODEL` 慣例

專案 smoke 已經有 dispatch 慣例：

```bash
SMOKE_TRAIN_MODEL=v2 bash scripts/smoke/smoke_train.sh
SMOKE_TRAIN_MODEL=EDL bash scripts/smoke/smoke_train.sh
```

只需在 `smoke_train.sh` 內部 dispatch 表新增一行：

```
"FM_BASELINE") uv run python scripts/train/train_fm_baseline.py ... ;;
```

之後：

```bash
SMOKE_TRAIN_MODEL=FM_BASELINE bash scripts/smoke/smoke_train.sh
```

即可跑通。整合到既有 CI / smoke 慣例**零學習成本**。

---

## 8. 與其他 model type 的關係（並列，不干擾）

完成後的 model type 家族：

| model_type | 骨幹 | 用途 |
|---|---|---|
| `v2` | ml4floods `get_model()`（既有） | baseline segmentation |
| `EDL` | ml4floods `get_model()`（既有） | 單模型不確定性 |
| `EDL-SAR` | ml4floods `get_model()`（既有，SAR 專用） | SAR 版本 |
| `ensemble` | ml4floods `get_model()`（既有，多顆） | 深度整合不確定性 |
| `dropout` | ml4floods `get_model()`（既有，加 dropout） | MC Dropout |
| **`fm_baseline`（新）** | **TerraMind v1 base** | **驗證骨幹接線與表徵可用性** |

每個 model type 各自有 config、有 script、有 model 類別，**完全平行**。新增 `fm_baseline` 不影響任何既有分支的可用性與結果。

> **加法式演進**：這個「並列不干擾」的設計，讓 FM 相關里程碑（F1–F6）可以在**不動任何既有 EDL/v2/ensemble 訓練**的前提下完成。實務上可以在同一個 branch 一邊做 fm_baseline 訓練實驗，另一邊 EDL baseline 訓練照跑不誤——它們讀不同 config、跑不同 script、產出不同 checkpoint 目錄。回頭發現不對就整包移除，不影響既有實驗結果。

---

## 9. 與 BEF 咬合的方式（前情提要）

當 BEF 側工作也完成後，會有一支 `models/bef.py`，其 `__init__` 需要注入兩顆 `nn.Module` 當 backbone。到時只需：

```python
model = BEF_ML4FloodsModel(
    model_params,
    backbone_s1=TerraMindBackbone(modalities=("S1GRD",), ...),   ← 引用 FM 側成果
    backbone_s2=TerraMindBackbone(modalities=("S2L2A",), ...),   ← 引用 FM 側成果
)
```

**注入即握手**——FM 側只負責「造出一顆符合 `BackboneProtocol` 的模組」，BEF 側只負責「拿進來用」。兩邊在時間軸上完全獨立完成，最後在 `scripts/train/train_bef.py` 內組裝時才碰面。

BEF 側的合約與里程碑請參考 [`BEF_integration_plan.md`](./BEF_integration_plan.md)。

---

## 10. 一句話總結

**FM 整合的所有工作，都集中在「多長一顆 backbone、多一支 script、多一份 config」**——資料層、Trainer、既有 model 家族一律不動。這讓 FM 引入的**風險面積**壓到最小：新元件出問題只影響 `fm_baseline` 分支，既有 EDL / v2 / ensemble / dropout 訓練繼續運作。
