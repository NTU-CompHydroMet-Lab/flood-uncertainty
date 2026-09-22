# 基礎模型骨幹整合（首例：TerraMind v1）

> 範圍：把遙測基礎模型（Foundation Model, FM）作為 backbone 引入 flood-uncertainty，
> 供任何下游 decoder + head 使用。首例採用 IBM/ESA **TerraMind v1**，
> 透過 `terratorch` 取得預訓權重。
>
> **本文件不涉及不確定性方法**。骨幹輸出的 pyramid feature 是通用介面，可餵給既有
> EDL、baseline segmentation、或 [`BEF_integration_plan.md`](./BEF_integration_plan.md)
> 描述的融合方法，兩件工作可獨立進行。

---

## 0. TL;DR

**只借 backbone factory，不套框架全家桶。** `terratorch` 加為 uv 依賴，但僅呼叫其
backbone factory 一個函式——不繼承 Task、不用 registry、不用 YAML CLI。保留專案原有
`pl.LightningModule` + JSON config + script 慣例。

TerraMind 進來只多出「**一顆骨幹、一支 script、一份 config**」，其他一律不動。

```
原本：  [ml4floods backbone] ── decoder ── head
換成：  [TerraMind backbone] ── decoder ── head
                 ▲
                 └── 只有這顆換了；decoder、head、資料、Trainer、config schema 都不動
```

工作量約 2.5–3 天 + 完整訓練時間。

> **實務原則**：關鍵不是「引入 TerraMind」，而是**引入時只切最小介面**。常見錯誤是
> 「用什麼就繼承什麼」，導致專案結構被外部框架綁架、換掉時要動半個 codebase。此處
> `BackboneProtocol` 只有兩個屬性 + 一個函式，是能想像的最小約束——介面越薄，上下游
> 各自的實作自由度越高，換元件成本越低。

---

## 1. 名詞速查

| 術語 | 中文 | 說明 |
|---|---|---|
| Foundation Model (FM) | 基礎模型 | 於大規模遙測資料預訓的通用表徵模型 |
| TerraMind v1 | 保留原名 | IBM/ESA 發布的多模態遙測基礎模型 |
| terratorch | 保留原名 | TerraMind 官方支援之訓練框架，本專案僅取其 backbone factory |
| Pyramid features | 特徵金字塔 | 多層級特徵圖，供 UNet decoder 使用 |
| Tokenizer | 分詞器 | 將影像切為 token 序列的模組，多模態 FM 常見設計 |
| L2A / GRD | 保留原名 | Sentinel-2 大氣校正產品 / Sentinel-1 地距校正產品 |

---

## 2. 決策摘要

| 決策點 | 選擇 | 理由 |
|---|---|---|
| 引入方式 | `terratorch` 加為 uv 依賴，**僅呼叫其 backbone factory** | 避免自行重寫 HuggingFace 權重載入邏輯 |
| 不採用範圍 | 不繼承 `SemanticSegmentationTask`、不用 registry、不用 YAML CLI | 保留專案 script + JSON 慣例，避免雙頭馬車 |
| 骨幹包裝形式 | 純 `nn.Module`，輸入 `dict[str, Tensor]`，輸出 pyramid feature list | 通用介面，任何下游 decoder 皆可接 |
| 模態支援 | 首階段 `S2L2A` 與 `S1GRD` 各自獨立骨幹（不共用權重） | 對齊 TerraMind 預訓習慣，下游若需獨立塔亦適用 |
| 離線權重 | 支援 `backbone_local_ckpt` 設定 | 訓練節點無網路時可用 |
| 預設輸出層 | `select_indices = [2, 5, 8, 11]` | 對齊 TerraMind base 常見 UNet decoder 配置 |

> 本文件只描述**表徵層**。下游任務（單模態分割、多任務、EDL、BEF 融合）由呼叫端組裝。

---

## 3. 介面契約：對外的唯一握手

任何實作於 `flood_uncertainty/models/backbones/` 底下的骨幹須符合：

```python
class BackboneProtocol(Protocol):
    def __call__(self, image_dict: dict[str, Tensor]) -> list[Tensor]:
        """
        input:  image_dict — key 為模態名稱（"S2L2A"、"S1GRD"），value shape (B, C, H, W)
        output: list[Tensor] — pyramid feature，通常 4 層
        """

    out_channels: list[int]        # 每層 pyramid 的 channel 數
    downsample_ratios: list[int]   # 每層相對原圖的下採樣倍數
```

- **輸入 dict**：讓多模態（S2L2A、S1GRD、未來其他）自然擴展
- **輸出 pyramid**：讓下游 decoder 用 `out_channels` 自動配置，不需硬編通道數

**契約不含**預訓權重載入方式、模態組合限制、tokenizer 設計——這些是實作細節。

換 TerraMind 為 Prithvi / SatMAE，只需在 `models/backbones/` 新增一份符合契約的實作，
下游任務零改動。

---

## 4. 檔案落點

```
flood-uncertainty/
├── flood_uncertainty/
│   ├── models/
│   │   ├── backbones/                       ← 【新增】子套件
│   │   │   ├── __init__.py
│   │   │   ├── base.py                      ← BackboneProtocol 介面契約
│   │   │   └── terramind_backbone.py        ← TerraMind 包裝
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

「不動」欄位是重點：現有 EDL、v2、ensemble、dropout 訓練管線**零風險**——FM baseline 是
完全並列的新分支。回頭發現不對就整包移除，不影響既有實驗結果。

**不在此清單**：EDL 損失、Dirichlet 融合、成對 S1/S2 dataset——屬下游/資料層工作。

---

## 5. 資料流：翻譯只有一行

既有 `WorldFloodsDataset` 吐 `(B, C, H, W)`，TerraMind 要 dict。翻譯集中在
`FMBaselineSegModel.forward()`：

```python
class FMBaselineSegModel(pl.LightningModule):
    def __init__(self, model_params):
        self.backbone = TerraMindBackbone(          # 呼叫 terratorch 拿預訓權重
            variant="terramind_v1_base",
            modalities=("S2L2A",),                  # 單模態，簡化 baseline
        )
        self.decoder = UNetDecoder(
            in_channels=self.backbone.out_channels, # 用契約暴露的屬性自動配置
        )
        self.head = Conv1x1(num_classes)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)   # 對齊專案 label 慣例

    def forward(self, batch):
        image_dict = {"S2L2A": batch["image"]}      # ← 唯一翻譯層
        features   = self.backbone(image_dict)
        return self.head(self.decoder(features))
```

`WorldFloodsDataset`、`pl.Trainer`、config schema、`utils/config_loader` 全部**零改動**：
Trainer 只認 `pl.LightningModule`，看不到裡面是 TerraMind 還是 ml4floods。

訓練資料沿用既有 S2-only batch，只取 `mask[:, 1]`（water 任務）、忽略 cloud，這是
baseline 簡化。

---

## 6. TerraMind 骨幹實作

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

**內部流程**

1. 呼叫 `terratorch.models.backbones.<terramind factory>` 建構 encoder
2. 若 `local_ckpt_path` 有值，跳過 HuggingFace 下載改讀本地 `.ckpt`
3. Forward：
   - Encoder 輸出每層 `(B, n_tokens, D)` token 序列
   - `SelectIndices(select_layers)` 抽取指定層
   - `ReshapeTokensToImage`：`(B, n_tokens, D) → (B, D, H', W')`
   - `LearnedInterpolateToPyramidal`：多尺度化為 4 層 pyramid
4. 回傳 `list[Tensor]`

三個 helper 以純 `nn.Module` 實作於同檔內，**不進 terratorch registry**。

**離線權重**：`local_ckpt_path` 可指向手動下載的
`artifacts/backbones/terramind_v1_base.ckpt`，或由既有 HuggingFace cache 符號連結而來。

**凍結**：`freeze=True` 關閉所有 encoder 參數 `.requires_grad`，用於只訓 decoder + head。

**依賴**：`pyproject.toml` 加 `terratorch`，執行 `uv sync --python 3.10`。動手前先
`uv add terratorch --dry-run` 檢查是否與 `ml4floods` / `lightning` 衝突；若衝突，優先
方向為升級 `lightning`。

---

## 7. Config：四段結構完全沿用

`configurations/fm_baseline.json` 維持 `shared / train / infer / validate_only` 四段，
`utils/config_loader.load_mode_config()` **零改動**即可讀取。

`shared.model_params.hyperparameters` 新增欄位：

| 欄位 | 型別 | 用途 |
|---|---|---|
| `model_type` | `"fm_baseline"` | 給 dispatch 用 |
| `backbone` | `"terramind_v1_base"` | TerraMind 變體 |
| `backbone_modalities` | `["S2L2A"]` 或 `["S1GRD"]` | 骨幹接受之模態 |
| `backbone_pretrained` | `bool` | 是否走 HF 下載 |
| `backbone_local_ckpt` | `str \| null` | 離線權重路徑 |
| `backbone_select_layers` | `[int, int, int, int]` | Encoder 抽取層，預設 `[2, 5, 8, 11]` |
| `backbone_freeze` | `bool` | 是否凍結骨幹 |
| `decoder_channels` | `[int, int, int, int]` | UNet decoder 每階通道，預設 `[512, 256, 128, 64]` |

`channel_configuration` 等既有 key 照常沿用。**不含**任何不確定性、融合、多任務欄位。

---

## 8. Script 與 smoke

`scripts/train/train_fm_baseline.py` 以 `train_v2.py` 為樣板，**只換 model 建構那一行**：

```python
# train_v2.py：
model = get_model(config.model_params)                # ml4floods 那顆
# train_fm_baseline.py：
model = FMBaselineSegModel(config.model_params)       # 內部組 TerraMind + decoder + head
```

其餘邏輯（`--config` / `--mode` / `--resume_ckpt` / `--data_root`、callbacks、logger、
optimizer、`trainer.fit()`）逐字複製，CLI 體驗與其他 model 一致：

```bash
uv run python scripts/train/train_fm_baseline.py --config configurations/fm_baseline.json --mode train
```

Smoke 沿用既有 `SMOKE_TRAIN_MODEL` dispatch，只需在 `smoke_train.sh` 加一行：

```bash
SMOKE_TRAIN_MODEL=FM_BASELINE bash scripts/smoke/smoke_train.sh
```

**推論**：baseline 階段沿用 `run_inference.py` 的 `v2` 分支即可（單張輸出），不需新增
model_type。

---

## 9. 與既有 model type 並列

| model_type | 骨幹 | 用途 |
|---|---|---|
| `v2` | ml4floods `get_model()` | baseline segmentation |
| `EDL` | ml4floods `get_model()` | 單模型不確定性 |
| `EDL-SAR` | ml4floods `get_model()`（SAR 專用） | SAR 版本 |
| `ensemble` | ml4floods `get_model()`（多顆） | 深度整合不確定性 |
| `dropout` | ml4floods `get_model()`（加 dropout） | MC Dropout |
| **`fm_baseline`（新）** | **TerraMind v1 base** | **驗證骨幹接線與表徵可用性** |

每個 model type 各自有 config、script、model 類別，**完全平行**。實務上可以在同一個
branch 一邊跑 fm_baseline 實驗，另一邊 EDL 訓練照跑不誤——它們讀不同 config、跑不同
script、產出不同 checkpoint 目錄。

---

## 10. 與 BEF 握手

BEF 側完成後會有一支 `models/bef.py`，其 `__init__` 需注入兩顆 backbone：

```python
model = BEF_ML4FloodsModel(
    model_params,
    backbone_s1=TerraMindBackbone(modalities=("S1GRD",), ...),
    backbone_s2=TerraMindBackbone(modalities=("S2L2A",), ...),
)
```

**注入即握手**——FM 側只負責造出符合 `BackboneProtocol` 的模組，BEF 側只負責拿進來用。
兩邊在時間軸上獨立完成，最後在 `scripts/train/train_bef.py` 組裝時才碰面。

詳見 [`BEF_integration_plan.md`](./BEF_integration_plan.md)。

---

## 11. 里程碑

```
F1 依賴解析 ─▶ F2 骨幹包裝 ─▶ F3 離線權重接口 ─▶ F4 baseline 模型 smoke
                                                          │
                                                          ▼
                                                F5 baseline train smoke ─▶ F6 完整訓練驗證
```

| 里程碑 | 交付內容 | 驗證方式 | 估算 |
|---|---|---|---|
| **F1** | `pyproject.toml` 加 terratorch + `uv sync` 成功 | `uv run python -c "import terratorch"` 通過 | 0.5 天 |
| **F2** | `backbones/base.py` + `terramind_backbone.py` | CPU dummy input → pyramid shape 正確；屬性合理 | 1 天 |
| **F3** | 離線 ckpt 路徑接口 | 給定假 ckpt 路徑檢查跳過 HF 下載 | 0.5 天 |
| **F4** | `models/fm_baseline.py` + smoke | `pytest tests/test_fm_baseline_smoke.py` 一輪 forward + backward | 0.5 天 |
| **F5** | `fm_baseline.json` + `train_fm_baseline.py` + smoke | `SMOKE_TRAIN_MODEL=FM_BASELINE` 跑 1 epoch | 0.5 天 |
| **F6** | sen1floods11 train/val 完整訓練，記錄 mIoU | 對比既有 v2 baseline 有進步或至少相當 | 另計 |

**F1–F4 為純接線工作，無需 GPU**；F5–F6 涉及訓練實境。里程碑**不依賴下游任務選型**：
F1–F5 可完全不觸及不確定性、融合、多任務等議題。

總計約 2.5–3 天 + F6 訓練時間。

---

## 12. 風險登記

| # | 風險 | 對策 |
|---|---|---|
| 1 | `terratorch` 與現有 `ml4floods` / `lightning` 版本衝突 | F1 前先 `uv add --dry-run`；必要時鎖 `lightning` 版本 |
| 2 | HuggingFace 權重於訓練節點下載失敗 | `backbone_local_ckpt` 接口於 F2 就定義並測試 |
| 3 | TerraMind base 骨幹顯存壓力（尤其 BEF 雙塔） | `freeze=True`、`precision: bf16-mixed`、小 batch、gradient checkpointing |
| 4 | Token → image reshape 座標順序錯誤 | F2 dummy input 測試檢查空間 anchor（左上像素 ↔ 左上 token） |
| 5 | Baseline mIoU 明顯低於 v2 | 檢查三件事：(a) 波段順序是否符合 TerraMind L2A 預期 (b) 正規化統計 (c) `select_layers` 是否合理 |
| 6 | `terratorch` 更新破壞 API | `pyproject.toml` 鎖定 minor 版本，CI 加 smoke test 早期偵測 |

**不在本文件風險範圍**：融合公式錯誤、多分支損失權重、KL 退火行為——屬下游任務工作。

---

## 13. 相關文件

- [`BEF_integration_plan.md`](./BEF_integration_plan.md)：BEF 融合方法（下游，可獨立進行）
- [`architecture.md`](./architecture.md)：既有架構的模組與類別圖
