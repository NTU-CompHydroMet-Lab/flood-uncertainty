# 基礎模型骨幹整合計畫（首例：TerraMind v1）

> 範圍：把遙測基礎模型（Foundation Model, FM）作為 backbone 引入 flood-uncertainty，
> 供任何下游 decoder + head 使用。首例採用 IBM/ESA **TerraMind v1**，
> 透過 `terratorch` 函式庫取得預訓權重。
>
> **本文件不涉及不確定性方法**。骨幹輸出的 pyramid feature 是通用介面，
> 可餵給既有 EDL、baseline segmentation、或另一份文件描述的 BEF 融合方法，
> 兩件工作可獨立進行。
>
> 若要看「新元件放在專案哪個位置、與既有元件如何銜接」的流程解說，請參考
> [`foundation_model_integration_walkthrough.md`](./foundation_model_integration_walkthrough.md)。

---

## 0. 名詞速查

| 術語 | 中文 | 說明 |
|---|---|---|
| Foundation Model (FM) | 基礎模型 | 於大規模遙測資料預訓的通用表徵模型 |
| TerraMind v1 | 保留原名 | IBM/ESA 發布的多模態遙測基礎模型 |
| terratorch | 保留原名 | TerraMind 官方支援之訓練框架，本專案僅取其 backbone factory |
| Pyramid features | 特徵金字塔 | 多層級特徵圖，供 UNet decoder 使用 |
| Tokenizer | 分詞器 | 將影像切為 token 序列的模組，多模態 FM 常見設計 |
| L2A / GRD | 保留原名 | Sentinel-2 大氣校正產品 / Sentinel-1 地距校正產品 |

---

## 1. 決策摘要

| 決策點 | 選擇 | 理由 |
|---|---|---|
| 引入方式 | `terratorch` 加為 uv 依賴，**僅呼叫其 backbone factory** | 避免自行重寫 HuggingFace 權重載入邏輯 |
| 不採用範圍 | 不繼承 `SemanticSegmentationTask`、不用 registry、不用 YAML CLI | 保留 flood-uncertainty 的 script + JSON 慣例 |
| 骨幹包裝形式 | 純 `nn.Module`，輸入 `dict[str, Tensor]`，輸出 pyramid feature list | 通用介面，任何下游 decoder 皆可接 |
| 模態支援 | 首階段 `S2L2A` 與 `S1GRD` 各自獨立骨幹（不共用權重） | 對齊 TerraMind 預訓習慣，且下游若需獨立塔亦適用 |
| 離線權重 | 支援 `backbone_local_ckpt` 設定 | 訓練節點無網路時可用 |
| 預設輸出層 | `select_indices = [2, 5, 8, 11]` | 對齊 TerraMind base 常見 UNet decoder 配置 |

> **關鍵原則**：本文件只描述表徵層。下游任務（單模態分割、多任務、EDL、BEF 融合）由呼叫端組裝，不在此文件範圍。

---

## 2. 目錄與檔案落點

```
flood_uncertainty/
├── models/
│   └── backbones/
│       ├── __init__.py
│       ├── base.py                    # BackboneProtocol：介面契約定義
│       └── terramind_backbone.py      # TerraMindBackbone（terratorch factory 包裝）
├── data/
│   └── multimodal_input.py            # dict[str, Tensor] 輸入慣例的輕量 helper（可選）
└── models/
    └── fm_baseline.py                 # 示範用途：TerraMind + UNet decoder + 一般 CE head

scripts/
├── train/train_fm_baseline.py         # baseline 訓練，驗證骨幹可用
└── smoke/smoke_train_fm.sh            # 純骨幹 smoke

configurations/
└── fm_baseline.json                   # baseline 設定檔

tests/
├── test_terramind_backbone.py         # CPU dummy input → pyramid shape 檢查
└── test_fm_baseline_smoke.py          # baseline 一輪 forward + backward

pyproject.toml                         # 新增 terratorch 依賴
```

**不會出現在本文件**：EDL 損失、Dirichlet 融合、成對 S1/S2 dataset。這些屬下游/資料層工作。

---

## 3. Backbone 介面契約（本文件對外的唯一握手）

任何實作於 `flood_uncertainty/models/backbones/` 底下的骨幹須符合：

```
class BackboneProtocol(Protocol):
    def __call__(self, image_dict: dict[str, Tensor]) -> list[Tensor]:
        """
        input:
            image_dict — key 為模態名稱（例如 "S2L2A"、"S1GRD"），
                         value shape (B, C, H, W)
        output:
            list[Tensor] — pyramid feature，通常 4 層，
                           channel 數與 spatial resolution 由骨幹自定
        """

    out_channels: list[int]        # 每層 pyramid 的 channel 數
    downsample_ratios: list[int]   # 每層相對原圖的下採樣倍數
```

`out_channels` 與 `downsample_ratios` 為屬性，讓下游 decoder 可自動配置通道。

**契約不含**：預訓權重載入方式、模態組合限制、tokenizer 設計——這些為實作細節。

---

## 4. TerraMind 骨幹：`terramind_backbone.py`

### 4.1 類別簽名

```
class TerraMindBackbone(nn.Module):
    def __init__(self,
                 variant: str = "terramind_v1_base",
                 modalities: tuple[str, ...] = ("S2L2A",),
                 pretrained: bool = True,
                 select_layers: tuple[int, ...] = (2, 5, 8, 11),
                 local_ckpt_path: str | None = None,
                 freeze: bool = False): ...
```

### 4.2 內部流程

1. 呼叫 `terratorch.models.backbones.<terramind factory>` 建構 encoder
2. 若 `local_ckpt_path` 有值，跳過 HuggingFace 下載改讀本地 `.ckpt`
3. Forward 時：
   - Encoder 輸出每層 `(B, n_tokens, D)` token 序列
   - `SelectIndices(select_layers)` 抽取指定層
   - `ReshapeTokensToImage`：`(B, n_tokens, D) → (B, D, H', W')`
   - `LearnedInterpolateToPyramidal`：多尺度化為 4 層 pyramid
4. 回傳 `list[Tensor]`

前三個 helper（SelectIndices、ReshapeTokensToImage、LearnedInterpolateToPyramidal）以純 `nn.Module` 實作於同檔內，**不進 terratorch registry**。

### 4.3 離線權重

`local_ckpt_path` 支援兩種來源：

- 手動下載後放於 `artifacts/backbones/terramind_v1_base.ckpt`
- 由既有 HuggingFace cache 目錄符號連結而來

於無網路節點，透過 `configurations/*.json` 傳入路徑；有網路時可省略。

### 4.4 凍結

`freeze=True` 時關閉所有 encoder 參數 `.requires_grad`。用於下游 fine-tuning 時只訓 decoder + head 的情境。

---

## 5. 依賴管理

`pyproject.toml`：

```
[project]
dependencies = [
    ...,
    "terratorch >= <對齊 TerraMind v1 base 支援之版本>",
]
```

執行 `uv sync --python 3.10` 重建 `.venv` 與 `uv.lock`。

**風險預警**：在動手前先執行 `uv add terratorch --dry-run` 檢查解析結果，避免與現有 `ml4floods`、`lightning` 版本衝突。若衝突，優先方向為升級 `lightning`（terratorch 通常要求較新版）。

---

## 6. Baseline 示範：`models/fm_baseline.py` + `train_fm_baseline.py`

**目的**：驗證骨幹接線正確、預訓權重載入成功、下游 decoder 可產出合理分割。與任何不確定性方法無關。

### 6.1 模型組成

```
class FMBaselineSegModel(pl.LightningModule):
    backbone : TerraMindBackbone(modalities=("S2L2A",))    # 單模態，簡化 baseline
    decoder  : UNetDecoder(in_channels=backbone.out_channels)
    head     : Conv1x1 → num_classes
    loss     : nn.CrossEntropyLoss(ignore_index=0)          # 對齊專案 label 慣例
```

### 6.2 訓練資料

沿用既有 `WorldFloodsDataset`（S2-only），batch 為 `{"image": (B, C, H, W), "mask": (B, 2, H, W)}`。訓練時只取 `mask[:, 1]`（water 任務），忽略 cloud，這是 baseline 簡化。

### 6.3 用途

- **接線驗證**：一批資料前後向皆通過
- **表徵可用性**：baseline mIoU 應可達到專案既有 v2 模型附近水準（若明顯較差，表示骨幹接線或權重載入有問題）
- **下游 template**：其他任務（含 BEF）可複製此檔的骨幹注入樣板

---

## 7. 設定檔：`configurations/fm_baseline.json`

維持 `shared / train / infer / validate_only` 四段。新增 `shared.model_params.hyperparameters` 欄位：

| 欄位 | 型別 | 用途 |
|---|---|---|
| `backbone` | `"terramind_v1_base"` | TerraMind 變體 |
| `backbone_modalities` | `["S2L2A"]` 或 `["S1GRD"]` | 骨幹接受之模態 |
| `backbone_pretrained` | `bool` | 是否走 HF 下載 |
| `backbone_local_ckpt` | `str \| null` | 離線權重路徑 |
| `backbone_select_layers` | `[int, int, int, int]` | Encoder 抽取層 |
| `backbone_freeze` | `bool` | 是否凍結骨幹 |
| `decoder_channels` | `[int, int, int, int]` | UNet decoder 每階通道，預設 `[512, 256, 128, 64]` |

**不含**任何不確定性、融合、多任務相關欄位。

---

## 8. Script 對接點

### 8.1 `scripts/train/train_fm_baseline.py`

以 `train_v2.py` 為樣板複製，僅換：

- model 換成 `FMBaselineSegModel(model_params)`
- 骨幹於模型 `__init__` 內建構：`TerraMindBackbone(**backbone_kwargs)`

保留 CLI：`--config / --mode / --resume_ckpt / --data_root`。

### 8.2 `scripts/smoke/smoke_train_fm.sh`

一輪 epoch smoke，允許 `SMOKE_TRAIN_MODEL=FM_BASELINE bash scripts/smoke/smoke_train.sh` 觸發。

### 8.3 推論

Baseline 階段沿用 `run_inference.py` 的 `v2` 分支邏輯即可（單張輸出），不需為 baseline 新增 model_type。當有其他任務組合此骨幹時，該任務自行處理推論輸出格式。

---

## 9. 分階段里程碑

| 里程碑 | 交付內容 | 驗證方式 |
|---|---|---|
| **F1** | `pyproject.toml` 加 terratorch + `uv sync` 成功 | `uv run python -c "import terratorch"` 通過 |
| **F2** | `models/backbones/base.py`（BackboneProtocol）+ `terramind_backbone.py` | CPU dummy input → pyramid shape 正確；`out_channels` / `downsample_ratios` 屬性合理 |
| **F3** | 離線 ckpt 路徑接口驗證 | 給定假 ckpt 路徑檢查跳過 HF 下載邏輯 |
| **F4** | `models/fm_baseline.py` + smoke | `pytest tests/test_fm_baseline_smoke.py`：一輪 forward + backward 通過 |
| **F5** | `configurations/fm_baseline.json` + `train_fm_baseline.py` + smoke 訓練 | `SMOKE_TRAIN_MODEL=FM_BASELINE bash scripts/smoke/smoke_train.sh` 跑 1 epoch |
| **F6** | 於 sen1floods11 train/val 上跑一輪完整訓練，記錄 mIoU | 對比專案既有 v2 baseline 有明顯進步或至少相當 |

F1–F4 為純接線工作，F5–F6 涉及訓練實境。

**里程碑不依賴下游任務選型**：F1–F5 可完全不觸及不確定性、融合、多任務等下游議題。

---

## 10. 風險登記

| # | 風險 | 對策 |
|---|---|---|
| 1 | `terratorch` 版本與現有 `ml4floods` / `lightning` 衝突 | F1 前先 `uv add --dry-run`；必要時鎖 `lightning` 版本 |
| 2 | HuggingFace 權重下載於訓練節點失敗 | `backbone_local_ckpt` 接口於 F2 就要定義並測試 |
| 3 | TerraMind base 骨幹單顆已不小，顯存壓力 | 提供 `freeze=True`、`precision: bf16-mixed`、gradient checkpointing 選項 |
| 4 | Encoder 輸出 token → image 的 reshape 座標順序錯誤 | F2 dummy input 測試需檢查空間 anchor（例如左上角像素對應到左上角 token） |
| 5 | Baseline mIoU 明顯低於 v2 | 檢查：(a) 波段順序是否符合 TerraMind L2A 預期 (b) 正規化統計是否正確 (c) `select_layers` 是否合理 |
| 6 | `terratorch` 更新破壞 API | 在 `pyproject.toml` 鎖定 minor 版本，於 CI 加 smoke test 早期偵測 |

**不在本文件風險範圍**：融合公式錯誤、多分支損失權重、KL 退火行為——這些屬下游任務工作。

---

## 11. 產出物清單

- 新增：`models/backbones/base.py`、`models/backbones/terramind_backbone.py`
- 新增：`models/fm_baseline.py`
- 新增：`configurations/fm_baseline.json`
- 新增：`scripts/train/train_fm_baseline.py`、`scripts/smoke/smoke_train_fm.sh`
- 新增：`tests/test_terramind_backbone.py`、`tests/test_fm_baseline_smoke.py`
- 修改：`pyproject.toml` + 重建 `uv.lock`
- 更新：`README.md`（基礎模型骨幹使用段落）

**不在此清單**：不確定性損失、融合模組、成對資料集、EDL/BEF 相關設定與 script。

---

## 12. 工作量估算

| 階段 | 估算 | 說明 |
|---|---|---|
| F1（依賴解析） | 0.5 天 | 若衝突可能延長 |
| F2（Backbone 包裝 + 測試） | 1 天 | 主要是 reshape 與 pyramid 拼接 |
| F3（離線 ckpt） | 0.5 天 | 主要是 monkey patch HF 下載 |
| F4（Baseline 模型 smoke） | 0.5 天 | 接線工作 |
| F5（Baseline 訓練 smoke） | 0.5 天 | 複製 `train_v2.py` |
| F6（完整 baseline 訓練） | 另計（半日–1 日訓練時間） | 依 GPU 資源 |

總計約 2.5–3 天 + F6 訓練時間。

---

## 13. 與下游任務之握手

本文件與外部工程的唯一介面：**§3 BackboneProtocol**。

只要遵守「輸入 `dict[str, Tensor]`、輸出 pyramid feature `list[Tensor]`、暴露 `out_channels` 與 `downsample_ratios` 屬性」，此骨幹可注入任何下游任務（含既有 EDL、其他文件描述的 BEF 融合、未來新增之任務等）。

下游任務實作可完全不知道此處使用 TerraMind、terratorch、或任何特定基礎模型；當未來替換為其他 FM（例如 Prithvi、SatMAE 等），僅需在 `models/backbones/` 底下新增一份符合契約的實作，下游任務零改動。
