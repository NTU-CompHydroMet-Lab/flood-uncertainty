# BEF 融合方法移植到 flood-uncertainty 的整合計畫

> 來源：`/home/NAS/house/ycchen-10014/repo/rsfm/`（Bayesian Evidential Framework，以下簡稱 BEF）
> 目標：`/home/NAS/house/ycchen-10014/repo/flood-uncertainty/`
> 立場：**完全依照 flood-uncertainty 既有架構**（`pl.LightningModule` + JSON 設定檔 + script 直呼），
> 不引入 terratorch 的 Task / registry / CLI 生態；僅將 `terratorch` 當作提供 TerraMind 骨幹網路的函式庫。

---

## 0. 名詞速查

| 術語 | 中文 | 說明 |
|---|---|---|
| Bayesian Evidential Framework (BEF) | 貝氏證據框架 | 兩個單模態 EDL 模型輸出的 Dirichlet 證據，在 uniform prior 下以 `α_f = α_s1 + α_s2 − 1` 進行後驗融合 |
| Evidential Deep Learning (EDL) | 證據深度學習 | 以 Dirichlet 分布建模 class posterior，回傳 α 而非機率，可分離 aleatoric / epistemic |
| Sentinel-1 (S1) / Sentinel-2 (S2) | 保留原名 | 分別為 C-band SAR 與光學衛星，此專案 tile 命名依 sen1floods11 慣例對齊 |
| SAR | 保留縮寫 | 合成孔徑雷達 |
| Foundation Model (TerraMind v1) | 基礎模型 | IBM/ESA 遙測基礎模型，多模態 tokenizer + Transformer 編碼器 |
| Backbone / Neck / Head | 骨幹網路 / 頸部 / 頭部 | 語意分割模型三段標準組合 |
| Vacuity | 空缺度 | Dempster-Shafer 意義下的整體不確定性，`K / S`，K=類別數，S=Σα |
| Aleatoric / Epistemic | 隨機不確定性 / 知識不確定性 | 以全變異數定律（LoTV）拆解 |
| Annealing (of KL) | KL 退火 | 訓練初期壓低 KL 正則項權重，避免早期 α 全部塌到 1 |
| Multi-branch | 多分支 | 一次前向計算多個分支損失（fused / per-modality），共享或不共享骨幹 |

---

## 1. 決策摘要（先讀這一節）

| 決策點 | 選擇 | 理由 |
|---|---|---|
| 融合層次 | **Late fusion（evidence-sum）** | BEF 的本體公式就在後驗層次；early fusion 或 multi-branch 屬於延伸變體，可留待後續 |
| 資料模態 | **S1 + S2 雙塔** | BEF 的價值來自跨模態 Bayes 融合，僅 S2 內部融合會偏離公式意涵 |
| TerraMind 引入 | **`terratorch` 加為 uv 依賴，僅取用其 backbone factory** | 避免自行重寫 HuggingFace 權重載入邏輯；不繼承其 Task 與 registry |
| 模型類別 | 新增 `BEF_ML4FloodsModel(pl.LightningModule)` | 與既有 `EDL_ML4FloodsModel` 對齊，維持相同 script 風格 |
| 設定檔 | 新增 `configurations/bef.json`，維持 `shared / train / infer / validate_only` 四段 | 讓 `utils/config_loader.load_mode_config` 可直接重用 |
| 標籤相容 | 多任務 head 拆兩支：cloud head 沿用 sigmoid、water head 走 EDL Dirichlet | flood-uncertainty 的 mask 是 `(B, 2, H, W)` 多任務結構，BEF 僅對 water 有意義 |

> **關鍵洞察**：BEF 的數學是純 PyTorch，可以整支檔案級搬移。真正的工作量在於「把 terratorch 的 Task 骨架拆掉、改寫成 flood-uncertainty 慣用的 `pl.LightningModule` 直寫 `training_step / validation_step`」，以及「補上 S1 + S2 對齊的 dataloader」。

---

## 2. 兩專案架構差異對照

| 面向 | rsfm（來源） | flood-uncertainty（目標） |
|---|---|---|
| 訓練入口 | `terratorch` CLI + `configs/*.yaml` | `scripts/train/train_*.py` + `configurations/*.json` |
| Task 基底 | `SemanticSegmentationTask`（繼承 `LightningModule`） | 直接寫 `pl.LightningModule`（`EDL_ML4FloodsModel`） |
| Backbone 取得 | terratorch backbone registry（YAML `backbone: terramind_v1_base`） | 手動 `nn.Module`（`get_model` from ml4floods） |
| 資料模組 | `GenericMultiModalDataModule`（多模態原生） | ml4floods 的 `WorldFloodsDataset`（單一 `input_folder`） |
| 標籤形狀 | `(B, H, W)` 二元 water | `(B, 2, H, W)` 多任務（cloud + water，各 {invalid, neg, pos}） |
| 融合機制 | `ModalityMaskNeck` + evidence-sum | 尚無 |
| KL 退火 knob | 僅 linear（`annealing_step`） | `annealing_mode ∈ {fixed, linear}` + `annealing_coefficient`（commit `e848e5e`） |
| 驗收方式 | `custom_modules/test_*_smoke.py` 純 CPU + 骨幹 mock | `tests/` + `scripts/smoke/*.sh` |

移植時採「flood-uncertainty 慣例覆蓋 rsfm 慣例」，KL 退火兩邊 knob **合流**（詳見 §5.4）。

---

## 3. 目錄與檔案落點

新增與修改（依 flood-uncertainty 命名慣例）：

```
flood_uncertainty/
├── models/
│   ├── bef.py                        # 新增：BEF_ML4FloodsModel（雙塔 + evidence-sum）
│   └── backbones/                    # 新增子套件
│       └── terramind_backbone.py     # 呼叫 terratorch backbone factory，回傳 pyramid feature list
├── losses/
│   ├── edl_dirichlet.py              # 新增：alpha_from_logits / dirichlet_stats / kl_dirichlet /
│   │                                 #        edl_seg_loss / total_variance / annealing_lambda
│   └── edl_loss.py                   # 保留現行 sigmoid-binary EDL 損失，不動
├── fusion/                           # 新增子套件（BEF 本體與 baseline 融合規則）
│   ├── __init__.py
│   ├── evidence_sum.py               # α_f = α_s1 + α_s2 − 1（BEF）
│   ├── prob_avg.py                   # 對照組
│   └── logit_avg.py                  # 對照組
├── data/                             # 新增子套件
│   └── paired_s1s2_dataset.py        # S1/S2 tile 對齊之 dataset 包裝
├── inference/
│   └── infer.py                      # 修改：load_model 新增 "BEF" 分支 + bef_predict
├── metrics/
│   └── segmentation.py               # 修改：新增 vacuity / aleatoric / epistemic 匯出欄位
└── utils/
    └── config_loader.py              # 不動（schema 已足夠通用）

scripts/
├── train/
│   └── train_bef.py                  # 新增：複製 train_edl.py 為樣板
├── inference/
│   └── run_inference.py              # 修改：--model_type 新增 "BEF"
├── eval/
│   └── eval_metrics.py               # 修改：--model_type 新增 "BEF"，接 vacuity 輸出
└── smoke/
    └── smoke_train_bef.sh            # 新增

configurations/
└── bef.json                          # 新增：與 edl.json 同 schema

tests/
├── test_bef_fusion.py                # 純 CPU 單元測試（evidence-sum、KL、退火）
├── test_paired_dataset.py            # S1/S2 對齊測試
└── test_bef_smoke.py                 # mock TerraMind 的整合 smoke

docs/
└── BEF_integration_plan.md           # 本文件

pyproject.toml                        # 新增 terratorch 依賴
uv.lock                               # 由 uv sync 重建
README.md                             # 補上 BEF 用法段落
```

---

## 4. 融合公式重寫（`losses/edl_dirichlet.py` 與 `fusion/evidence_sum.py`）

### 4.1 公式本體

在共同 uniform 先驗 `Dir(1, ..., 1)` 與條件獨立假設下，兩顆單模態模型的 Dirichlet 後驗融合為：

```
α_fused_k = e_s1_k + e_s2_k + 1     （以 evidence 表示）
          = α_s1_k + α_s2_k − 1     （以 concentration 表示）
```

其中 `k = 1..K` 為類別索引，`e_i = α_i − 1`。

### 4.2 `losses/edl_dirichlet.py` 公開介面

> 新建一支模組，不覆蓋 `losses/edl_loss.py`。原因：既有 `EDL_ML4FloodsModel` 走的是 sigmoid-binary EDL（水/雲各 2 個 evidence channel），與此處 K-class softmax-EDL 語意不同，並存可避免破壞現有 EDL 訓練。

```
alpha_from_logits(logits, activation="softplus", clamp_min=1e-6) -> α (B, K, H, W)
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

### 4.3 三種資料項損失（沿用 rsfm 選項）

- `digamma`（預設）：`L_data = Σ_k y_k · (ψ(S) − ψ(α_k))`
- `log`：`L_data = Σ_k y_k · (log S − log α_k)`
- `mse`（Sensoy Eq. 5）：`L_data = ||y − P||² + Σ_k α_k(S − α_k) / (S²(S+1))`

KL 正則項：

```
α_hat = (α − 1) · (1 − y_onehot) + 1
L_kl  = KL( Dir(α_hat) || Dir(1, ..., 1) )
```

總損失（每像素）：

```
L_pix = L_data + λ(epoch) · L_kl
```

### 4.4 `fusion/evidence_sum.py`

```
fuse_evidence_sum(alpha_s1, alpha_s2) -> alpha_fused
```

單一函式，公式即 `α_s1 + α_s2 − 1`。前置條件（`α ≥ 1`）**只在單元測試裡驗證**，不做 defensive runtime 檢查（符合 flood-uncertainty「不對內部邊界加防禦」的慣例）。

---

## 5. TerraMind 骨幹網路引入

### 5.1 依賴

`pyproject.toml`：

```
[project]
dependencies = [
    ...,
    "terratorch >= <對齊 rsfm 版本>",
]
```

用 `uv sync --python 3.10` 重建 `.venv` 與 `uv.lock`。

### 5.2 `flood_uncertainty/models/backbones/terramind_backbone.py`

`nn.Module` 包裝，**不繼承任何 terratorch 類別**。介面：

```
class TerraMindBackbone(nn.Module):
    def __init__(self,
                 variant: str = "terramind_v1_base",
                 modalities: tuple[str, ...] = ("S2L2A",),
                 pretrained: bool = True,
                 select_layers: tuple[int, ...] = (2, 5, 8, 11),
                 local_ckpt_path: str | None = None): ...

    def forward(self, image_dict: dict[str, Tensor]) -> list[Tensor]:
        """
        input : {"S2L2A": (B, C, H, W)}  或  {"S1GRD": (B, 2, H, W)}
        output: 4 層 pyramid feature，供 UNet decoder 使用
        """
```

內部流程：`terratorch backbone factory → SelectIndices → ReshapeTokensToImage → LearnedInterpolateToPyramidal`。這些子步驟在 rsfm 是 registry-based neck，在此**改寫成純 `nn.Module`**，不進註冊器。

### 5.3 離線權重載入

在無網路環境（例如訓練節點），支援：

```
local_ckpt_path = config.model_params.backbone_local_ckpt
```

有給就跳過 HuggingFace 下載；沒給就走 `pretrained=True` 的預設路徑。此接口在 M2 就要定義完成，避免後續整合階段才發現卡住。

### 5.4 雙獨立骨幹（**不共用權重**）

`BEF_ML4FloodsModel` 內部建**兩顆獨立** `TerraMindBackbone`：

- `backbone_s1`：`modalities=("S1GRD",)`
- `backbone_s2`：`modalities=("S2L2A",)`

**理由**：BEF 融合公式的前提是「兩個模型對 D₁、D₂ 條件獨立」。共用骨幹會破壞獨立假設，使 `α_f = α_s1 + α_s2 − 1` 不再對應貝氏後驗。此點在 `tests/test_bef_smoke.py` 內以 `assert model.backbone_s1 is not model.backbone_s2` 明確守護。

顯存代價：兩顆 TerraMind base 並存，訓練時預設走 `precision: bf16-mixed`、`batch_size` 減半，於 `bef.json` 內先寫保守值。

---

## 6. 模型類別合約：`models/bef.py`

（僅列合約與資料流，實作留待實作階段。）

```
class BEF_ML4FloodsModel(pl.LightningModule):
    # -------- I/O --------
    batch input:
        image_s1: (B, 2, H, W)          # VV, VH
        image_s2: (B, C_s2, H, W)       # 依 channel_configuration，預設 bgriswirs=6
        mask:     (B, 2, H, W)          # (cloud task, water task)，值 ∈ {0=invalid, 1=neg, 2=pos}

    # -------- 子模組 --------
    backbone_s1  : TerraMindBackbone(modalities=("S1GRD",))
    backbone_s2  : TerraMindBackbone(modalities=("S2L2A",))
    decoder_s1   : UNetDecoder(channels=[512, 256, 128, 64])
    decoder_s2   : UNetDecoder(channels=[512, 256, 128, 64])
    head_water_s1: Conv1x1 → K=2      # water evidence
    head_water_s2: Conv1x1 → K=2
    head_cloud   : 沿用現行 sigmoid 頭，僅接於 S2 塔的較早期特徵（避免與 water EDL head 共用最後一層）

    # -------- forward --------
    forward(batch) -> {
        "alpha_water_s1"   : (B, 2, H, W),
        "alpha_water_s2"   : (B, 2, H, W),
        "alpha_water_fused": (B, 2, H, W),   # fuse_evidence_sum(...)
        "logits_cloud"     : (B, 2, H, W),
    }

    # -------- training_step --------
    L =   w_s1    · edl_seg_loss(logits_water_s1,    y_water, epoch, ...)
        + w_s2    · edl_seg_loss(logits_water_s2,    y_water, epoch, ...)
        + w_fused · edl_seg_loss(logits_water_fused, y_water, epoch, ...)
        + w_cloud · BCE_masked(logits_cloud, y_cloud)

    # -------- validation_step --------
    只跑 fused 分支，記錄 IoU / F1 / vacuity mean

    # -------- test_step / predict_step --------
    輸出 α_fused + vacuity + aleatoric + epistemic
    寫檔格式沿用既有 EDL：npy / zarr，欄位對齊 metrics/segmentation.py 讀取合約
```

**標籤處理**：flood-uncertainty 的 label 使用 `{0, 1, 2}` 且 0 表無效，rsfm 用 `ignore_index=-1`。轉換僅在 `edl_seg_loss` 入口做映射（`invalid = (label == 0)`；水任務 one-hot 從 `label − 1` 取 K=2），不動上游資料層。

---

## 7. 設定檔 schema：`configurations/bef.json`

保留 `shared / train / infer / validate_only` 四段。`shared.model_params.hyperparameters` 新增欄位：

| 欄位 | 型別 | 用途 |
|---|---|---|
| `bef_fusion_rule` | `"evidence_sum" \| "prob_avg" \| "logit_avg"` | 對照 rsfm 三種融合，預設 `evidence_sum` |
| `evidence_activation` | `"softplus" \| "relu" \| "exp"` | Logits → evidence 的轉換 |
| `edl_variant` | `"digamma" \| "log" \| "mse"` | 資料項損失公式 |
| `annealing_mode` | `"linear" \| "fixed"` | 兩專案 knob 合流 |
| `annealing_step` | `int` | linear 時的 T，預設 10 |
| `annealing_coefficient` | `float` | fixed 時的 λ，預設 1.0 |
| `branch_weights` | `{s1: float, s2: float, fused: float, cloud: float}` | 訓練時各分支損失權重 |
| `backbone` | `"terramind_v1_base"` | TerraMind 變體 |
| `backbone_pretrained` | `bool` | 是否走 HF 下載 |
| `backbone_local_ckpt` | `str \| null` | 離線權重路徑（無網路時使用） |
| `select_indices` | `list[int]` | encoder 層抽取，預設 `[2, 5, 8, 11]` |
| `s1_ckpt_path` / `s2_ckpt_path` | `str \| null` | 若採 rsfm 兩顆單模態 ckpt 融合模式（僅 infer 時） |

`shared.data_params` 新增：

- `s1_input_folder`：與既有 `input_folder: "S2"` 並列
- `s1_channel_configuration`：例如 `"vv_vh"`
- `pair_alignment_strategy`：`"filename_match"`（依 sen1floods11 命名對齊）

---

## 8. 資料管線改動（S1 loader）

### 8.1 現況

既有 `EDL_SAR_Unet` 走「S1 作為單一 `input_folder`」的替代式資料集，**無法同時吐出** S1 與 S2。BEF 需要成對輸入。

### 8.2 兩階段方案

- **階段 A（快）**：新增 `flood_uncertainty/data/paired_s1s2_dataset.py`，包裝兩份 `WorldFloodsDataset`，用檔名對齊（sen1floods11 命名有 pair 對應規則）。
- **階段 B（穩）**：擴展 `filter_windows` 也對 S1 影像圖磚生效，避免出現 S1 缺對應 tile 的空窗。

### 8.3 對齊驗證

`tests/test_paired_dataset.py`：抽 8 個 tile，驗證：

- S1 與 S2 影像的地理範圍、shape、dtype 對齊
- `mask` 只讀一次（避免 S1、S2 side 各取自不同 mask 目錄造成標籤分歧）
- `filter_windows` 應用後 S1/S2/mask 三方筆數一致

---

## 9. Script 對接點

### 9.1 `scripts/train/train_bef.py`

以 `train_edl.py` 為樣板複製，僅換：

- dataset：改用 `PairedS1S2Dataset`
- model：`EDL_ML4FloodsModel(...)` → `BEF_ML4FloodsModel(...)`
- Trainer 參數（callbacks、logger、optimizer）不動

保留 CLI：`--config`、`--mode {train, validate_only}`、`--resume_ckpt`、`--data_root`。使用者體驗與其他模型一致。

### 9.2 `scripts/inference/run_inference.py`

```
--model_type {EDL, v2, EDL-SAR, BEF}
```

`load_model` 內：

```
elif model_type == "BEF":
    model = BEF_ML4FloodsModel(config.model_params)
```

`load_inference_function` 新增 `bef_predict(batch)`，回傳欄位：

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

欄位命名對齊現行 EDL 輸出，讓 `metrics/segmentation.py` 與 `analysis/` 系列腳本可零改動讀取。

### 9.3 `scripts/eval/eval_metrics.py`

`--model_type` 新增 `BEF`；接手 vacuity / aleatoric / epistemic 三張 uncertainty map，產出 retention curve、PAvPU、epistemic FP/FN 對照圖。這是 BEF 方法論的評估重點。

### 9.4 `scripts/smoke/smoke_train_bef.sh`

樣板：

```
SMOKE_TRAIN_MODEL=BEF bash scripts/smoke/smoke_train.sh
```

且需在 `smoke_train.sh` 內 dispatch `BEF → scripts/train/train_bef.py`。

---

## 10. 分階段實作里程碑

| 里程碑 | 交付內容 | 驗證方式 |
|---|---|---|
| **M1** | `losses/edl_dirichlet.py` + `fusion/evidence_sum.py` + 單元測試 | `pytest tests/test_bef_fusion.py`：對假 α 驗證公式、KL 值、退火曲線 |
| **M2** | `models/backbones/terramind_backbone.py` + CPU dummy smoke | 假輸入 → 輸出 pyramid feature shape 正確 |
| **M3** | `models/bef.py` + mock backbone smoke | `pytest tests/test_bef_smoke.py`：假資料一輪 forward + `loss.backward()` 通過；驗證兩顆 backbone 為不同物件 |
| **M4** | `data/paired_s1s2_dataset.py` + `configurations/bef.json` | 手動印 batch shape 與檔名對齊 |
| **M5** | `scripts/train/train_bef.py` + `smoke_train_bef.sh` | `SMOKE_TRAIN_MODEL=BEF bash scripts/smoke/smoke_train.sh` 跑 1 epoch |
| **M6** | `run_inference.py` + `eval_metrics.py` 分支 | 全流程 smoke：train → infer → eval → analysis |
| **M7** | Retention curve、vacuity map、與 EDL baseline 比較 | 產出到 `artifacts/results/analysis_S2/`，做圖表比較 |

M1–M3 為純 CPU、可獨立驗證的階段，屬「零外部依賴」的安全基石。M4 之後才觸及資料與訓練實境。

---

## 11. 風險登記與對應

| # | 風險 | 影響 | 對策 |
|---|---|---|---|
| 1 | `(B, 2, H, W)` mask 語意（invalid=0）與 rsfm 的 `ignore_index=-1` 不同 | Loss 計算錯誤，訓練無法收斂 | 於 `edl_seg_loss` 入口做映射，並在 M1 單元測試驗證 mask 正確被忽略 |
| 2 | 兩顆 TerraMind base 骨幹佔顯存 | 訓練 OOM | 預設 `precision: bf16-mixed`、`batch_size` 減半，並提供 gradient checkpointing 選項 |
| 3 | HuggingFace 權重下載在無網路節點失敗 | 訓練無法啟動 | `backbone_local_ckpt` 接口從 M2 就要定義並測試 |
| 4 | KL 退火兩套 mode 合流時預設值誤用 | 收斂行為與預期不符 | `bef.json` 內 `annealing_mode` 分派邏輯明確；於 M1 測試涵蓋 linear/fixed 兩條路徑 |
| 5 | 兩顆骨幹意外共享權重（違反 BEF 條件獨立假設） | 融合公式不再對應貝氏後驗 | `test_bef_smoke.py` 內 `assert backbone_s1 is not backbone_s2` 且 `id(w_s1) != id(w_s2)` |
| 6 | Cloud head 與 water EDL head 共用最後一層造成梯度混淆 | Cloud 任務性能退化 | Cloud head 從共用 decoder 前的骨幹特徵分岔，不接 water head 的最終 conv |
| 7 | sen1floods11 有 tile 缺 S1 或 S2 對應 | 訓練時 batch 錯位 | `paired_s1s2_dataset.py` 做嚴格檔名交集，並在 `test_paired_dataset.py` 統計並回報缺失比例 |
| 8 | `terratorch` 版本與 flood-uncertainty 現有依賴衝突 | `uv sync` 失敗 | 在 M2 動手前先執行 `uv add terratorch --dry-run` 檢查解析結果 |

---

## 12. 產出物清單（供 PR review 對照）

- 新增模組：`losses/edl_dirichlet.py`、`fusion/`、`models/bef.py`、`models/backbones/terramind_backbone.py`、`data/paired_s1s2_dataset.py`
- 新增設定檔：`configurations/bef.json`
- 修改 script：`scripts/inference/run_inference.py`、`scripts/eval/eval_metrics.py`；新增 `scripts/train/train_bef.py`、`scripts/smoke/smoke_train_bef.sh`
- 新增測試：`tests/test_bef_fusion.py`、`tests/test_paired_dataset.py`、`tests/test_bef_smoke.py`
- 修改 `metrics/segmentation.py`：新增 vacuity / aleatoric / epistemic 匯出欄位
- 更新 `README.md`：新增 BEF 訓練/推論/評估段落
- 更新 `pyproject.toml` + 重建 `uv.lock`
- 本文件：`docs/BEF_integration_plan.md`

---

## 13. 工作量估算

| 階段 | 估算 | 說明 |
|---|---|---|
| M1（純數學搬移 + 測試） | 1 天 | 檔案級搬移為主，只需驗證數值 |
| M2（TerraMind 包裝） | 0.5–1 天 | 主要卡在版本相容與離線權重路徑 |
| M3（BEF 模型骨架 + smoke） | 0.5 天 | 有 M1、M2 基石後主要是接線 |
| M4（S1/S2 對齊 dataset） | 0.5–1 天 | 需摸熟 sen1floods11 命名 |
| M5（訓練 script + smoke） | 0.5 天 | 複製 `train_edl.py` 為主 |
| M6（推論 + 評估分支） | 0.5 天 | 沿用既有格式 |
| M7（分析、baseline 比較） | 另計 | 依需要迭代 |

總計約 3.5–5 天的實作工時，加上 M7 的分析迭代。

---

## 14. 下一步建議

從 **M1** 動手：純數學搬移 + 測試，100% 可驗、無外部依賴。完成後 BEF 損失與融合的基石即已可信，往上疊 TerraMind 與資料層時風險大幅下降。

## 附錄 A：來源檔案指引（給實作者查證）

| 用途 | rsfm 檔案 | 對應 flood-uncertainty 目的地 |
|---|---|---|
| 證據轉換、Dirichlet 統計、KL、退火、三種變體 loss | `custom_modules/edl_losses.py` | `flood_uncertainty/losses/edl_dirichlet.py` |
| Evidence-sum 融合公式 | `custom_modules/edl_merge_task.py:130-140` | `flood_uncertainty/fusion/evidence_sum.py` |
| 多分支訓練迴圈樣式 | `custom_modules/multi_branch_task_v2.py:167-227` | `flood_uncertainty/models/bef.py:training_step` |
| Feature-level 遮罩概念 | `custom_modules/modality_mask_neck.py` | 本次不移植（採雙獨立骨幹，不需遮罩） |
| TerraMind 骨幹使用範例 | `configs/terramind_v1_base_sen1floods11_L2A_edl_merge.yaml` | `flood_uncertainty/models/backbones/terramind_backbone.py` |
| BEF 公式推導文件 | `docs/EDL-merge.md`（rsfm 側） | 本文件 §4.1 已摘錄 |
