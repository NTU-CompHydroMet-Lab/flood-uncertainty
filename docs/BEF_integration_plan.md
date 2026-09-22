# BEF 融合方法整合計畫

> 範圍：在 flood-uncertainty 專案內落地 Bayesian Evidential Framework（BEF）——
> 兩顆單模態 EDL 模型的 Dirichlet 後驗，以 uniform 先驗與條件獨立假設，
> 透過 `α_fused = α_s1 + α_s2 − 1` 融合為聯合後驗。
>
> **本文件不涉及 backbone 選型**。backbone 以介面契約方式引入（見 §4），
> 可搭配專案既有 `nn.Module` 骨幹，或搭配另一份文件描述的基礎模型骨幹，
> 兩件工作可獨立進行。

---

## 0. 名詞速查

| 術語 | 中文 | 說明 |
|---|---|---|
| Bayesian Evidential Framework (BEF) | 貝氏證據融合 | 兩顆單模態 EDL 模型輸出的 Dirichlet 濃度，在 uniform 先驗下以 `α_f = α_s1 + α_s2 − 1` 融合 |
| Evidential Deep Learning (EDL) | 證據深度學習 | 以 Dirichlet 分布建模 class posterior，輸出 α 而非 softmax 機率 |
| Vacuity | 空缺度 | `K / S`，K=類別數，S=Σα，Dempster-Shafer 意義下的整體不確定性 |
| Aleatoric / Epistemic | 隨機不確定性 / 知識不確定性 | 以全變異數定律（LoTV）拆解 |
| KL Annealing | KL 退火 | 訓練初期壓低 KL 正則權重，避免早期 α 塌至 1 |
| Sentinel-1 (S1) / Sentinel-2 (S2) | 保留原名 | 分別為 SAR 與光學遙測衛星 |

---

## 1. 決策摘要

| 決策點 | 選擇 | 理由 |
|---|---|---|
| 融合層次 | Late fusion（evidence-sum） | BEF 的本體公式即於後驗層次融合；其他變體屬延伸 |
| 資料模態 | S1 + S2 成對輸入 | BEF 的價值來自跨模態貝氏融合 |
| Task 骨架 | `pl.LightningModule` 直寫 | 對齊既有 `EDL_ML4FloodsModel` 慣例 |
| Backbone 供給 | **由呼叫端注入 `nn.Module`**，本文件不指定 | 解耦於基礎模型選型，可用既有 UNet 或後續引入的 FM 骨幹 |
| 設定檔 | 新增 `configurations/bef.json`，維持 `shared / train / infer / validate_only` 四段 | 讓 `utils/config_loader.load_mode_config` 可重用 |
| 標籤相容 | Cloud head 沿用 sigmoid，water head 走 EDL Dirichlet | 專案 mask 為 `(B, 2, H, W)` 多任務結構，BEF 僅對 water 有意義 |

> **關鍵原則**：本文件只描述數學層與訓練層。骨幹來源、預訓權重、多模態 tokenizer 等**表徵層問題**不在此文件範圍。

---

## 2. 目錄與檔案落點

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

**不會出現在本文件**：`models/backbones/*`——backbone 供給由呼叫端負責，見 §4.3。

---

## 3. 數學層（`losses/edl_dirichlet.py` + `fusion/evidence_sum.py`）

### 3.1 公式本體

在共同 uniform 先驗 `Dir(1, ..., 1)` 與條件獨立假設下：

```
α_fused_k = e_s1_k + e_s2_k + 1     （以 evidence 表示）
          = α_s1_k + α_s2_k − 1     （以 concentration 表示）
```

`k = 1..K` 為類別索引；`e_i = α_i − 1`。

### 3.2 `losses/edl_dirichlet.py` 公開介面

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

### 3.3 三種資料項損失

- `digamma`（預設）：`L_data = Σ_k y_k · (ψ(S) − ψ(α_k))`
- `log`：`L_data = Σ_k y_k · (log S − log α_k)`
- `mse`（Sensoy Eq. 5）：`L_data = ||y − P||² + Σ_k α_k(S − α_k) / (S²(S+1))`

KL 正則項：`α_hat = (α − 1)·(1 − y_onehot) + 1`，`L_kl = KL(Dir(α_hat) || Dir(1,...,1))`

總損失（每像素）：`L_pix = L_data + λ(epoch) · L_kl`

### 3.4 KL 退火：兩套 knob 合流

| 模式 | 公式 | 用途 |
|---|---|---|
| `linear` | `λ = min(1, (epoch + 1) / annealing_step)` | 訓練初期壓低 KL 權重 |
| `fixed` | `λ = annealing_coefficient` | 消融實驗、固定 KL 強度 |

由 `annealing_mode` 分派。此設計與現有 `losses/edl_loss.py`（commit `e848e5e`）的 knob 命名保持一致。

### 3.5 `fusion/evidence_sum.py`

```
fuse_evidence_sum(alpha_s1, alpha_s2) -> alpha_fused
```

單一函式，即 `α_s1 + α_s2 − 1`。前置條件（`α ≥ 1`）**僅於單元測試驗證**，不做 runtime 檢查，遵守 flood-uncertainty「不對內部函式邊界加防禦」的慣例。

---

## 4. 模型合約：`models/bef.py`

### 4.1 類別簽名

```
class BEF_ML4FloodsModel(pl.LightningModule):
    def __init__(self,
                 model_params,               # 沿用現有 config schema
                 backbone_s1: nn.Module,     # 由呼叫端注入
                 backbone_s2: nn.Module,     # 由呼叫端注入
                 ...): ...
```

**兩顆 backbone 為建構參數注入**，本模組不知道也不關心它們是哪種骨幹。這是 BEF 與骨幹選型解耦的關鍵設計。

### 4.2 I/O 合約

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

### 4.3 Backbone 介面契約（本文件唯一與外部工程握手的介面）

呼叫端注入的 `backbone_s{i}` 必須符合：

- **輸入**：`dict[str, Tensor]`，key 為模態名稱（`"S1GRD"` 或 `"S2L2A"`），value shape `(B, C, H, W)`
- **輸出**：`list[Tensor]`，4 層 pyramid features，channel 數與 spatial resolution 由 backbone 自定
- **無狀態**：呼叫端負責 `.eval()` / `.train()` 切換與權重載入

BEF 模型內部隨後接：`decoder_s{i}: UNetDecoder(...)` → `head_water_s{i}: Conv1x1 → K=2`。UNet decoder 的實作可沿用專案既有 `ml4floods` 骨幹的 decoder，或使用簡單 4 層上採樣 conv，本文件不限定。

### 4.4 訓練損失

```
L =   w_s1    · edl_seg_loss(logits_water_s1,    y_water, epoch, ...)
    + w_s2    · edl_seg_loss(logits_water_s2,    y_water, epoch, ...)
    + w_fused · edl_seg_loss(logits_water_fused, y_water, epoch, ...)
    + w_cloud · BCE_masked(logits_cloud, y_cloud)
```

三支 water 損失並行為「多分支訓練」，讓單模態塔在融合前先各自學到穩定 α。權重 `w_*` 由設定檔 `branch_weights` 給定。

### 4.5 標籤對齊

專案 label 使用 `{0, 1, 2}` 且 `0` 表無效。轉換於 `edl_seg_loss` 入口處理：`invalid = (label == 0)`，water 任務 one-hot 從 `label − 1` 取 K=2。**不動上游資料層**。

---

## 5. 設定檔：`configurations/bef.json`

維持 `shared / train / infer / validate_only` 四段。`shared.model_params.hyperparameters` 新增：

| 欄位 | 型別 | 用途 |
|---|---|---|
| `bef_fusion_rule` | `"evidence_sum" \| "prob_avg" \| "logit_avg"` | 融合規則選擇 |
| `evidence_activation` | `"softplus" \| "relu" \| "exp"` | Logits → evidence 轉換 |
| `edl_variant` | `"digamma" \| "log" \| "mse"` | 資料項損失公式 |
| `annealing_mode` | `"linear" \| "fixed"` | KL 退火模式 |
| `annealing_step` | `int` | linear 時 T，預設 10 |
| `annealing_coefficient` | `float` | fixed 時 λ，預設 1.0 |
| `branch_weights` | `{s1, s2, fused, cloud: float}` | 各分支損失權重 |
| `s1_ckpt_path` / `s2_ckpt_path` | `str \| null` | 若採兩顆單模態 ckpt 離線融合（僅 infer 時） |

**不含**任何 backbone-specific 欄位。骨幹相關 hyperparameter 由骨幹來源文件負責定義。

`shared.data_params` 新增：

- `s1_input_folder`（與既有 `input_folder: "S2"` 並列）
- `s1_channel_configuration`（例如 `"vv_vh"`）
- `pair_alignment_strategy`（例如 `"filename_match"`）

---

## 6. 資料管線：`data/paired_s1s2_dataset.py`

### 6.1 需求

BEF 需成對 S1 + S2 tile。既有 `WorldFloodsDataset` 只吃單一 `input_folder`。

### 6.2 兩階段方案

- **階段 A**：包裝兩份 `WorldFloodsDataset`，用檔名對齊（sen1floods11 命名有 pair 對應）。
- **階段 B**：擴展 `filter_windows` 亦作用於 S1 影像圖磚，避免 tile 缺失造成空窗。

### 6.3 對齊驗證（`tests/test_paired_dataset.py`）

- S1 與 S2 影像地理範圍、shape、dtype 一致
- `mask` 只讀一次（不從兩份 side 各取）
- `filter_windows` 應用後 S1/S2/mask 三方筆數一致

---

## 7. Script 對接點

### 7.1 `scripts/train/train_bef.py`

以 `train_edl.py` 為樣板複製，僅換：

- dataset 呼叫改用 `PairedS1S2Dataset`
- model 換成 `BEF_ML4FloodsModel(model_params, backbone_s1=..., backbone_s2=...)`
- **backbone 來源在 script 內組裝**：可從專案既有 `get_model()` 產出，或呼叫另一份文件描述的 backbone factory；本文件不指定

CLI 保留：`--config / --mode / --resume_ckpt / --data_root`。

### 7.2 `scripts/inference/run_inference.py`

```
--model_type {EDL, v2, EDL-SAR, BEF}
```

`load_model` 內新增 BEF 分支；`load_inference_function` 新增 `bef_predict(batch)` 回傳：

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

欄位命名對齊現有 EDL 輸出，讓 `metrics/segmentation.py` 與 `analysis/` 系列腳本零改動。

### 7.3 `scripts/eval/eval_metrics.py`

`--model_type` 新增 `BEF`，接手 vacuity / aleatoric / epistemic 三張圖，產出 retention curve、PAvPU、epistemic FP/FN 對照圖。

---

## 8. 分階段里程碑

| 里程碑 | 交付內容 | 驗證方式 |
|---|---|---|
| **M1** | `losses/edl_dirichlet.py` + `fusion/evidence_sum.py` + 單元測試 | `pytest tests/test_bef_fusion.py`：假 α，驗證公式、KL、退火 |
| **M2** | `models/bef.py` + mock backbone smoke | `pytest tests/test_bef_smoke.py`：假資料一輪 forward + `loss.backward()`，斷言兩顆 backbone 為不同物件 |
| **M3** | `data/paired_s1s2_dataset.py` + `configurations/bef.json` | 手動印 batch shape、確認檔名對齊 |
| **M4** | `scripts/train/train_bef.py` + `smoke_train_bef.sh` | 用**任意可用骨幹**跑 1 epoch smoke |
| **M5** | `run_inference.py` + `eval_metrics.py` BEF 分支 | 全流程 smoke：train → infer → eval |
| **M6** | 與 baseline（EDL、v2）之 retention curve 比較 | 產出到 `artifacts/results/analysis_S2/` |

M1–M2 為純 CPU、無外部依賴的安全基石；M3 之後涉及資料層與訓練實境。

**里程碑不依賴骨幹選型**：M1–M3 可用 mock backbone（例如三層 Conv）完成；M4 之後可用專案既有骨幹或後續引入的基礎模型骨幹。

---

## 9. 風險登記

| # | 風險 | 對策 |
|---|---|---|
| 1 | `(B, 2, H, W)` mask 語意（invalid=0）與 EDL 論文的 `ignore_index=-1` 不同 | 於 `edl_seg_loss` 入口做映射，M1 單元測試涵蓋 |
| 2 | 兩顆骨幹意外共享權重（違反 BEF 條件獨立假設） | `test_bef_smoke.py` 斷言 `backbone_s1 is not backbone_s2` 且 `id(w_s1) != id(w_s2)` |
| 3 | Cloud head 與 water EDL head 共用最後一層造成梯度混淆 | Cloud head 從共用 decoder 前分岔，不接 water head 最終 conv |
| 4 | KL 退火兩套 mode 預設值誤用 | 於 M1 測試涵蓋 linear/fixed 兩條路徑 |
| 5 | sen1floods11 有 tile 缺 S1 或 S2 對應 | `paired_s1s2_dataset.py` 做嚴格檔名交集，測試回報缺失比例 |

**不在本文件風險範圍**：骨幹權重下載失敗、顯存爆量、基礎模型版本相容性——這些由骨幹來源文件負責。

---

## 10. 產出物清單

- 新增：`losses/edl_dirichlet.py`、`fusion/`、`models/bef.py`、`data/paired_s1s2_dataset.py`
- 新增：`configurations/bef.json`
- 修改：`scripts/inference/run_inference.py`、`scripts/eval/eval_metrics.py`
- 新增：`scripts/train/train_bef.py`、`scripts/smoke/smoke_train_bef.sh`
- 新增：`tests/test_bef_fusion.py`、`tests/test_paired_dataset.py`、`tests/test_bef_smoke.py`
- 修改：`metrics/segmentation.py`（新增 vacuity / aleatoric / epistemic 匯出欄位）
- 更新：`README.md`（BEF 訓練/推論/評估段落）

**不在此清單**：骨幹相關模組、骨幹相關依賴（`pyproject.toml`）、骨幹相關設定欄位。

---

## 11. 工作量估算

| 階段 | 估算 | 說明 |
|---|---|---|
| M1（純數學） | 1 天 | 檔案級搬移為主，只需驗證數值 |
| M2（模型骨架 + smoke） | 0.5 天 | 有 M1 基石後主要是接線 |
| M3（S1/S2 對齊 dataset） | 0.5–1 天 | 需摸熟 sen1floods11 命名 |
| M4（訓練 script + smoke） | 0.5 天 | 複製 `train_edl.py` 為主 |
| M5（推論 + 評估分支） | 0.5 天 | 沿用既有格式 |
| M6（分析、baseline 比較） | 另計 | 依需要迭代 |

總計約 3–3.5 天 + M6 迭代。

---

## 12. 與骨幹來源之握手

本文件與外部工程的唯一介面：**§4.3 backbone 介面契約**。

只要注入的 `backbone_s{i}` 滿足「輸入 `dict[str, Tensor]`、輸出 pyramid feature `list[Tensor]`」，BEF 模型即可運作。這讓 BEF 實作可獨立於任何特定骨幹選型（包含專案既有 UNet、後續引入的基礎模型骨幹、或未來替換的任何 `nn.Module`）先行完成。

M1–M3 可用最簡三層 Conv mock backbone 完成；M4 起才需要真正骨幹，此時只需在 `train_bef.py` 內把想用的骨幹「注入建構參數」，其餘完全不變。
