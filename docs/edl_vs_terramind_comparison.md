# EDL vs EDL + TerraMind encoder：實驗比較

分支 `feat/terramind-edl-backbone`（PR #1）。三組模型共用 EDL head / loss / 資料 / 訓練超參數，只差 encoder 與初始權重。
實作與設計理由見 `terramind_edl_integration_log.md`。

## 1. 實驗模型

| | A. EDL baseline | B. EDL + TerraMind (frozen) | C. EDL + TerraMind (frozen) + WF2 decoder |
|---|---|---|---|
| **模型類別** | `EDL_ML4FloodsModel` | `EDL_TerraMind_ML4FloodsModel` | 同 B |
| **Encoder** | UNet encoder（`dconv_down1–4`） | TerraMind v1 base，ViT-B/16，12 層，取第 2/5/8/11 層 → 1×1 lateral → stride 4/8/16/32 金字塔 | 同 B |
| **Encoder 權重** | WF2 UNet 預訓練，**可訓練** | TerraMind 預訓練（HF `ibm-esa-geospatial/TerraMind-1.0-base`），**凍結** | 同 B |
| **Decoder** | UNet decoder（`dconv_up3/2/1` + `conv_last`） | 同結構 + `dconv_full`（stride 4 → 全解析度） | 同 B |
| **Decoder 權重** | WF2 預訓練 | 隨機 | `dconv_up3/2/1` 載入 WF2（12/30 tensors）；`lateral`、`dconv_full`、`conv_last` 隨機 |
| **輸入** | 6 波段 `bgriswirs`（B2/B3/B4/B8/B11/B12），ml4floods z-score | 同，再反正規化成 TerraMind 的 S2L2A mean/std | 同 B |
| **可訓練參數** | 全部 | decoder + lateral + head | 同 B |
| **batch / steps per epoch** | 64 / 1025 | 16 / 4099 | 16 / 4099 |
| **Config** | `configurations/edl.json` | `artifacts/runs/edl_terramind_freeze.json` | `artifacts/runs/edl_terramind_freeze_wf2dec.json` |
| **實驗名稱 / 權重目錄** | `edl_EpochKL_20260819`（chlunchen NAS，symlink 到 `artifacts/models/`） | `edl_terramind_v1_base_bgriswirs_freeze` | `edl_terramind_v1_base_bgriswirs_freeze_wf2dec` |
| **選用 ckpt** | `epoch=8-step=9225`（lab 指定 trusted ckpt） | best `val_bce_land_water` | best `val_bce_land_water` |

`artifacts/runs/*.json` 由 `configurations/edl_terramind.json` 複製，只改 `experiment_name`、`backbone_freeze`、`pretrained_path`；
`configurations/edl_terramind.json` 本身（`backbone_freeze: false`）是可選的第四組 D，回答「TerraMind encoder 微調後能否超過 UNet」。

**共同條件**：WorldFloods v2（train 475 / val 17 / test 18 張 S2；每 epoch 約 65,600 個 window），`max_tile_size 256`、
`filter_windows` 同（`threshold_clouds 0.8`）、Adam `lr 1e-4`、ReduceLROnPlateau（factor 0.5 / patience 2）on `val_bce_land_water`、
30 epoch、`early_stopping_patience 50`（實際跑滿）、`pos_weight` / `weight_problem` 同、KL annealing linear / coef 0.5 / step 10。

**三組回答的問題**

- A vs B：凍結的 foundation-model 特徵 + 從零學的 decoder，追不追得上 fine-tune 過的 UNet。
- B vs C：decoder 初始化是不是 B 落後的主因（單變因對照）。
- A vs C：encoder 換成凍結 TerraMind、其餘盡量對齊後的真實差距。

## 2. 評估項目

每組用同一條 pipeline（`artifacts/runs/compare/run_pipeline.sh <model_type> <config> <ckpt>`）：

1. `scripts/inference/run_inference.py` → `artifacts/results/val_test_inference/{val,test}/<model_type>/`
2. `scripts/eval/eval_metrics.py --save_tif` → `metrics_<model_type>.csv`（per-event + OVERALL pixel 加總 + AVERAGE per-event 平均）與 `cm_*.tif`
3. `scripts/analysis/analysis_S2.py` → retention curve（DST / aleatoric / epistemic / a+e）
4. `scripts/analysis/compute_pavpu.py` → PAvPU

加上訓練曲線（`lightning_logs/version_0/metrics.csv` 的 `val_bce_land_water`、`val_iou_land_water water`）。

B、C 的 model_type 都是 `EDL-TERRAMIND`，輸出目錄依 model_type 命名，所以 B 跑完後其輸出改名為 `EDL-TERRAMIND_freeze`，再跑 C。

## 3. 結果

（待補：B 訓練 2026-09-29 00:20 結束，C 之後啟動。）

### 3.1 Segmentation（val / test）

### 3.2 Uncertainty（retention curve / PAvPU）

### 3.3 訓練曲線

## 4. 結論

（待補）
