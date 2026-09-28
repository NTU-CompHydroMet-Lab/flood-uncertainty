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

A、B 已完成（2026-09-29 01:26）。C 訓練中（預計 05:00 結束），數字待補。
表格由 `artifacts/runs/compare/summarize.py` 產生；ckpt：A `epoch=8-step=9225`，B `epoch=3-step=16396`（best `val_bce_land_water`）。

### 3.1 Segmentation（val / test）

**val OVERALL**（pixel 加總）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9666 | 0.8890 | 0.9816 | 0.9330 | 0.8745 |
| B. TerraMind frozen | 0.9581 | 0.8594 | 0.9838 | 0.9174 | 0.8474 |

**val AVERAGE**（per-event 平均）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9623 | 0.8487 | 0.9315 | 0.8831 | 0.8016 |
| B. TerraMind frozen | 0.9536 | 0.8074 | 0.9301 | 0.8576 | 0.7631 |

**test OVERALL**（pixel 加總）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9418 | 0.8894 | 0.9236 | 0.9062 | 0.8285 |
| B. TerraMind frozen | 0.9382 | 0.8915 | 0.9074 | 0.8994 | 0.8171 |

**test AVERAGE**（per-event 平均）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9586 | 0.9089 | 0.9050 | 0.9035 | 0.8303 |
| B. TerraMind frozen | 0.9544 | 0.8956 | 0.9055 | 0.8984 | 0.8211 |


### 3.2 Uncertainty（retention curve / PAvPU）

**PAvPU（OVERALL，patch 3×3，retention target 0.9）**

| model | subset | DST | Aleatoric | Epistemic | A+E |
|---|---|---|---|---|---|
| A. EDL baseline | val | 0.9010 | 0.9012 | 0.9008 | 0.9011 |
| A. EDL baseline | test | 0.8394 | 0.8393 | 0.8395 | 0.8394 |
| B. TerraMind frozen | val | 0.9005 | 0.9003 | 0.9008 | 0.9005 |
| B. TerraMind frozen | test | 0.8479 | 0.8482 | 0.8482 | 0.8482 |

**Retention curve：IoU-vs-retention 曲線下面積（normalized AUC）/ retention 50% 時的 IoU**

| model | subset | DST | Aleatoric | Epistemic | A+E |
|---|---|---|---|---|---|
| A. EDL baseline | val | 0.9909 / 0.9976 | 0.9908 / 0.9976 | 0.9909 / 0.9976 | 0.9909 / 0.9976 |
| A. EDL baseline | test | 0.9688 / 0.9828 | 0.9688 / 0.9828 | 0.9688 / 0.9828 | 0.9688 / 0.9828 |
| B. TerraMind frozen | val | 0.9872 / 0.9967 | 0.9866 / 0.9967 | 0.9870 / 0.9967 | 0.9867 / 0.9967 |
| B. TerraMind frozen | test | 0.9665 / 0.9849 | 0.9665 / 0.9849 | 0.9665 / 0.9849 | 0.9665 / 0.9849 |

### 3.3 訓練曲線（`lightning_logs/version_0/metrics.csv`，val 17 張；訓練時的 IoU 是 tile 級、雙類別 water IoU，數值與 §3.1 整圖推論不可直接比）

| model | best val_bce：epoch / val_bce / IoU water | best IoU water：epoch / IoU | final：epoch / val_bce / IoU | epoch 0 IoU |
|---|---|---|---|---|
| A. EDL baseline | 8 / 0.0621 / 0.7030 | 3 / 0.7054 | 29 / 0.0777 / 0.6737 | 0.6298 |
| B. TerraMind frozen | 3 / 0.0645 / 0.5132 | 2 / 0.5153 | 29 / 0.0765 / 0.5012 | 0.4973 |
| C. frozen + WF2 decoder | 待補 | | | |

## 4. 初步結論（A vs B，C 待補）

- **Segmentation**：B 落後 A，但差距不大：val OVERALL IoU 0.847 vs 0.874（−2.7 pt），test 0.817 vs 0.828（−1.1 pt）。差距幾乎全在 precision（val 0.859 vs 0.889，多報水），recall 持平。
  per-event AVERAGE 差距較大（val 0.763 vs 0.802），表示 B 在少數小事件上錯得比較多。
- **Uncertainty 品質**：兩者相當。PAvPU val 0.90 vs 0.90，test B 反而略高（0.848 vs 0.839）；retention AUC val 0.987 vs 0.991、test 0.967 vs 0.969；retention 50% 時 test IoU B 略高（0.985 vs 0.983）。
  四種 uncertainty（DST / aleatoric / epistemic / a+e）在兩個模型上的排序效果幾乎一樣，這是 EDL head 的性質，與 encoder 無關。
- **訓練行為**：B 在 epoch 2–3 就到平台（訓練時 tile 級 IoU 0.51，A 是 0.70），之後 27 個 epoch 沒有進步；encoder 凍結時 decoder 容量很快用完。
  訓練時 IoU 差 0.19 但整圖推論只差 0.01–0.03，代表 tile 級 metric 對 B 特別悲觀（可能與 256 tile 的 ViT 邊界效應有關，整圖推論時 padding/裁切後影響變小）。
- **意義**：完全不微調 TerraMind encoder、decoder 從零訓練，就能達到 fine-tune 過的 UNet baseline 97–99% 的 IoU 與同等的 uncertainty 品質。要超越 baseline 需要解凍 encoder（可選的 D 組）。
