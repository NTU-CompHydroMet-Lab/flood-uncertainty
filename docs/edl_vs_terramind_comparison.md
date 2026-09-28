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

三組皆完成（2026-09-29 07:41）。表格由 `artifacts/runs/compare/summarize.py` 產生。
ckpt：A `epoch=8-step=9225`（lab 指定）；B `epoch=3-step=16396`、C `epoch=21-step=90178`（各自 best `val_bce_land_water`）。

### 3.1 Segmentation（val / test）

**val OVERALL**（pixel 加總）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9666 | 0.8890 | 0.9816 | 0.9330 | 0.8745 |
| B. TerraMind frozen | 0.9581 | 0.8594 | 0.9838 | 0.9174 | 0.8474 |
| C. frozen + WF2 decoder | 0.9610 | 0.8692 | 0.9832 | 0.9227 | 0.8564 |

**val AVERAGE**（per-event 平均）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9623 | 0.8487 | 0.9315 | 0.8831 | 0.8016 |
| B. TerraMind frozen | 0.9536 | 0.8074 | 0.9301 | 0.8576 | 0.7631 |
| C. frozen + WF2 decoder | 0.9557 | 0.8258 | 0.9233 | 0.8646 | 0.7752 |

**test OVERALL**（pixel 加總）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9418 | 0.8894 | 0.9236 | 0.9062 | 0.8285 |
| B. TerraMind frozen | 0.9382 | 0.8915 | 0.9074 | 0.8994 | 0.8171 |
| C. frozen + WF2 decoder | 0.9231 | 0.8386 | 0.9258 | 0.8800 | 0.7857 |

**test AVERAGE**（per-event 平均）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9586 | 0.9089 | 0.9050 | 0.9035 | 0.8303 |
| B. TerraMind frozen | 0.9544 | 0.8956 | 0.9055 | 0.8984 | 0.8211 |
| C. frozen + WF2 decoder | 0.9438 | 0.8415 | 0.9301 | 0.8818 | 0.7946 |


### 3.2 Uncertainty（retention curve / PAvPU）

**PAvPU（OVERALL，patch 3×3，retention target 0.9）**

| model | subset | DST | Aleatoric | Epistemic | A+E |
|---|---|---|---|---|---|
| A. EDL baseline | val | 0.9010 | 0.9012 | 0.9008 | 0.9011 |
| A. EDL baseline | test | 0.8394 | 0.8393 | 0.8395 | 0.8394 |
| B. TerraMind frozen | val | 0.9005 | 0.9003 | 0.9008 | 0.9005 |
| B. TerraMind frozen | test | 0.8479 | 0.8482 | 0.8482 | 0.8482 |
| C. frozen + WF2 decoder | val | 0.9015 | 0.9014 | 0.9016 | 0.9014 |
| C. frozen + WF2 decoder | test | 0.8276 | 0.8269 | 0.8278 | 0.8273 |

**Retention curve：IoU-vs-retention 曲線下面積（normalized AUC）/ retention 50% 時的 IoU**

| model | subset | DST | Aleatoric | Epistemic | A+E |
|---|---|---|---|---|---|
| A. EDL baseline | val | 0.9909 / 0.9976 | 0.9908 / 0.9976 | 0.9909 / 0.9976 | 0.9909 / 0.9976 |
| A. EDL baseline | test | 0.9688 / 0.9828 | 0.9688 / 0.9828 | 0.9688 / 0.9828 | 0.9688 / 0.9828 |
| B. TerraMind frozen | val | 0.9872 / 0.9967 | 0.9866 / 0.9967 | 0.9870 / 0.9967 | 0.9867 / 0.9967 |
| B. TerraMind frozen | test | 0.9665 / 0.9849 | 0.9665 / 0.9849 | 0.9665 / 0.9849 | 0.9665 / 0.9849 |
| C. frozen + WF2 decoder | val | 0.9890 / 0.9973 | 0.9888 / 0.9973 | 0.9889 / 0.9973 | 0.9889 / 0.9973 |
| C. frozen + WF2 decoder | test | 0.9552 / 0.9738 | 0.9550 / 0.9738 | 0.9551 / 0.9738 | 0.9551 / 0.9738 |

### 3.3 訓練曲線（`lightning_logs/version_0/metrics.csv`，val 17 張）

訓練時的 IoU 是 256 tile 級、land/water 雙類別的 water IoU，數值與 §3.1 整圖推論不可直接比。

| model | best val_bce：epoch / val_bce / IoU water | best IoU water：epoch / IoU | final：epoch / val_bce / IoU | epoch 0 IoU |
|---|---|---|---|---|
| A. EDL baseline | 8 / 0.0621 / 0.7030 | 3 / 0.7054 | 29 / 0.0777 / 0.6737 | 0.6298 |
| B. TerraMind frozen | 3 / 0.0645 / 0.5132 | 2 / 0.5153 | 29 / 0.0765 / 0.5012 | 0.4973 |
| C. frozen + WF2 decoder | 21 / 0.0674 / 0.5154 | 20 / 0.5155 | 29 / 0.0690 / 0.5120 | 0.4802 |

### 3.4 Checkpoint epoch 對照（B、C 各取 epoch 3 與 epoch 21，只跑 inference + eval）

B 的 best ckpt 落在 epoch 3、C 落在 epoch 21，直接比會把「decoder 初始化」和「訓練多久」混在一起，所以補跑交叉組合：

| model / ckpt | val IoU | val P | val R | test IoU | test P | test R |
|---|---|---|---|---|---|---|
| A. EDL baseline (ep8) | 0.8745 | 0.8890 | 0.9816 | 0.8285 | 0.8894 | 0.9236 |
| B. frozen, ep3 (best val_bce) | 0.8474 | 0.8594 | 0.9838 | 0.8171 | 0.8915 | 0.9074 |
| B. frozen, ep21 | 0.8347 | 0.8463 | 0.9838 | 0.7999 | 0.8480 | 0.9339 |
| C. wf2dec, ep3 | 0.8421 | 0.8529 | 0.9852 | 0.8160 | 0.8867 | 0.9110 |
| C. wf2dec, ep21 (best val_bce) | 0.8564 | 0.8692 | 0.9832 | 0.7857 | 0.8386 | 0.9258 |

## 4. 結論

**A vs B（凍結 TerraMind + 隨機 decoder 能不能追上 fine-tune 過的 UNet）**

- Segmentation：B 落後 A，但差距小：val OVERALL IoU 0.847 vs 0.874（−2.7 pt），test 0.817 vs 0.828（−1.1 pt）。差距在 precision（val 0.859 vs 0.889，多報水），recall 持平。
  per-event AVERAGE 差距較大（val 0.763 vs 0.802），B 在幾個小事件（`ST1_BinhDinh_Lake` 0.73 vs 0.88、`EMSR358` 0.55 vs 0.67）明顯較差。
- Uncertainty 品質相當：PAvPU val 0.90 vs 0.90，test B 反而略高（0.848 vs 0.839）；retention AUC val 0.987 vs 0.991、test 0.967 vs 0.969。
  四種 uncertainty（DST / aleatoric / epistemic / a+e）在三個模型上的排序效果幾乎一樣，這是 EDL head 的性質，與 encoder 無關。
- 訓練時 tile 級 IoU 差 0.19（0.51 vs 0.70），整圖推論只差 0.01–0.03：tile 級 metric 對 ViT 模型特別悲觀（256 tile 的邊界效應），評估要看整圖。

**B vs C（decoder 用 WF2 預訓練有沒有幫助）**

- 同 epoch 比較（§3.4）差異在雜訊內：epoch 3 時 B 0.847/0.817、C 0.842/0.816（val/test）；epoch 21 時 B 0.835/0.800、C 0.856/0.786。
  decoder 預訓練沒有帶來一致的增益；C 訓練起點反而更低（epoch 0 IoU 0.480 vs 0.497），因為預訓練 decoder 期待的是 UNet 特徵，接上隨機 lateral 的 ViT 特徵後要重新適應。
- C 的 best-val ckpt（epoch 21）在 val 上是三組中僅次於 A 的（0.856），但 test 掉到 0.786，precision 全面下降，不是單一事件造成；B 的 epoch 21 也一樣（test 0.800）。
  **凍結 encoder 的兩組都在後期 epoch 出現 val 進步、test 退步**，用 17 張 val 的 `val_bce` 挑 ckpt 會偏向晚期 ckpt 而在 test 吃虧。建議凍結 encoder 時取早期 ckpt（epoch 2–4），或加大 val set。
- 所以「B 落後 A 的主因是 decoder 初始化」這個假設**不成立**；主因是 encoder 凍結（特徵不能針對水體微調），以及 A 的 encoder+decoder 是整體從 WF2 起跑。

**整體**

- 完全不微調 TerraMind encoder、decoder 從零訓練，就能達到 fine-tune 過的 UNet baseline 97–99% 的整圖 IoU，uncertainty 品質相同。TerraMind 特徵對水體分割是可用的，但凍結狀態下略遜於任務專用 UNet。
- 要超越 baseline 需要解凍 encoder（`configurations/edl_terramind.json` 預設的 D 組，`backbone_freeze: false`、`backbone_lr_mult 0.1`），建議 batch 拉到 64（顯存 3.5 GB / 10 GB 有餘裕）並保留 WF2 decoder 初始化與否各一組。

## 5. 執行紀錄與注意事項

- 兩次訓練（B、C）都遇到**另一台機器對同一 experiment 目錄啟動同一 config** 的情況（B：22:00–23:00；C：00:04–03:18），造成 `-v1.ckpt` / `lightning_logs/version_1` / 第二個 wandb run 與 stdout log 混寫。已依 `-v1` 後綴與 mtime 分辨歸屬，把另一台的檔案移到 `artifacts/runs/duplicate_run_220013/`、`artifacts/runs/duplicate_run_000449/`；本文件所有數字皆來自本機的 run。
  同一 NAS 上多機訓練，`experiment_name` 必須不同。
- `analysis_S2.py` / `compute_pavpu.py` 的輸出路徑只有 model_type 沒有 subset，val 與 test 會互相覆蓋；本次用 `analysis_S2/{val,test}/<model_type>/` 分開。
- 推論 / 分析腳本與產生表格的程式在 `artifacts/runs/compare/`（`run_pipeline.sh`、`ckpt_control.sh`、`summarize.py`、`curves.py`），未納入 git。
