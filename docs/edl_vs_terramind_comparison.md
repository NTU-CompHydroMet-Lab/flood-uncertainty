# EDL vs EDL + TerraMind encoder：實驗比較

分支 `feat/terramind-edl-backbone`（PR #1）。各組共用 EDL head / loss / 資料 / 訓練超參數，只差 encoder 與輸入模態。
實作與設計理由見 `terramind_edl_integration_log.md`。

**實驗方向**：TerraMind encoder **全部解凍**微調，以 `tm_l1c13` 為 TerraMind 主線的基準，每個新實驗只對它改一個變因；
`unet_baseline` 是要超越的目標。評估有兩個測試集：WorldFloods v2 test（同資料集）與 Sen1Floods11（跨資料集，所有模型都沒在上面訓練）。
凍結 encoder 的結果移到附錄 A；`tm_frozen_l2a6_wf2dec` 與 `tm_l2a6_wf2dec` 已回答「WF2 decoder 初始化沒有一致的幫助」，
不再列入比較（數字見 commit `e9cc0ba` 的本文件）。

## 0. 摘要

- **WorldFloods test（同資料集）**：所有 TerraMind 設定都**低於** `unet_baseline` 1–3 pt。最接近的是 `tm_l1c13_tim`（OVERALL −1.1 pt [−3.3, −0.2]）。
- **Sen1Floods11（跨資料集）**：TerraMind 設定普遍**高於** `unet_baseline`。`tm_l1c13_tim` 在 Sen1Floods11 test 上 +3.9 pt [+0.8, +7.5]，
  是第一個在統計上顯著勝過 baseline 的設定；`tm_l2a6` 與 `tm_l1c13_inskip` 也高 3–4 pt，但信賴區間跨過 0。
- 推測：`unet_baseline` 的 encoder + decoder 都從 WF2（在 WorldFloods 上訓練的 UNet）起跑，在同資料集上有主場優勢；換到沒看過的資料時，通用預訓練的 TerraMind 泛化較好。
- 相對 `tm_l1c13`：TiM（`tm_l1c13_tim`）與全解析度 input skip（`tm_l1c13_inskip`）在 Sen1Floods11 上各帶來 +2.8 / +2.3 pt（皆顯著），在 WorldFloods 上差異不顯著。
- Uncertainty：同資料集與 Sen1Floods11 test 上，`unet_baseline` 的 retention AUC 仍是最高或並列最高；TerraMind 預測較準，但「知道自己哪裡不準」的能力沒有更好。

## 1. 實驗組合

**命名規則**：`<encoder>[_frozen]_<輸入>[_<變因>]`。`tm` = TerraMind v1 base；`l2a6` = `bgriswirs` 6 波段經 S2L2A embedding；
`l1c13` = 13 波段經 S2L1C embedding；未標 `_frozen` 即 encoder 全部解凍。指定 checkpoint 時寫成 `<名稱>@ep<epoch>`。

| 名稱 | 設定 | 單變因對照 | 狀態 |
|---|---|---|---|
| **`unet_baseline`** | ml4floods UNet（`EDL_ML4FloodsModel`），encoder + decoder 從 WF2 預訓練起跑，batch 64 | 目標 | 完成 |
| **`tm_l2a6`** | TerraMind 解凍，`bgriswirs` 6 波段 → S2L2A embedding（波段子集） | 消融：與 `tm_l1c13` 比，看 L1C 13 波段的效果 | 完成 |
| **`tm_l1c13`** | TerraMind 解凍，`all` 13 波段 → **S2L1C** embedding | TerraMind 主線基準 | 完成 |
| **`tm_l1c13_tim`** | `tm_l1c13` + TiM（從 S2L1C 生成 LULC token，argmax 解碼，生成器凍結） | `tm_l1c13` | 完成（WorldFloods ep29 評估中） |
| **`tm_l1c13_inskip`** | `tm_l1c13` + 全解析度 input skip（`decoder_input_skip_channels: 32`） | `tm_l1c13` | 完成 |

**TerraMind 各組共同設定**：`EDL_TerraMind_ML4FloodsModel`；TerraMind v1 base（ViT-B/16，HF `ibm-esa-geospatial/TerraMind-1.0-base`），
取第 2/5/8/11 層 → 1×1 lateral → stride 4/8/16/32 金字塔；decoder 同 UNet 結構 + `dconv_full`（stride 4 → 全解析度），隨機初始化；
encoder 全部可訓練，lr = 1e-4 × `backbone_lr_mult` 0.1 = 1e-5（decoder / head 1e-4）；batch 16（4099 steps / epoch）、`num_workers 16`、fp32。

**`tm_l1c13` 的動機**：WorldFloods v2 的 S2 是 **L1C**（13 波段），`tm_l2a6` 卻用 TerraMind 的 S2L2A embedding（地表反射率）並靠重新正規化對齊；
TerraMind 權重本身有 `untok_sen2l1c@224`（13 波段），`tm_l1c13` 讓模態與資料一致。這也是 `tm_l1c13_tim` 的前置條件：TiM 生成器只吃完整的預訓練模態。

**`tm_l1c13_inskip` 的動機**：ViT-B/16 最細的特徵是 stride 16；原始輸入經一個 stride 1 的 `double_conv` 後接到 `dconv_full` 前（同 UNETR 的做法），
讓最後一層 decoder 有逐像素的光譜證據，針對水體周圍 1–2 個 patch 範圍的多報水（§3.7）。約多 3.5 萬個參數。

| 名稱 | Config | 實驗目錄（`artifacts/models/`） | 結果目錄 tag |
|---|---|---|---|
| `unet_baseline` | `configurations/edl.json`（訓練於 chlunchen 機器） | `edl_EpochKL_20260819`（symlink） | `EDL` |
| `tm_l2a6` | `configurations/terramind/tm_l2a6.json` | `edl_terramind_v1_base_bgriswirs_nofreeze` | `EDL-TERRAMIND_nofreeze_ep*` |
| `tm_l1c13` | `configurations/terramind/tm_l1c13.json` | `edl_terramind_v1_base_s2l1c_all_nofreeze` | `EDL-TERRAMIND_s2l1c_ep*` |
| `tm_l1c13_tim` | `configurations/terramind/tm_l1c13_tim.json` | `edl_terramind_v1_base_s2l1c_all_nofreeze_tim_lulc` | `EDL-TERRAMIND_tim_ep*` |
| `tm_l1c13_inskip` | `configurations/terramind/tm_l1c13_inskip.json` | `edl_terramind_v1_base_s2l1c_all_nofreeze_inskip32` | `EDL-TERRAMIND_inskip_ep*` |

**共同條件**：WorldFloods v2（train 475 / **val 16** / test 18 張 S2；每 epoch 約 65,600 個 window），`max_tile_size 256`、
`filter_windows`（`threshold_clouds 0.8`）、Adam、ReduceLROnPlateau（factor 0.5 / patience 2）on `val_bce_land_water`、
30 epoch（實際跑滿）、`pos_weight` / `weight_problem` 同、KL annealing linear / coef 0.5 / step 10。

## 2. 評估規則與指標

**Checkpoint 規則（看 test 之前就固定）**：每個 run 報兩個 ckpt，**結論只用第一個**：

1. **best val**：`val_bce_land_water` 最低的 ckpt。
2. **last**：最後一個 epoch（ep29），只作補充，不用來下結論。

`unet_baseline` 用 lab 指定的 `epoch=8-step=9225`，它同時也是它的 best val。挑選邏輯：`artifacts/runs/compare/pick_ckpts.py`（只讀 `metrics.csv`）。

| 名稱 | best val（結論用） | last（補充） |
|---|---|---|
| `unet_baseline` | ep8 | — |
| `tm_l2a6` | ep20 | ep29（未評估） |
| `tm_l1c13` | ep7 | ep29（只有 1024 tile 結果） |
| `tm_l1c13_tim` | ep16 | ep29 |
| `tm_l1c13_inskip` | ep13 | ep29 |

**推論 tile 尺寸**：訓練用 256×256。**所有主要結果用 256 tile 推論**（事先固定，不依結果挑選）。當時透過包裝腳本
`artifacts/runs/compare/run_inference_tile.py` 實現；現在 `run_inference.py --max_tile_size` 原生支援，
且 `EDL-TERRAMIND` 未指定時預設即為 256（其他 model_type 維持 1024）。原本的 1024 tile 結果保留為敏感度分析（§3.4）。

**Pipeline**：`artifacts/runs/compare/run_pipeline_tile.sh`：`run_inference.py`（整圖）→ `eval_metrics.py` → `analysis_S2.py`（retention）→ `compute_pavpu.py`（PAvPU）；
報表由 `report.py` / `make_reports.sh` 產生（`artifacts/runs/compare/report_*_tile256.md`），另以 `eval_val_iou.py` 算 val 的 pooled IoU（§3.6）。

**指標**

- 分割：每張影像把 EDL 機率四捨五入成水 / 非水，與 GT 比出 TP / FP / FN / TN（invalid 排除）。IoU water = TP / (TP + FP + FN)。
  **OVERALL** = 所有影像像素加總後算一次（大事件權重大）；**AVERAGE** = 每張影像各算再平均（每個事件等權重）。**兩者都要看。**
  Sen1Floods11 有許多水體很少的 chip，單張 IoU 容易接近 0，AVERAGE 的參考價值較低。
- Retention curve：依不確定性由低到高保留最確定的 X% 像素，對保留的像素算 IoU；報曲線下面積（normalized AUC）。
- PAvPU：3×3 patch 判斷「準不準」（patch IoU ≥ 門檻）與「確不確定」（平均不確定性 ≥ 門檻），PAvPU = (準且確定 + 不準且不確定) / 全部。
  `compute_pavpu.py` 的兩個門檻取自各模型自己 retention 90% 那一點。3×3 patch 的 IoU 只能取離散值（最接近 1 的是 1 與 8/9 ≈ 0.889），
  各模型自己的準確度門檻幾乎都落在 (0.889, 1]，實際規則皆為「patch 完全正確才算準」，所以**跨模型比較基本有效**；
  改用 `unet_baseline` 的門檻統一重算（`pavpu_fixed.py`），差異大多在 0.01 以內（最大 0.013，`tm_l1c13@ep7` 於 bolivia）。
- 顯著性：對測試集的事件 / chip 做 paired bootstrap（重抽 10,000 次），報差距與 95% 信賴區間。只涵蓋事件抽樣的變異，不含重新訓練（seed）的變異。

**跨資料集測試：Sen1Floods11 v1.1**

- 來源 `/home/NAS/house/ycchen-10014/data/sen1floods11_v1.1/v1.1`，512×512 手工標註 chip。用 `S2L1CHand`（13 波段，順序與數值尺度同 WorldFloods）
  與 `LabelHand`（−1 無資料 / 0 非水 / 1 水，轉成 WorldFloods GT 格式），由 `artifacts/runs/compare/prepare_sen1floods11.py` 轉成
  `/home/NAS/house/ycchen-10014/data/sen1floods11_wf/<group>/test/{S2,gt}`，整條 pipeline 不需修改。
- 分組：**`test`** = 官方 test split 90 chip 扣掉 Spain 6 chip，共 84 chip / 9 個國家地區；**`bolivia`** = 15 chip，Sen1Floods11 中完全未參與訓練的事件。
- **資料洩漏**：Spain 的 6 個 test chip 與 WorldFloods **train** 的 EMSR388（2019 年 9 月同一場洪水）空間重疊，因此不列入評估（另存 `test_spain_leak`）。其他國家無重疊。

## 3. 結果（256 tile 推論）

完整報表：`artifacts/runs/compare/report_worldfloods_tile256.md`、`report_sen1floods11_test_tile256.md`、`report_sen1floods11_bolivia_tile256.md`。

### 3.1 WorldFloods v2

**IoU water**（val 16 / test 18 個事件）

| model | val OVERALL | val AVERAGE | test OVERALL | test AVERAGE | test Precision | test Recall |
|---|---|---|---|---|---|---|
| **`unet_baseline@ep8`** | 0.8744 | 0.8012 | **0.8285** | **0.8302** | 0.8896 | 0.9235 |
| `tm_l2a6@ep20` | 0.8885 | 0.7826 | 0.8090 | 0.8095 | 0.9013 | 0.8876 |
| `tm_l1c13@ep7` | 0.8934 | 0.7938 | 0.8082 | 0.8126 | 0.8818 | 0.9064 |
| **`tm_l1c13_tim@ep16`** | 0.8932 | 0.7899 | **0.8179** | 0.8132 | 0.8835 | 0.9167 |
| `tm_l1c13_inskip@ep13` | 0.8943 | 0.7946 | 0.8000 | 0.8080 | 0.8978 | 0.8802 |
| `tm_l1c13_inskip@ep29`（last） | 0.8955 | 0.7938 | 0.7987 | 0.8024 | 0.8975 | 0.8789 |

**test 差距（paired bootstrap 95% CI）**

| model | vs `unet_baseline`：OVERALL | vs `unet_baseline`：AVERAGE | vs `tm_l1c13`：OVERALL | vs `tm_l1c13`：AVERAGE |
|---|---|---|---|---|
| `tm_l2a6@ep20` | **−2.0 pt [−3.5, −0.4]** | −2.1 pt [−5.2, +0.4] | +0.1 pt [−1.0, +1.9] | −0.3 pt [−2.2, +1.3] |
| `tm_l1c13@ep7` | **−2.0 pt [−4.2, −0.6]** | −1.8 pt [−5.7, +1.0] | — | — |
| `tm_l1c13_tim@ep16` | **−1.1 pt [−3.3, −0.2]** | −1.7 pt [−4.2, +0.2] | +1.0 pt [−0.5, +1.9] | +0.1 pt [−2.0, +2.0] |
| `tm_l1c13_inskip@ep13` | **−2.9 pt [−5.0, −0.5]** | −2.2 pt [−5.5, −0.0] | −0.8 pt [−2.5, +1.1] | −0.5 pt [−1.8, +0.7] |

### 3.2 Sen1Floods11（跨資料集）

**test**（84 chip，不含 Spain）

| model | IoU OVERALL | IoU AVERAGE | Precision | Recall | vs `unet_baseline`：OVERALL | vs `tm_l1c13`：OVERALL |
|---|---|---|---|---|---|---|
| `unet_baseline@ep8` | 0.6366 | 0.4131 | 0.6744 | 0.9191 | — | — |
| `tm_l2a6@ep20` | 0.6735 | 0.4218 | 0.7112 | 0.9270 | +3.7 pt [−0.2, +7.9] | +2.6 pt [−0.6, +6.2] |
| `tm_l1c13@ep7` | 0.6476 | 0.4265 | 0.6784 | 0.9345 | +1.1 pt [−2.0, +4.6] | — |
| **`tm_l1c13_tim@ep16`** | **0.6756** | **0.4333** | 0.7116 | 0.9304 | **+3.9 pt [+0.8, +7.5]** | **+2.8 pt [+0.5, +5.1]** |
| `tm_l1c13_tim@ep29`（last） | 0.6867 | 0.4317 | 0.7265 | 0.9261 | +5.0 pt [+1.6, +8.9] | — |
| `tm_l1c13_inskip@ep13` | 0.6708 | 0.4284 | 0.7042 | 0.9340 | +3.4 pt [−0.3, +7.2] | **+2.3 pt [+0.2, +4.2]** |
| `tm_l1c13_inskip@ep29`（last） | 0.6637 | 0.4224 | 0.6983 | 0.9305 | +2.7 pt [−0.8, +6.3] | — |

**bolivia**（15 chip，Sen1Floods11 中未參與訓練的事件；樣本少，信賴區間很寬）

| model | IoU OVERALL | IoU AVERAGE | Precision | Recall | vs `unet_baseline`：OVERALL | vs `tm_l1c13`：OVERALL |
|---|---|---|---|---|---|---|
| `unet_baseline@ep8` | 0.5416 | 0.3997 | 0.5500 | 0.9726 | — | — |
| `tm_l2a6@ep20` | 0.6498 | 0.4518 | 0.6680 | 0.9597 | +10.8 pt [−1.0, +23.2] | +12.0 pt [+5.9, +18.8] |
| `tm_l1c13@ep7` | 0.5298 | 0.3784 | 0.5350 | 0.9820 | −1.2 pt [−7.8, +5.8] | — |
| `tm_l1c13_tim@ep16` | 0.5962 | 0.4059 | 0.6052 | 0.9758 | +5.5 pt [−3.2, +14.1] | +6.6 pt [+3.6, +9.0] |
| `tm_l1c13_tim@ep29`（last） | 0.6231 | 0.4331 | 0.6359 | 0.9687 | +8.1 pt [−1.1, +17.3] | — |
| `tm_l1c13_inskip@ep13` | 0.5832 | 0.4096 | 0.5906 | 0.9788 | +4.2 pt [−3.8, +14.1] | +5.3 pt [+2.2, +8.8] |
| `tm_l1c13_inskip@ep29`（last） | 0.6012 | 0.4209 | 0.6102 | 0.9761 | +6.0 pt [−3.0, +16.4] | — |

所有模型在 Sen1Floods11 上 precision 都偏低（0.54–0.73）、recall 都很高（> 0.92），即普遍多報水；
可能來自兩個資料集標註習慣的差異（例如 Sen1Floods11 手工標註較保守），屬於跨資料集的系統性差異，各模型同樣受影響。

### 3.3 Uncertainty（DST；四種 uncertainty 的值幾乎相同）

| model | WF val：retention AUC / PAvPU | WF test：retention AUC / PAvPU | S1F11 test：retention AUC / PAvPU | bolivia：retention AUC / PAvPU |
|---|---|---|---|---|
| `unet_baseline@ep8` | 0.9909 / 0.9010 | **0.9688** / **0.8394** | **0.9427** / 0.8295 | 0.8901 / 0.6938 |
| `tm_l2a6@ep20` | 0.9914 / 0.9037 | 0.9498 / 0.7953 | 0.9244 / 0.8313 | **0.9544** / **0.7958** |
| `tm_l1c13@ep7` | 0.9927 / 0.8946 | 0.9648 / 0.8266 | 0.9355 / 0.8355 | 0.9190 / 0.7204 |
| `tm_l1c13_tim@ep16` | 0.9903 / 0.8982 | 0.9685 / 0.8273 | 0.9318 / **0.8442** | 0.8308 / 0.7789 |
| `tm_l1c13_inskip@ep13` | 0.9912 / 0.9007 | 0.9431 / 0.8051 | 0.9311 / 0.8390 | 0.8267 / 0.7713 |

- WorldFloods test 與 Sen1Floods11 test 上，`unet_baseline` 的 retention AUC 最高（`tm_l1c13_tim` 在 WorldFloods test 上並列）。
  TerraMind 在 Sen1Floods11 上分割較準，但不確定性排序沒有更好。
- PAvPU 在 Sen1Floods11 test 上 TerraMind 各組略高（0.831–0.844 vs 0.830）。bolivia 只有 15 chip，數值波動大。

### 3.4 推論 tile 尺寸的敏感度（WorldFloods test，1024 → 256）

| model | OVERALL 1024 → 256 | AVERAGE 1024 → 256 | Precision 1024 → 256 |
|---|---|---|---|
| `unet_baseline@ep8` | 0.8285 → 0.8285 | 0.8303 → 0.8302 | 0.8894 → 0.8896 |
| `tm_l2a6@ep20` | 0.8086 → 0.8090 | 0.8029 → 0.8095 | 0.8930 → 0.9013 |
| `tm_l1c13@ep7` | 0.8070 → 0.8082 | 0.8063 → 0.8126 | 0.8508 → 0.8818 |

- UNet 完全不受影響（對照組，符合預期）。TerraMind 在 256 tile 下 OVERALL 變化 ≤ 0.1 pt、AVERAGE 進步 0.6–0.7 pt，`tm_l1c13` 的 precision 明顯回升（多報水減少）。
- tile 尺寸**不改變結論**：TerraMind 在 WorldFloods test 上仍低於 `unet_baseline`。1024 tile 版的詳細數字見 commit `af11788` 的本文件。

### 3.5 訓練曲線（`lightning_logs/version_0/metrics.csv`）

圖：`artifacts/runs/compare/curves_ABDF.png`。`val_iou_land_water water` 不是 IoU 的正確估計，且依 batch size 而變（§3.6），只能看同組趨勢。

| model | best val_bce：epoch / val_bce | final（ep29）val_bce | epoch 0 val_bce |
|---|---|---|---|
| `unet_baseline`（batch 64） | 8 / 0.0621 | 0.0777 | 0.24 |
| `tm_l2a6` | 20 / 0.0536 | 0.0543 | 0.96 |
| `tm_l1c13` | 7 / 0.0514 | 0.0544 | 0.29 |
| `tm_l1c13_tim` | 16 / 0.0535 | 0.0563 | 0.53 |
| `tm_l1c13_inskip` | 13 / 0.0522 | 0.0538 | 0.50 |

解凍的各組 epoch 0 val_bce 很高（encoder 沒有 warmup），epoch 1 起恢復；epoch 10 之後都收斂在 0.052–0.056，明顯低於 `unet_baseline`。

### 3.6 訓練時 val IoU 的偏差與 pooled IoU

`EDL_ML4FloodsModel.validation_step` 沿用 ml4floods 的寫法：每個 val batch 用 `metrics.calculate_iou` 算一次 IoU，再由 Lightning 依 batch size 加權平均。兩個問題：

1. `calculate_iou`（`ml4floods/models/utils/metrics.py`）在 TP 加 1e-6、FP / FN 用總和減 TP 算出，所以**整個 batch 沒有水時**，全對會得到 −1、只要誤報 1 個像素就得到 0。
2. 小水量的 batch 稍錯就 IoU 很低，卻與水多的 batch 等權重。結果這個值**依 batch size 而變**。
   以真實 val（3524 tile，1817 張有水）計算，完美模型在 batch 16 只會被記成 0.80（無水 batch 22/221），在 batch 64 記成 0.93（2/56）。

`artifacts/runs/compare/eval_val_iou.py`（不改共用程式碼）用相同的 val dataloader 重算，`logged_*` 重現訓練 log 的值：

| model | logged，batch 16 | logged，batch 64 | **pooled IoU** | Precision | Recall |
|---|---|---|---|---|---|
| `unet_baseline@ep8` | 0.539 | 0.703 | 0.873 | 0.888 | 0.982 |
| `tm_l2a6@ep8` | 0.531 | 0.698 | 0.888 | 0.911 | 0.973 |
| `tm_l2a6@ep20` | — | — | 0.887 | 0.908 | 0.975 |
| `tm_l1c13@ep7` | — | — | 0.892 | 0.914 | 0.974 |

同一個 `unet_baseline` 在 batch 16 記成 0.539、batch 64 記成 0.703，所以訓練曲線上「UNet 0.70 vs TerraMind 0.51」的差距是 batch size 造成的假象
（先前版本歸因於「256 tile 對 ViT 的邊界效應」，**這個解釋是錯的**）。
`EDL_SAR_Unet.on_validation_epoch_end` 已有正確的 epoch 級 `val_Global_iou_*`，光學 EDL 基底類別沒有；修正共用 `edl.py` 前需與 SAR 支線協調。

### 3.7 錯誤離 GT 水陸邊界多遠（WorldFloods test，256 tile）

每個有效像素依「離最近 GT 水陸交界的距離」分層（1 px = 10 m），比較各層錯誤率
（`artifacts/runs/compare/boundary_error_analysis.py`，輸出 `boundary_error_test_tile256.txt`；1024 tile 版為 `boundary_error_test.txt`）。

| 離邊界的距離 | 像素占比 | `unet_baseline@ep8` | `tm_l1c13@ep7` | `tm_l2a6@ep20` | `tm_l1c13@ep7` 多出的錯誤落在此層的比例 |
|---|---|---|---|---|---|
| 1–2 px（緊貼邊界） | 1.8% | 0.397 | 0.398 | 0.403 | 0.5% |
| 2–4 | 2.6% | 0.264 | 0.276 | 0.276 | 4.3% |
| 4–8 | 3.7% | 0.152 | 0.170 | 0.164 | 9.7% |
| 8–16（約 1 個 patch） | 5.5% | 0.100 | 0.120 | 0.112 | 15.7% |
| 16–32（約 1–2 個 patch） | 8.2% | 0.072 | 0.093 | 0.085 | 23.9% |
| ≥32（遠離邊界） | 78.2% | 0.035 | 0.039 | 0.039 | 46.0% |

- **緊貼邊界處 TerraMind 與 UNet 一樣好**（1–2 px 錯誤率 0.398 vs 0.397）：「ViT 抓不到邊界細節」的假設不成立。
- `tm_l1c13@ep7` 多出的錯誤約 54% 在離水 2–32 px（約 1–2 個 ViT patch），錯誤率高 5–30%，以多報水為主（8–32 px 的多出錯誤中 FP 約 33 萬、FN 約 15 萬）。
  1024 tile 時這一段的多報水更嚴重（錯誤率高 20–50%），部分來自推論 tile 過大。
- 另約 46% 在遠離水體處（≥32 px），256 tile 下**以漏報為主**（FN 多約 83 萬、FP 反而少約 27 萬）：大水體內部有區塊沒抓到。`tm_l2a6@ep20` 也是同樣模式（FN 多約 121 萬）。
- `tm_l1c13_inskip` 原本針對「水體周圍多報水」，但在 WorldFloods test 上沒有改善（−0.8 pt vs `tm_l1c13`，CI 跨 0），recall 反而下降（0.880 vs 0.906）。

## 4. 結論

- **同資料集 vs 跨資料集結論相反**：WorldFloods test 上所有 TerraMind 設定都比 `unet_baseline` 低 1–3 pt（OVERALL 皆顯著）；
  Sen1Floods11 上 TerraMind 普遍較高，`tm_l1c13_tim@ep16` +3.9 pt [+0.8, +7.5] 顯著。
  最可能的解釋是 `unet_baseline` 整個網路從 WorldFloods 上訓練過的 WF2 起跑，在同資料集上有主場優勢；但這也受限於 §6 限制 1（無法確認它的訓練資料）。
- **`tm_l1c13_tim`（TiM）是目前最好的 TerraMind 設定**：WorldFloods test 上最接近 `unet_baseline`（−1.1 pt），Sen1Floods11 test 上相對 `tm_l1c13` +2.8 pt [+0.5, +5.1]、
  bolivia +6.6 pt [+3.6, +9.0]。生成的土地覆蓋先驗對跨資料集泛化有幫助。
- **`tm_l1c13_inskip`（全解析度 input skip）**：WorldFloods 上沒有幫助（−0.8 pt vs `tm_l1c13`），Sen1Floods11 上 +2.3 pt [+0.2, +4.2]、bolivia +5.3 pt [+2.2, +8.8]。
  原本針對的「水體周圍多報水」在 WorldFloods 上沒有改善，跨資料集的增益可能來自其他機制，尚無法解釋。
- **`tm_l2a6` vs `tm_l1c13`（L1C 13 波段）**：WorldFloods 上幾乎相同（+0.1 pt），Sen1Floods11 test 上 `tm_l2a6` 反而高 2.6 pt（CI 跨 0）、bolivia 高 12.0 pt。
  **L1C 13 波段沒有帶來一致的增益**；6 波段子集在跨資料集上可能較穩健（較少依賴兩個資料集間分佈差異較大的波段），待驗證。
- **val 無法可靠地選 ckpt**：`tm_l2a6` 的 best-val（ep20）test AVERAGE 比 ep8 低；`tm_l1c13` 的 best-val（ep7）test OVERALL 比 ep29 低（1024 tile）；
  Sen1Floods11 上 `tm_l1c13_tim@ep29` 又比 ep16 好。16 張 val 的 `val_bce` 對 test 表現幾乎沒有預測力。
- **Uncertainty**：TerraMind 預測較準的情況下（Sen1Floods11），retention AUC 仍不及 `unet_baseline`；EDL 不確定性的品質沒有隨 encoder 改善。

## 5. 待補結果

- `tm_l1c13_tim@ep29` 的 WorldFloods 結果（評估中）。
- WorldFloods 256 tile 的統一門檻 PAvPU（`make_reports.sh` 會一併算）。

## 6. 已知限制

1. **`unet_baseline` 的訓練資料與 ckpt 挑選無法確認**：它的 config 指向 `chlunchen-10030/datasets/cj/data/worldfloods_v2/data`，該目錄已不存在；
   無法確認它的訓練集與本文件各組相同。ckpt 是「lab 指定的 epoch 8」，挑選方式未記錄（若曾參考 test，`unet_baseline` 在 WorldFloods test 上會被高估）。
2. **單一 seed、單次訓練，硬體不同**：`unet_baseline` 在 chlunchen 機器、TerraMind 各組在 RTX 4090 上訓練。bootstrap CI 不含訓練變異，真實不確定性更大。
3. **val 只有 16 張**：挑 ckpt、LR scheduler（patience 2）與 early stopping 都依賴它。
4. **訓練預算不同**：`unet_baseline` batch 64（1025 steps / epoch），TerraMind batch 16（4099 steps / epoch），學習率未隨 batch size 調整。
5. **跨資料集差異**：Sen1Floods11 與 WorldFloods 的標註習慣不同（§3.2），所有模型 precision 都偏低；跨資料集比較只看模型間相對差距。
6. 訓練資料是 symlink 拼成的複本（train 來自 `cynthia-10040/.../worldfloods_v2_smoke_source`），檔案數與 metadata 一致，未逐檔比對內容。

## 7. 執行紀錄

- TerraMind 各組在 RTX 4090 24 GB 上訓練，依序執行不併行；`tm_l2a6` / `tm_l1c13` / `tm_l1c13_inskip` 30 epoch 各約 5.5 / 7.5 / 6.3 小時，顯存約 8 GB；
  `tm_l1c13_tim` 約 10.6 小時、顯存約 11 GB；凍結的 TiM 生成器也存進 ckpt，每個約 2 GB。
- `tm_l1c13_tim` 在 WorldFloods 整圖推論時，影像邊緣的非正方形 tile 會讓 TiM 的 token embedding 報錯（`Input tokens are not squared`）；
  已在 `TerraMindUNet.forward` 開啟 TiM 時把輸入補成正方形再裁回（訓練 tile 本來就是正方形，不受影響）。Sen1Floods11 的 chip 都是正方形，沒有受影響。
- 該容器的 `UV_PROJECT_ENVIRONMENT` 預設指向 image 自帶的 `/opt/venv`（不是專案環境），需改用專案 `.venv`（Python 3.10）；
  `compose.yml` 的 `shm_size` 曾誤拼為 `shm_sizes`，`/dev/shm` 只有 64 MB，修正後 DataLoader 才能正常使用多 worker。
- `analysis_S2.py` / `compute_pavpu.py` 的輸出路徑只有 model_type 沒有 subset，val 與 test 會互相覆蓋；本次用 `analysis_S2/{val,test}/<tag>/` 分開。
  `eval_metrics.py` / `analysis_S2.py` / `compute_pavpu.py` 的 `--subset` 只接受 `val|test`，所以 Sen1Floods11 每組各自是一個資料根目錄。
- 同一 NAS 上多機訓練時實驗目錄名稱必須不同（凍結的兩組曾因另一台機器用同一 config 啟動而混寫 ckpt 與 log）。
- 推論 / 分析腳本在 `artifacts/runs/compare/`（`run_pipeline_tile.sh`、`run_inference_tile.py`、`report.py`、`make_reports.sh`、`pick_ckpts.py`、
  `eval_val_iou.py`、`pavpu_fixed.py`、`boundary_error_analysis.py`、`prepare_sen1floods11.py`、`summarize.py`、`plot_curves.py`、`bench_train_step.py`），未納入 git。

## 附錄 A：凍結 encoder（`tm_frozen_l2a6`，1024 tile）

`tm_frozen_l2a6`：TerraMind 凍結、decoder 隨機、`bgriswirs` 6 波段 → S2L2A，batch 16；ckpt ep3（best val）。
Config `configurations/terramind/tm_frozen_l2a6.json`，實驗目錄 `edl_terramind_v1_base_bgriswirs_freeze`，結果 tag `EDL-TERRAMIND_freeze`。

| model | val OVERALL | val AVERAGE | test OVERALL | test AVERAGE | test retention AUC |
|---|---|---|---|---|---|
| `unet_baseline@ep8` | 0.8745 | 0.8016 | 0.8285 | 0.8303 | 0.9688 |
| `tm_frozen_l2a6@ep3` | 0.8474 | 0.7631 | 0.8171 | 0.8211 | 0.9665 |

與 `unet_baseline` 的 test 差距：OVERALL −1.1 pt [−2.0, +0.5]、AVERAGE −0.9 pt [−2.5, +0.8]。pooled val IoU 0.864（整圖 0.847）。
凍結的 TerraMind 加上從零訓練的 decoder，就達到 `unet_baseline` 97–99% 的整圖 IoU，uncertainty 品質相當；
解凍後 val 明顯進步（`tm_frozen_l2a6` → `tm_l2a6`），test 沒有。

`tm_frozen_l2a6_wf2dec`（凍結 + WF2 decoder）與 `tm_l2a6_wf2dec`（解凍 + WF2 decoder）的結論：WF2 decoder 只能部分載入（12/30 tensors），
與隨機初始化相比互有勝負、差距 ≤ 1 pt，沒有一致的幫助。完整數字見 commit `e9cc0ba` 的本文件與 `artifacts/runs/compare/tables_ABCDF.md`。
