# EDL vs EDL + TerraMind encoder：實驗比較

分支 `feat/terramind-edl-backbone`（PR #1）。各組共用 EDL head / loss / 資料 / 訓練超參數，只差 encoder 與輸入模態。
實作與設計理由見 `terramind_edl_integration_log.md`。

**實驗方向**：TerraMind encoder **全部解凍**微調，以 F 為 TerraMind 主線的基準，每個新實驗只對 F 改一個變因；A 是要超越的目標。
凍結 encoder 的結果移到附錄 A（B 組）；C（凍結 + WF2 decoder）與 D2（解凍 + WF2 decoder）已回答「WF2 decoder 初始化沒有一致的幫助」，
不再列入比較（數字見 commit `e9cc0ba` 的本文件）。

## 1. 實驗組合

| 組 | 設定 | 單變因對照 | 狀態 |
|---|---|---|---|
| **A** | UNet baseline（`EDL_ML4FloodsModel`），encoder + decoder 從 WF2 預訓練起跑，batch 64 | 目標 | 完成 |
| **D1** | TerraMind 解凍，`bgriswirs` 6 波段 → S2L2A embedding（波段子集） | 消融：與 F 比，看 L1C 13 波段的效果 | 完成 |
| **F** | TerraMind 解凍，`all` 13 波段 → **S2L1C** embedding | TerraMind 主線基準 | 完成 |
| **E** | F + TiM（從 S2L1C 生成 LULC token，argmax 解碼，生成器凍結） | F | 訓練中 |
| 計畫 | F + 全解析度 CNN skip | F | 未開始 |

**TerraMind 各組共同設定**：`EDL_TerraMind_ML4FloodsModel`；TerraMind v1 base（ViT-B/16，HF `ibm-esa-geospatial/TerraMind-1.0-base`），
取第 2/5/8/11 層 → 1×1 lateral → stride 4/8/16/32 金字塔；decoder 同 UNet 結構 + `dconv_full`（stride 4 → 全解析度），隨機初始化；
encoder 全部可訓練，lr = 1e-4 × `backbone_lr_mult` 0.1 = 1e-5（decoder / head 1e-4）；batch 16（4099 steps / epoch）、`num_workers 16`、fp32。

**F 的動機**：WorldFloods v2 的 S2 是 **L1C**（13 波段），D1 卻用 TerraMind 的 S2L2A embedding（地表反射率）並靠重新正規化對齊；
TerraMind 權重本身有 `untok_sen2l1c@224`（13 波段），F 讓模態與資料一致。這也是 E 的前置條件：TiM 生成器只吃完整的預訓練模態。

| | Config | 實驗名稱（`artifacts/models/`） |
|---|---|---|
| A | `configurations/edl.json`（訓練於 chlunchen 機器） | `edl_EpochKL_20260819`（symlink） |
| D1 | `artifacts/runs/edl_terramind_nofreeze.json` | `edl_terramind_v1_base_bgriswirs_nofreeze` |
| F | `artifacts/runs/edl_terramind_s2l1c_nofreeze.json` | `edl_terramind_v1_base_s2l1c_all_nofreeze` |
| E | `artifacts/runs/edl_terramind_s2l1c_nofreeze_tim.json` | `edl_terramind_v1_base_s2l1c_all_nofreeze_tim_lulc` |

**共同條件**：WorldFloods v2（train 475 / **val 16** / test 18 張 S2；每 epoch 約 65,600 個 window），`max_tile_size 256`、
`filter_windows`（`threshold_clouds 0.8`）、Adam、ReduceLROnPlateau（factor 0.5 / patience 2）on `val_bce_land_water`、
30 epoch（實際跑滿）、`pos_weight` / `weight_problem` 同、KL annealing linear / coef 0.5 / step 10。

## 2. 評估規則與指標

**Checkpoint 規則（看 test 之前就固定）**：每個 run 報兩個 ckpt，**結論只用第一個**：

1. **best val**：`val_bce_land_water` 最低的 ckpt。
2. **last**：最後一個 epoch（ep29），只作補充，不用來下結論。

A 用 lab 指定的 `epoch=8-step=9225`，它同時也是 A 的 best val。

| | best val（結論用） | last（補充） |
|---|---|---|
| A | ep8 | — |
| D1 | ep20 | ep29（待補評估） |
| F | ep7 | ep29 |

**Pipeline**（`artifacts/runs/compare/run_pipeline.sh <model_type> <config> <ckpt>`）：
`run_inference.py`（val + test 整圖）→ `eval_metrics.py` → `analysis_S2.py`（retention）→ `compute_pavpu.py`（PAvPU）；
TerraMind 輸出依 ckpt 改名為 tag（例如 `EDL-TERRAMIND_s2l1c_ep7`）。另以 `eval_val_iou.py` 算 val 的 pooled IoU（§3.4）。

**指標**

- 分割：每張影像把 EDL 機率四捨五入成水 / 非水，與 GT 比出 TP / FP / FN / TN（invalid 排除）。IoU water = TP / (TP + FP + FN)。
  **OVERALL** = 所有影像像素加總後算一次（大事件權重大）；**AVERAGE** = 每張影像各算再平均（每個事件等權重）。**兩者都要看。**
- Retention curve：依不確定性由低到高保留最確定的 X% 像素，對保留的像素算 IoU；報曲線下面積（normalized AUC）與 X = 50% 時的 IoU。
- PAvPU：3×3 patch 判斷「準不準」（patch IoU ≥ 門檻）與「確不確定」（平均不確定性 ≥ 門檻），PAvPU = (準且確定 + 不準且不確定) / 全部。
  **限制**：兩個門檻取自**各模型自己** retention 90% 的那一點，IoU 越高的模型門檻越嚴，所以 **PAvPU 不能跨模型比較**，只列供參考。
- 顯著性：test 18 個事件做 paired bootstrap（重抽事件 10,000 次），報與 A 的差距與 95% 信賴區間。只涵蓋事件抽樣的變異，不含重新訓練（seed）的變異。

## 3. 結果

D1 完成於 2026-10-01，F 於 2026-10-04。表格由 `artifacts/runs/compare/summarize.py` 產生（含已移除組別的完整版：`artifacts/runs/compare/tables_ABCDF.md`）。

### 3.1 Segmentation

**IoU water**（粗體為結論用 ckpt）

| model / ckpt | val OVERALL | val AVERAGE | test OVERALL | test AVERAGE |
|---|---|---|---|---|
| **A. UNet baseline ep8** | 0.8745 | 0.8016 | **0.8285** | **0.8303** |
| **D1. S2L2A-6, ep20 (best val)** | 0.8907 | 0.7820 | 0.8086 | 0.8029 |
| **F. S2L1C-13, ep7 (best val)** | 0.8941 | 0.7967 | 0.8070 | 0.8063 |
| F. S2L1C-13, ep29 (last) | 0.8951 | 0.7979 | 0.8264 | 0.8070 |
| D1. S2L2A-6, ep8（補充，非規則 ckpt） | 0.8887 | 0.7839 | 0.8075 | 0.8265 |

**與 A 的 test 差距（paired bootstrap 95% CI）**

| model / ckpt | test OVERALL | test AVERAGE |
|---|---|---|
| **D1 ep20 (best val)** | **−2.0 pt [−4.4, −1.0]** | **−2.7 pt [−6.3, −0.4]** |
| **F ep7 (best val)** | −2.1 pt [−7.3, +0.7] | −2.4 pt [−5.8, +0.5] |
| F ep29 (last) | −0.2 pt [−3.8, +1.7] | **−2.3 pt [−4.4, −0.4]** |

**test Precision / Recall（OVERALL）**

| model / ckpt | Precision | Recall | F1 |
|---|---|---|---|
| A ep8 | 0.8894 | 0.9236 | 0.9062 |
| D1 ep20 | 0.8930 | 0.8954 | 0.8942 |
| F ep7 | 0.8508 | 0.9399 | 0.8932 |
| F ep29 | 0.8857 | 0.9251 | 0.9050 |

**test per-event（A vs D1 ep20 vs F ep7）**：18 個事件依水體像素排序，前 5 大事件平均 IoU A 0.866 / D1 0.844 / F 0.833，
其餘 13 個 A 0.817 / D1 0.787 / F 0.796。D1 有 5/18、F 有 6/18 個事件贏 A。F ep7 在前 5 大事件中有 4 個比 A 多報水
（FP 多約 15 萬–100 萬像素），precision 0.851 是各組最低。

### 3.2 Uncertainty

**Retention curve（DST；四種 uncertainty 的值幾乎相同）：normalized AUC / retention 50% 時的 IoU**

| model / ckpt | val | test |
|---|---|---|
| A ep8 | 0.9909 / 0.9976 | 0.9688 / 0.9828 |
| D1 ep20 | 0.9914 / 0.9978 | 0.9554 / 0.9644 |
| F ep7 | 0.9923 / 0.9977 | **0.9709 / 0.9898** |
| F ep29 | 0.9895 / 0.9949 | 0.9698 / 0.9903 |

**PAvPU（DST，OVERALL，patch 3×3，retention target 0.9；不能跨模型比較，見 §2）**

| model / ckpt | val | test |
|---|---|---|
| A ep8 | 0.9010 | 0.8394 |
| D1 ep20 | 0.8984 | 0.8071 |
| F ep7 | 0.8856 | 0.8356 |
| F ep29 | 0.8933 | 0.7999 |

### 3.3 訓練曲線（`lightning_logs/version_0/metrics.csv`）

圖：`artifacts/runs/compare/curves_ABDF.png`。`val_iou_land_water water` 不是 IoU 的正確估計，且依 batch size 而變（§3.4），只能看同組趨勢。

| model | best val_bce：epoch / val_bce | final（ep29）val_bce | epoch 0 val_bce |
|---|---|---|---|
| A（batch 64） | 8 / 0.0621 | 0.0777 | 0.24 |
| D1 | 20 / 0.0536 | 0.0543 | 0.96 |
| F | 7 / 0.0514 | 0.0544 | 0.29 |

解凍的兩組 epoch 0 val_bce 很高（encoder 沒有 warmup），epoch 1 起恢復；epoch 10 之後都收斂在 0.054–0.056，明顯低於 A。
F 的 ep7（0.0514）是單點低值，前後 epoch 都較高。

### 3.4 訓練時 val IoU 的偏差與 pooled IoU

`EDL_ML4FloodsModel.validation_step` 沿用 ml4floods 的寫法：每個 val batch 用 `metrics.calculate_iou` 算一次 IoU，再由 Lightning 依 batch size 加權平均。兩個問題：

1. `calculate_iou`（`ml4floods/models/utils/metrics.py`）在 TP 加 1e-6、FP / FN 用總和減 TP 算出，所以**整個 batch 沒有水時**，全對會得到 −1、只要誤報 1 個像素就得到 0。
2. 小水量的 batch 稍錯就 IoU 很低，卻與水多的 batch 等權重。結果這個值**依 batch size 而變**。
   以真實 val（3524 tile，1817 張有水）計算，完美模型在 batch 16 只會被記成 0.80（無水 batch 22/221），在 batch 64 記成 0.93（2/56）。

`artifacts/runs/compare/eval_val_iou.py`（不改共用程式碼）用相同的 val dataloader 重算，`logged_*` 重現訓練 log 的值（D1 ep8 0.5311、A ep8 bs64 0.7030 皆與 csv 一致）：

| ckpt | logged，batch 16 | logged，batch 64 | **pooled IoU** | Precision | Recall |
|---|---|---|---|---|---|
| A ep8 | 0.539 | 0.703 | 0.873 | 0.888 | 0.982 |
| D1 ep8 | 0.531 | 0.698 | 0.888 | 0.911 | 0.973 |
| D1 ep20 | — | — | 0.887 | 0.908 | 0.975 |
| F ep7 | — | — | 0.892 | 0.914 | 0.974 |
| F ep29 | — | — | 0.893 | 0.916 | 0.973 |

同一個 A 模型在 batch 16 記成 0.539、batch 64 記成 0.703，所以訓練曲線上「A 0.70 vs TerraMind 0.51」的差距是 batch size 造成的假象
（先前版本歸因於「256 tile 對 ViT 的邊界效應」，**這個解釋是錯的**）。pooled IoU 與整圖推論的 val OVERALL 一致（A 0.873 vs 0.8745、D1 0.887 vs 0.8907、F 0.892 vs 0.8941）。
`EDL_SAR_Unet.on_validation_epoch_end` 已有正確的 epoch 級 `val_Global_iou_*`，光學 EDL 基底類別沒有；修正共用 `edl.py` 前需與 SAR 支線協調。

## 4. 結論

- **解凍的 TerraMind 在 val 上一致超過 A，在 test 上沒有**：依規則選的 ckpt，D1 與 F 的 test IoU 都比 A 低約 2 pt（OVERALL 與 AVERAGE 皆然）；
  D1 的差距在統計上顯著，F 的信賴區間跨過 0。**目前沒有任何 TerraMind 設定在 test 上超過 A。**
- **D1 vs F（L1C 13 波段）**：val 上 F 略好（OVERALL 0.894 vs 0.891、pooled 0.892 vs 0.887），test 上兩者幾乎一樣（OVERALL 0.807 vs 0.809、AVERAGE 0.806 vs 0.803）。
  依規則選 ckpt 時，**L1C 13 波段沒有帶來可測得的 test 增益**。F 的 last ckpt（ep29）test OVERALL 0.826 接近 A，但那是看過 test 才知道的，不能作為結論。
- **val 無法可靠地選 ckpt**：D1 的 best-val（ep20）test AVERAGE 比 ep8 低 2.4 pt；F 的 best-val（ep7）test OVERALL 比 ep29 低 1.9 pt。
  兩組偏差方向相反，表示 16 張 val 的 `val_bce` 對 test 表現幾乎沒有預測力。
- **主要失分模式**：大事件多報水（F ep7 precision 0.851）與小事件細節較差。後者與 ViT 最細只到 1/16 解析度的特徵一致，
  是「F + 全解析度 CNN skip」實驗要處理的問題；前者是 E（TiM，加入土地覆蓋先驗）要處理的問題。
- **Uncertainty**：F 的 test retention AUC（0.971）略高於 A（0.969），D1 較低（0.955）。PAvPU 不能跨模型比較，不據以下結論。

## 5. 已知限制

1. **A 的訓練資料與 ckpt 挑選無法確認**：A 的 config 指向 `chlunchen-10030/datasets/cj/data/worldfloods_v2/data`，該目錄已不存在；
   無法確認 A 的訓練集與本文件各組相同。A 的 ckpt 是「lab 指定的 epoch 8」，挑選方式未記錄（若曾參考 test，A 會被高估）。
2. **推論 tile 尺寸與訓練不一致**：訓練用 256×256，整圖推論 `run_inference.py` 用 `max_tile_size 1024`。UNet 為卷積網路影響不大，
   ViT 推論時 token 數為訓練的 16 倍、位置編碼內插 4 倍。val 上 D1 / F 的 pooled（256 tile）與整圖 IoU 一致，看不出影響；
   凍結的 B 掉 1.7 pt（附錄 A）。test 上尚未驗證，待以 `max_tile_size 256` 重跑。
3. **單一 seed、單次訓練，硬體不同**：A 在 chlunchen 機器、D1 / F 在 RTX 4090 上訓練。bootstrap CI 不含訓練變異，真實不確定性更大。
4. **val 只有 16 張**：挑 ckpt、LR scheduler（patience 2）與 early stopping 都依賴它。
5. **A 與 TerraMind 的訓練預算不同**：A batch 64（1025 steps / epoch），TerraMind batch 16（4099 steps / epoch），學習率未隨 batch size 調整。
6. 訓練資料是 symlink 拼成的複本（train 來自 `cynthia-10040/.../worldfloods_v2_smoke_source`），檔案數與 metadata 一致，未逐檔比對內容。

## 6. 執行紀錄

- D1 / F 在 RTX 4090 24 GB 上訓練，依序執行不併行；30 epoch 各約 5.5 / 7.5 小時，顯存約 8 GB。
  該容器的 `UV_PROJECT_ENVIRONMENT` 預設指向 image 自帶的 `/opt/venv`（不是專案環境），需改用專案 `.venv`（Python 3.10）；
  `compose.yml` 的 `shm_size` 曾誤拼為 `shm_sizes`，`/dev/shm` 只有 64 MB，修正後 DataLoader 才能正常使用多 worker。
- E（TiM）速度基準：3.05 it/s、顯存峰值 9.3 GiB（F 為 5.21 it/s、6.0 GiB），30 epoch 約 13 小時；凍結的生成器也存進 ckpt，每個約 2 GB。
- `analysis_S2.py` / `compute_pavpu.py` 的輸出路徑只有 model_type 沒有 subset，val 與 test 會互相覆蓋；本次用 `analysis_S2/{val,test}/<tag>/` 分開。
- 同一 NAS 上多機訓練時 `experiment_name` 必須不同（B / C 曾因另一台機器用同一 config 啟動而混寫 ckpt 與 log）。
- 推論 / 分析腳本與產生表格的程式在 `artifacts/runs/compare/`（`run_pipeline.sh`、`eval_D.sh`、`summarize.py`、`curves.py`、`plot_curves.py`、`eval_val_iou.py`、`bench_train_step.py`），未納入 git。

## 附錄 A：凍結 encoder（B 組）

B：TerraMind 凍結、decoder 隨機、`bgriswirs` 6 波段 → S2L2A，batch 16；ckpt ep3（best val）。
Config `artifacts/runs/edl_terramind_freeze.json`，實驗名稱 `edl_terramind_v1_base_bgriswirs_freeze`。

| | val OVERALL | val AVERAGE | test OVERALL | test AVERAGE | test retention AUC |
|---|---|---|---|---|---|
| A ep8 | 0.8745 | 0.8016 | 0.8285 | 0.8303 | 0.9688 |
| B ep3 | 0.8474 | 0.7631 | 0.8171 | 0.8211 | 0.9665 |

與 A 的 test 差距：OVERALL −1.1 pt [−2.0, +0.5]、AVERAGE −0.9 pt [−2.5, +0.8]。pooled val IoU 0.864（整圖 0.847）。
凍結的 TerraMind 加上從零訓練的 decoder，就達到 A 的 97–99% 整圖 IoU，uncertainty 品質相當；解凍後 val 明顯進步（B → D1），test 沒有。

C（凍結 + WF2 decoder）與 D2（解凍 + WF2 decoder）的結論：WF2 decoder 只能部分載入（12/30 tensors），與隨機初始化相比互有勝負、差距 ≤ 1 pt，
沒有一致的幫助。完整數字見 commit `e9cc0ba` 的本文件與 `artifacts/runs/compare/tables_ABCDF.md`。
