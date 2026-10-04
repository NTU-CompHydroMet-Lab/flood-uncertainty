# EDL vs EDL + TerraMind encoder：實驗比較

分支 `feat/terramind-edl-backbone`（PR #1）。各組模型共用 EDL head / loss / 資料 / 訓練超參數，只差 encoder、初始權重與（F 組）輸入模態。
實作與設計理由見 `terramind_edl_integration_log.md`。

## 1. 實驗模型

**凍結 encoder（A / B / C）**

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

**解凍 encoder（D1 / D2 / F）**：`backbone_freeze: false`、encoder lr = `lr × backbone_lr_mult`（1e-4 × 0.1 = 1e-5），其餘同 B。

| | D1. unfrozen | D2. unfrozen + WF2 decoder | F. unfrozen, S2L1C 13 波段 |
|---|---|---|---|
| **與誰單變因對照** | B（凍結與否） | D1（decoder 初始化） | D1（輸入模態 / 波段） |
| **Decoder 權重** | 隨機 | 同 C（WF2 12/30 tensors） | 隨機 |
| **輸入** | `bgriswirs` 6 波段 → TerraMind **S2L2A** embedding（波段子集） | 同 D1 | `all` 13 波段 → TerraMind **S2L1C** embedding（全波段，L1C 統計值） |
| **可訓練參數** | 97.1M（含 encoder） | 同 D1 | 同 D1（patch embedding 13 波段） |
| **batch / num_workers / 精度** | 16 / 16 / fp32 | 同 | 同 |
| **Config** | `artifacts/runs/edl_terramind_nofreeze.json` | `artifacts/runs/edl_terramind_nofreeze_wf2dec.json` | `artifacts/runs/edl_terramind_s2l1c_nofreeze.json` |
| **實驗名稱** | `edl_terramind_v1_base_bgriswirs_nofreeze` | `edl_terramind_v1_base_bgriswirs_nofreeze_wf2dec` | `edl_terramind_v1_base_s2l1c_all_nofreeze` |
| **評估 ckpt** | ep8、ep20（best `val_bce`） | ep8（best `val_bce`） | ep7（best `val_bce`）、ep29（最後） |

F 的動機：WorldFloods v2 的 S2 是 **L1C**（13 波段），B / C / D 卻用 TerraMind 的 S2L2A embedding（地表反射率）並靠重新正規化對齊；
TerraMind 權重本身有 `untok_sen2l1c@224`（13 波段），F 讓模態與資料一致。這也是 TiM（Thinking in Modalities）實驗 E 的前置條件，
TiM 的生成器只吃完整的預訓練模態，不能用波段子集。

`artifacts/runs/*.json` 由 `configurations/edl_terramind.json` 複製，只改上表列出的欄位。

**共同條件**：WorldFloods v2（train 475 / val 17 / test 18 張 S2；每 epoch 約 65,600 個 window），`max_tile_size 256`、
`filter_windows` 同（`threshold_clouds 0.8`）、Adam `lr 1e-4`、ReduceLROnPlateau（factor 0.5 / patience 2）on `val_bce_land_water`、
30 epoch、`early_stopping_patience 50`（實際跑滿）、`pos_weight` / `weight_problem` 同、KL annealing linear / coef 0.5 / step 10。

**各組回答的問題**

- A vs B：凍結的 foundation-model 特徵 + 從零學的 decoder，追不追得上 fine-tune 過的 UNet。
- B vs C：decoder 初始化是不是 B 落後的主因（單變因對照）。
- A vs C：encoder 換成凍結 TerraMind、其餘盡量對齊後的真實差距。
- B vs D1：解凍 encoder 有多少增益；A vs D1：解凍後能否超過 UNet。
- D1 vs D2：解凍時 decoder 用 WF2 初始化有沒有幫助。
- D1 vs F：輸入改成與資料一致的 S2L1C 全波段有沒有幫助。

## 2. 評估項目

每組用同一條 pipeline（`artifacts/runs/compare/run_pipeline.sh <model_type> <config> <ckpt>`）：

1. `scripts/inference/run_inference.py` → `artifacts/results/val_test_inference/{val,test}/<model_type>/`
2. `scripts/eval/eval_metrics.py --save_tif` → `metrics_<model_type>.csv`（per-event + OVERALL pixel 加總 + AVERAGE per-event 平均）與 `cm_*.tif`
3. `scripts/analysis/analysis_S2.py` → retention curve（DST / aleatoric / epistemic / a+e）
4. `scripts/analysis/compute_pavpu.py` → PAvPU

加上訓練曲線（`lightning_logs/version_0/metrics.csv` 的 `val_bce_land_water`、`val_iou_land_water water`）與 val 的 pooled IoU（§3.4）。

TerraMind 各組的 model_type 都是 `EDL-TERRAMIND`，輸出目錄依 model_type 命名，所以每組跑完後改名為各自的 tag
（`EDL-TERRAMIND_freeze`、`_wf2dec`、`_nofreeze_ep8`、`_nofreeze_ep20`、`_nofreeze_wf2dec_ep8`、`_s2l1c_ep7`、`_s2l1c_ep29`）。

**指標定義**

- 分割：每張影像把 EDL 機率四捨五入成水 / 非水，與 GT 比出 TP / FP / FN / TN（invalid 像素排除）。
  IoU water = TP / (TP + FP + FN)。**OVERALL** = 所有影像像素加總後算一次（大事件權重大）；**AVERAGE** = 每張影像各算再平均（每個事件等權重）。
- Retention curve：依不確定性由低到高保留最確定的 X% 像素，對保留的像素算 IoU；報曲線下面積（normalized AUC）與 X = 50% 時的 IoU。
- PAvPU：影像切 3×3 patch，判斷「準不準」（patch IoU ≥ 門檻）與「確不確定」（平均不確定性 ≥ 門檻），兩個門檻取自 retention 90% 那一點；
  PAvPU = (準且確定 + 不準且不確定) / 全部 patch。

## 3. 結果

A / B / C 完成於 2026-09-29，D1 / D2 於 2026-10-01，F 於 2026-10-04。表格由 `artifacts/runs/compare/summarize.py` 產生（完整版 `artifacts/runs/compare/tables_ABCDF.md`）。
ckpt：A `epoch=8-step=9225`（lab 指定）；B `epoch=3-step=16396`、C `epoch=21-step=90178`、D2 `epoch=8-step=36891`（各自 best `val_bce_land_water`）；
D1 ep8 `epoch=8-step=36891` 與 ep20 `epoch=20-step=86079`（best）；F ep7 `epoch=7-step=32792`（best）與 ep29 `epoch=29-step=122970`。

### 3.1 Segmentation（val / test）

**val OVERALL**（pixel 加總）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9666 | 0.8890 | 0.9816 | 0.9330 | 0.8745 |
| B. TerraMind frozen | 0.9581 | 0.8594 | 0.9838 | 0.9174 | 0.8474 |
| C. frozen + WF2 decoder | 0.9610 | 0.8692 | 0.9832 | 0.9227 | 0.8564 |
| D1. unfrozen, ep8 | 0.9712 | 0.9119 | 0.9722 | 0.9411 | 0.8887 |
| D1. unfrozen, ep20 (best val_bce) | 0.9718 | 0.9146 | 0.9715 | 0.9422 | 0.8907 |
| D2. unfrozen + WF2 decoder, ep8 (best val_bce) | 0.9683 | 0.8956 | 0.9803 | 0.9360 | 0.8798 |
| F. unfrozen S2L1C-13, ep7 (best val_bce) | 0.9728 | 0.9185 | 0.9711 | 0.9441 | 0.8941 |
| F. unfrozen S2L1C-13, ep29 | 0.9732 | 0.9230 | 0.9673 | 0.9446 | **0.8951** |

**val AVERAGE**（per-event 平均）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9623 | 0.8487 | 0.9315 | 0.8831 | **0.8016** |
| B. TerraMind frozen | 0.9536 | 0.8074 | 0.9301 | 0.8576 | 0.7631 |
| C. frozen + WF2 decoder | 0.9557 | 0.8258 | 0.9233 | 0.8646 | 0.7752 |
| D1. unfrozen, ep8 | 0.9602 | 0.8174 | 0.9495 | 0.8660 | 0.7839 |
| D1. unfrozen, ep20 (best val_bce) | 0.9610 | 0.8275 | 0.9302 | 0.8661 | 0.7820 |
| D2. unfrozen + WF2 decoder, ep8 (best val_bce) | 0.9565 | 0.8092 | 0.9519 | 0.8618 | 0.7781 |
| F. unfrozen S2L1C-13, ep7 (best val_bce) | 0.9634 | 0.8347 | 0.9400 | 0.8769 | 0.7967 |
| F. unfrozen S2L1C-13, ep29 | 0.9658 | 0.8464 | 0.9259 | 0.8796 | 0.7979 |

**test OVERALL**（pixel 加總）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9418 | 0.8894 | 0.9236 | 0.9062 | **0.8285** |
| B. TerraMind frozen | 0.9382 | 0.8915 | 0.9074 | 0.8994 | 0.8171 |
| C. frozen + WF2 decoder | 0.9231 | 0.8386 | 0.9258 | 0.8800 | 0.7857 |
| D1. unfrozen, ep8 | 0.9337 | 0.8744 | 0.9135 | 0.8935 | 0.8075 |
| D1. unfrozen, ep20 (best val_bce) | 0.9355 | 0.8930 | 0.8954 | 0.8942 | 0.8086 |
| D2. unfrozen + WF2 decoder, ep8 (best val_bce) | 0.9350 | 0.8743 | 0.9186 | 0.8959 | 0.8115 |
| F. unfrozen S2L1C-13, ep7 (best val_bce) | 0.9315 | 0.8508 | 0.9399 | 0.8932 | 0.8070 |
| F. unfrozen S2L1C-13, ep29 | 0.9408 | 0.8857 | 0.9251 | 0.9050 | 0.8264 |

**test AVERAGE**（per-event 平均）

| model | Acc | Precision | Recall | F1 | IoU water |
|---|---|---|---|---|---|
| A. EDL baseline | 0.9586 | 0.9089 | 0.9050 | 0.9035 | **0.8303** |
| B. TerraMind frozen | 0.9544 | 0.8956 | 0.9055 | 0.8984 | 0.8211 |
| C. frozen + WF2 decoder | 0.9438 | 0.8415 | 0.9301 | 0.8818 | 0.7946 |
| D1. unfrozen, ep8 | 0.9537 | 0.8822 | 0.9277 | 0.9024 | 0.8265 |
| D1. unfrozen, ep20 (best val_bce) | 0.9503 | 0.9140 | 0.8689 | 0.8843 | 0.8029 |
| D2. unfrozen + WF2 decoder, ep8 (best val_bce) | 0.9521 | 0.8629 | 0.9349 | 0.8949 | 0.8146 |
| F. unfrozen S2L1C-13, ep7 (best val_bce) | 0.9480 | 0.8723 | 0.9167 | 0.8884 | 0.8063 |
| F. unfrozen S2L1C-13, ep29 | 0.9525 | 0.9008 | 0.8880 | 0.8872 | 0.8070 |

**test per-event（A vs D1 ep8 vs F ep29）**：18 個事件依水體像素排序，前 5 大事件的平均 IoU A 0.866 / D1 0.835 / F 0.854，
其餘 13 個 A 0.817 / D1 0.823 / F 0.789。D1 ep8 有 5/18 個事件贏 A，F ep29 有 4/18。
D1 輸 A 最多的是水量最大的事件（例如 `EMSR342_07SOUTHNORMANTON` 1800 萬水像素 −2.4 pt、`EMSR438_AOI02` −6.2 pt），主因是多報水（FP 多約 30 萬–43 萬像素）；
F 在大事件上追回大部分差距，但在 `EMSR264_18MIANDRIVAZODETAIL`（−11.7 pt）、`EMSR466_AOI01`（−11.3 pt）等小事件明顯較差。

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
| D1. unfrozen, ep8 | val | 0.8775 | 0.8776 | 0.8771 | 0.8775 |
| D1. unfrozen, ep8 | test | 0.8276 | 0.8268 | 0.8279 | 0.8272 |
| D1. unfrozen, ep20 (best val_bce) | val | 0.8984 | 0.8985 | 0.8981 | 0.8985 |
| D1. unfrozen, ep20 (best val_bce) | test | 0.8071 | 0.8068 | 0.8073 | 0.8070 |
| D2. unfrozen + WF2 decoder, ep8 (best val_bce) | val | 0.8905 | 0.8905 | 0.8904 | 0.8905 |
| D2. unfrozen + WF2 decoder, ep8 (best val_bce) | test | 0.8310 | 0.8304 | 0.8312 | 0.8307 |
| F. unfrozen S2L1C-13, ep7 (best val_bce) | val | 0.8856 | 0.8857 | 0.8851 | 0.8856 |
| F. unfrozen S2L1C-13, ep7 (best val_bce) | test | 0.8356 | 0.8352 | 0.8359 | 0.8354 |
| F. unfrozen S2L1C-13, ep29 | val | 0.8933 | 0.8932 | 0.8930 | 0.8932 |
| F. unfrozen S2L1C-13, ep29 | test | 0.7999 | 0.7993 | 0.8002 | 0.7996 |

**Retention curve：IoU-vs-retention 曲線下面積（normalized AUC）/ retention 50% 時的 IoU**

| model | subset | DST | Aleatoric | Epistemic | A+E |
|---|---|---|---|---|---|
| A. EDL baseline | val | 0.9909 / 0.9976 | 0.9908 / 0.9976 | 0.9909 / 0.9976 | 0.9909 / 0.9976 |
| A. EDL baseline | test | 0.9688 / 0.9828 | 0.9688 / 0.9828 | 0.9688 / 0.9828 | 0.9688 / 0.9828 |
| B. TerraMind frozen | val | 0.9872 / 0.9967 | 0.9866 / 0.9967 | 0.9870 / 0.9967 | 0.9867 / 0.9967 |
| B. TerraMind frozen | test | 0.9665 / 0.9849 | 0.9665 / 0.9849 | 0.9665 / 0.9849 | 0.9665 / 0.9849 |
| C. frozen + WF2 decoder | val | 0.9890 / 0.9973 | 0.9888 / 0.9973 | 0.9889 / 0.9973 | 0.9889 / 0.9973 |
| C. frozen + WF2 decoder | test | 0.9552 / 0.9738 | 0.9550 / 0.9738 | 0.9551 / 0.9738 | 0.9551 / 0.9738 |
| D1. unfrozen, ep8 | val | 0.9921 / 0.9980 | 0.9921 / 0.9980 | 0.9921 / 0.9980 | 0.9921 / 0.9980 |
| D1. unfrozen, ep8 | test | 0.9602 / 0.9761 | 0.9600 / 0.9761 | 0.9601 / 0.9761 | 0.9601 / 0.9761 |
| D1. unfrozen, ep20 (best val_bce) | val | 0.9914 / 0.9978 | 0.9913 / 0.9978 | 0.9914 / 0.9978 | 0.9913 / 0.9978 |
| D1. unfrozen, ep20 (best val_bce) | test | 0.9554 / 0.9644 | 0.9554 / 0.9644 | 0.9554 / 0.9644 | 0.9554 / 0.9644 |
| D2. unfrozen + WF2 decoder, ep8 (best val_bce) | val | 0.9913 / 0.9981 | 0.9913 / 0.9981 | 0.9913 / 0.9981 | 0.9913 / 0.9981 |
| D2. unfrozen + WF2 decoder, ep8 (best val_bce) | test | 0.9576 / 0.9747 | 0.9575 / 0.9747 | 0.9576 / 0.9747 | 0.9576 / 0.9747 |
| F. unfrozen S2L1C-13, ep7 (best val_bce) | val | 0.9923 / 0.9977 | 0.9923 / 0.9977 | 0.9923 / 0.9977 | 0.9923 / 0.9977 |
| F. unfrozen S2L1C-13, ep7 (best val_bce) | test | 0.9709 / 0.9898 | 0.9708 / 0.9898 | 0.9708 / 0.9898 | 0.9708 / 0.9898 |
| F. unfrozen S2L1C-13, ep29 | val | 0.9895 / 0.9949 | 0.9895 / 0.9949 | 0.9895 / 0.9949 | 0.9895 / 0.9949 |
| F. unfrozen S2L1C-13, ep29 | test | 0.9698 / 0.9903 | 0.9698 / 0.9903 | 0.9698 / 0.9903 | 0.9698 / 0.9903 |

### 3.3 訓練曲線（`lightning_logs/version_0/metrics.csv`，val 17 張）

**注意：訓練時記錄的 `val_iou_land_water water` 不是 IoU 的正確估計，而且會隨 batch size 改變，A（batch 64）與其他組（batch 16）不能比**（見 §3.4）。
這個欄位只能看同一組內的趨勢；跨組比較請用 `val_bce`、§3.4 的 pooled IoU 或 §3.1 的整圖推論。圖：`artifacts/runs/compare/curves_ABDF.png`。

| model | best val_bce：epoch / val_bce / IoU water* | best IoU water*：epoch / IoU | final：epoch / val_bce / IoU* | epoch 0 IoU* |
|---|---|---|---|---|
| A. EDL baseline（batch 64） | 8 / 0.0621 / 0.7030 | 3 / 0.7054 | 29 / 0.0777 / 0.6737 | 0.6298 |
| B. TerraMind frozen | 3 / 0.0645 / 0.5132 | 2 / 0.5153 | 29 / 0.0765 / 0.5012 | 0.4973 |
| C. frozen + WF2 decoder | 21 / 0.0674 / 0.5154 | 20 / 0.5155 | 29 / 0.0690 / 0.5120 | 0.4802 |
| D1. unfrozen | 20 / 0.0536 / 0.5296 | 8 / 0.5311 | 29 / 0.0543 / 0.5240 | 0.4852 |
| D2. unfrozen + WF2 decoder | 8 / 0.0545 / 0.5236 | 4 / 0.5321 | 29 / 0.0555 / 0.5192 | 0.4964 |
| F. unfrozen S2L1C-13 | 7 / 0.0514 / 0.5253 | 17 / 0.5329 | 29 / 0.0544 / 0.5291 | 0.5121 |

\* 逐 batch 平均的 IoU，見 §3.4。解凍的三組 epoch 0 的 val_bce 很高（D1 0.96、D2 0.39、F 0.29），epoch 1 起恢復。

### 3.4 訓練時 val IoU 的偏差與 pooled IoU

`EDL_ML4FloodsModel.validation_step` 沿用 ml4floods 的寫法：每個 val batch 用 `metrics.calculate_iou` 算一次 IoU，再由 Lightning 依 batch size 加權平均。兩個問題：

1. `calculate_iou`（`ml4floods/models/utils/metrics.py`）在 TP 加 1e-6、FP / FN 用總和減 TP 算出，所以**整個 batch 沒有水時**，全對會得到 −1、只要誤報 1 個像素就得到 0。
2. 小水量的 batch 稍錯就 IoU 很低，卻與水多的 batch 等權重。結果這個值**依 batch size 而變**：batch 越大，無水 batch 越少、越接近 pooled 值。
   以真實 val（3524 tile，1817 張有水）計算，完美模型在 batch 16 只會被記成 0.80（無水 batch 22/221），在 batch 64 記成 0.93（2/56）。

`artifacts/runs/compare/eval_val_iou.py`（不改共用程式碼）用相同的 val dataloader 重算，`logged_bs*` 重現訓練 log 的值（D1 ep8 0.5311、B ep3 0.5132、A ep8 bs64 0.7030 皆與 csv 一致）：

| ckpt | logged，batch 16 | logged，batch 64 | **pooled IoU** | Precision | Recall |
|---|---|---|---|---|---|
| A. EDL baseline ep8 | 0.539 | 0.703 | 0.873 | 0.888 | 0.982 |
| B. frozen ep3 | 0.513 | 0.679 | 0.864 | 0.878 | 0.982 |
| D1. unfrozen ep8 | 0.531 | 0.698 | 0.888 | 0.911 | 0.973 |
| D1. unfrozen ep20 | — | — | 0.887 | 0.908 | 0.975 |
| D2. unfrozen + WF2 decoder ep8 | — | — | 0.883 | 0.900 | 0.980 |
| F. unfrozen S2L1C-13 ep7 | — | — | 0.892 | 0.914 | 0.974 |
| F. unfrozen S2L1C-13 ep29 | — | — | **0.893** | 0.916 | 0.973 |

同一個 A 模型在 batch 16 記成 0.539、batch 64 記成 0.703，所以訓練曲線上「A 0.70 vs 其他 0.51」的差距是 batch size 造成的假象。
pooled IoU 與整圖推論（§3.1 val OVERALL）一致：A 0.873 vs 0.8745；B 0.864 vs 0.847，TerraMind 在整圖推論時多掉約 1.7 pt，可能來自整圖 tiling / padding。
`EDL_SAR_Unet.on_validation_epoch_end` 已有正確的 epoch 級 `val_Global_iou_*`，光學 EDL 基底類別沒有；修正共用 `edl.py` 前需與 SAR 支線協調。

### 3.5 Checkpoint epoch 對照（B、C 各取 epoch 3 與 epoch 21，只跑 inference + eval）

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
  四種 uncertainty（DST / aleatoric / epistemic / a+e）在各模型上的排序效果幾乎一樣，這是 EDL head 的性質，與 encoder 無關。
- 訓練曲線上 tile 級 IoU 差 0.19（0.51 vs 0.70）是指標的假象（§3.4）：同為 batch 16 時 A 0.539、B 0.513；pooled IoU A 0.873、B 0.864。
  先前版本把這個差距歸因於「256 tile 對 ViT 的邊界效應」，**這個解釋是錯的**。

**B vs C（decoder 用 WF2 預訓練有沒有幫助）**

- 同 epoch 比較（§3.5）差異在雜訊內：epoch 3 時 B 0.847/0.817、C 0.842/0.816（val/test）；epoch 21 時 B 0.835/0.800、C 0.856/0.786。
  decoder 預訓練沒有帶來一致的增益；C 訓練起點反而更低，因為預訓練 decoder 期待的是 UNet 特徵，接上隨機 lateral 的 ViT 特徵後要重新適應。
- C 的 best-val ckpt（epoch 21）在 val 上是 A / B / C 中僅次於 A 的（0.856），但 test 掉到 0.786，precision 全面下降，不是單一事件造成；B 的 epoch 21 也一樣（test 0.800）。
  **凍結 encoder 的兩組都在後期 epoch 出現 val 進步、test 退步**，用 17 張 val 的 `val_bce` 挑 ckpt 會偏向晚期 ckpt 而在 test 吃虧。
- 所以「B 落後 A 的主因是 decoder 初始化」這個假設**不成立**；主因是 encoder 凍結，以及 A 的 encoder + decoder 是整體從 WF2 起跑。

**B / A vs D1（解凍 encoder）**

- 解凍後 val 明顯進步，且超過 A：val OVERALL 0.889–0.891 vs A 0.875，val_bce 0.054 vs 0.062，pooled IoU 0.888 vs 0.873。
- 但 **test 沒有超過 A，連 B 都沒超過**：test OVERALL D1 ep8 0.808 / ep20 0.809 vs A 0.829、B 0.817。差距集中在水量最大的幾個事件，主要是多報水（§3.1 per-event）。
  用每個事件等權重的 AVERAGE，D1 ep8 0.827 與 A 0.830 幾乎打平。
- best-val 的 ep20 在 test AVERAGE 比 ep8 低 2.4 pt（0.803 vs 0.827）：再次顯示 17 張 val 挑 ckpt 不可靠。
- Uncertainty 略差於 A：test retention AUC 0.955–0.960 vs 0.969，test PAvPU 0.807–0.828 vs 0.839。

**D1 vs D2（解凍時 decoder 用 WF2 初始化）**

- 互有勝負、差距 ≤ 1 pt（test OVERALL D2 0.812 vs D1 0.808；AVERAGE 0.815 vs 0.827），與 B vs C 的結論相同：WF2 decoder 初始化沒有一致的幫助。

**D1 vs F（S2L1C 13 波段，與資料模態一致）**

- val 是目前最好的：OVERALL 0.894–0.895、pooled 0.892–0.893。
- **test OVERALL F ep29 0.826，追平 A（0.829，−0.2 pt）**，是 TerraMind 各組中最接近 A 的，比 D1 高約 1.9 pt，主要來自 precision（0.886 vs 0.874，少報水），大事件上追回大部分差距。
- 但 test AVERAGE F 0.807 反而低於 D1 ep8（0.827）與 A（0.830）：F 在幾個小事件明顯退步（§3.1 per-event）。
- 這次是晚期 ckpt（ep29）在 test 較好，best-val 的 ep7 test OVERALL 只有 0.807：挑 ckpt 的方向與 D1 相反，進一步說明 val 無法可靠地選 ckpt。
- Uncertainty：test retention AUC F 0.970–0.971，**略高於 A（0.969）**，是各組最好的；但 F ep29 的 test PAvPU 0.800 是各組最低。兩個不確定性指標的結論不一致。

**整體**

- 完全不微調 TerraMind encoder、decoder 從零訓練，就能達到 fine-tune 過的 UNet baseline 97–99% 的整圖 IoU，uncertainty 品質相同。
- 解凍 encoder 後，val 一致超過 A，但 test 沒有。改成與資料一致的 S2L1C 全波段後，test pixel 加總 IoU 追平 A，per-event 平均仍輸約 2.3 pt。
  **目前沒有任何 TerraMind 設定在 test 的 OVERALL 與 AVERAGE 上同時超越 A。**
- val 與 test 各只有 17 / 18 張影像，1–2 pt 的差距、以及 val/test 方向相反的現象，都可能落在評估雜訊內。要下更強的結論，需要更大的評估集、跨事件交叉驗證，或多個 seed。
- 下一個實驗 E：F 加上 TiM（先從 S2L1C 生成 LULC token 再一起編碼，argmax 解碼），回答「生成的土地覆蓋先驗能否減少大事件的多報水」，與 F 做單變因對照。

## 5. 執行紀錄與注意事項

- 兩次訓練（B、C）都遇到**另一台機器對同一 experiment 目錄啟動同一 config** 的情況（B：22:00–23:00；C：00:04–03:18），造成 `-v1.ckpt` / `lightning_logs/version_1` / 第二個 wandb run 與 stdout log 混寫。已依 `-v1` 後綴與 mtime 分辨歸屬，把另一台的檔案移到 `artifacts/runs/duplicate_run_220013/`、`artifacts/runs/duplicate_run_000449/`；本文件所有數字皆來自本機的 run。
  同一 NAS 上多機訓練，`experiment_name` 必須不同。
- `analysis_S2.py` / `compute_pavpu.py` 的輸出路徑只有 model_type 沒有 subset，val 與 test 會互相覆蓋；本次用 `analysis_S2/{val,test}/<model_type>/` 分開。
- D / F 在另一台機器（RTX 4090 24 GB）上訓練，batch 16、`num_workers 16`、fp32，依序執行不併行。D1 / D2 / F 每組 30 epoch 約 5.5–7.5 小時，顯存約 8 GB。
  該容器的 `UV_PROJECT_ENVIRONMENT` 預設指向 image 自帶的 `/opt/venv`（不是專案環境），需改用專案的 `.venv`（Python 3.10）；`compose.yml` 的 `shm_size` 曾誤拼為 `shm_sizes`，`/dev/shm` 只有 64 MB，修正後 DataLoader 才能正常使用多 worker。
- E（TiM）速度基準：3.05 it/s、顯存峰值 9.3 GiB（F 為 5.21 it/s、6.0 GiB），30 epoch 約 13 小時；凍結的生成器也會存進 ckpt，每個約 2 GB。
- 推論 / 分析腳本與產生表格的程式在 `artifacts/runs/compare/`（`run_pipeline.sh`、`ckpt_control.sh`、`eval_D.sh`、`summarize.py`、`curves.py`、`plot_curves.py`、`eval_val_iou.py`、`bench_train_step.py`），未納入 git。
