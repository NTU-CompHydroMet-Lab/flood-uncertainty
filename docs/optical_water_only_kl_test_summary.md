# Water-only EDL 測試摘要

**Mask、DataLoader 對齊、loss 數值與梯度都有通過紀錄。** 以下整理既有結果。實際影像檢查涵蓋訓練事件 `EMSR260_01CICOGNARA_DEL_MONIT01_v2`。

## 實驗結果圖

### 圖 1　哪些像素接受 KL 監督？

![整場 Sentinel-2 影像、水體標籤、KL 遮罩及疊圖](../artifacts/kl_masks/worldfloods_v2/train/preview/EMSR260_01CICOGNARA_DEL_MONIT01_v2_overview.png)

黃線是標註範圍（AOI），粉紅色是接受 KL 監督的位置。整場共 890,240 像素，其中 56,487 個同時符合「輸入有效、位於 AOI 內、水體標籤為 invalid」，與存檔逐像素一致。這個規則標出可能難判讀的區域，個別像素的成因仍需人工確認。

### 圖 2　旋轉後，影像和標籤是否對齊？

![第 11 個 patch，來源 TIFF 與 DataLoader 旋轉後的 RGB、水體 GT、KL 及疊圖](../artifacts/kl_masks/worldfloods_v2/train/dataloader_check/EMSR260_01CICOGNARA_DEL_MONIT01_v2_20261123T071156164990Z/patch_011.png)

上排是來源 TIFF，下排是 DataLoader 輸出。第 11 個 patch 旋轉了 180°，含 6,046 個水體像素與 2,379 個 KL 像素。按相同角度對齊後，兩通道 GT（參考標籤）與 KL 的像素差異都是 0。

### 圖 3　如何對照遮罩與模型不確定性？

![固定 patch 的雲與水體標籤、水體機率、不確定性及 KL 邊界](../artifacts/models/edl_water_kl/20261123T162613333376Z/train_probe/epoch_000.png)

右下圖的粉紅線是 KL 邊界，色階是像素的不確定性 $u=2/(\alpha_{\rm land}+\alpha_{\rm water})$。分子 2 是類別數，分母是 land、water 各自的「1 + evidence」相加。兩類 evidence 都是 0 時，$u=2/(1+1)=1$，代表最大不確定性。這是 epoch 0 結束時的基準圖，當時退火權重 $\beta=0$，新增 KL 對這一輪的訓練梯度貢獻為 0。

### 圖 4　第一個 epoch 的 loss 和水體 IoU

![本機訓練紀錄中的 loss、有效標籤水體 IoU 及固定 patch 的平均不確定性](../artifacts/models/edl_water_kl/20261123T162613333376Z/train_summary.png)

左圖的總 loss 與有效標籤 EDL 重合（0.147732），因為此時加權後的 invalid KL 為 0。中圖是有效訓練標籤上的水體 IoU（0.722）。右圖的橫軸 0 是預訓練權重，1 是完成第一個 epoch。評估新增 KL 對不確定性的影響，需要比較 $\beta>0$ 的後續 epoch 與原版基線。

## 測試結果

| 項目 | 已確認的結果 |
|---|---|
| Mask | 整張影像共 890,240 像素。依「有效輸入且在 AOI 內且 water GT 為 invalid」重算，56,487 個 KL 像素與 TIFF **逐像素完全一致**。AOI 外、無效輸入、有效標籤上的 KL 均為 0。[稽核報告](../artifacts/kl_masks/worldfloods_v2/train/preview/EMSR260_01CICOGNARA_DEL_MONIT01_v2_audit.json) · [整張疊圖](../artifacts/kl_masks/worldfloods_v2/train/preview/EMSR260_01CICOGNARA_DEL_MONIT01_v2_overview.png) |
| DataLoader | 門檻 0.8 保留 14／20 個 patch，共 29,379 個 KL 像素，與保留視窗的 TIFF 計數一致。依序畫前 16 張，實際 14 張**全部通過**。旋轉、翻轉後兩通道 GT 和 KL 的像素誤差均為 0，也涵蓋邊界補零。[14 張的報告](../artifacts/kl_masks/worldfloods_v2/train/dataloader_check/EMSR260_01CICOGNARA_DEL_MONIT01_v2_20261123T071156164990Z/report.json) · [第 11 張圖，有水也有 KL](../artifacts/kl_masks/worldfloods_v2/train/dataloader_check/EMSR260_01CICOGNARA_DEL_MONIT01_v2_20261123T071156164990Z/patch_011.png) |
| Loss | 真實標籤搭配固定隨機 logits，包含 63,157 個有效像素、2,379 個 KL 像素。關閉 KL 時與原版 EDL 相同。epoch 5 的總 loss **1.125853** 與「原版 EDL + 0.5 × PyTorch Dirichlet KL」一致。新增梯度只出現在 KL 選區。純 KL、全忽略（loss 和梯度皆為 0）、epoch 0 退火也通過。[CPU 測試腳本](/tmp/water-only-loss-check.EAhALGlM/check.py) |

產生器另有涵蓋 NaN、AOI 空洞、遮罩位移與斷點續跑的[單元測試](../tests/test_generate_water_only_kl_masks.py)。

實際訓練已完成 epoch 0 並進入 epoch 1。epoch 0 的 KL 權重為 0，總 loss 和有效標籤 EDL 都是 0.147732，符合退火設定。[訓練紀錄](../artifacts/models/edl_water_kl/20261123T162613333376Z/logs/version_0/metrics.csv)

## 原版與新版的訓練 loss

**原版只在有效 land／water 標籤上算 EDL。新版沿用這一項，另外對選中的 invalid 像素算 KL，把兩類 evidence 都往 0 拉。** 每個像素依原始 water GT 分流。

| Water GT | KL mask | 原版 | 新版 |
|---|---|---|---|
| 1（land）或 2（water） | 0 或 1 | 有效標籤 EDL | 同一個 EDL |
| 0（invalid） | 1 | 忽略 | 只算完整 evidence 的 KL |
| 0（invalid） | 0 | 忽略 | 忽略，梯度為 0 |

### 原版 loss 怎麼讀？

$$
L_{\mathrm{old}} =
w_{\mathrm{task}}\times
\frac{\displaystyle\sum_{i\in V}
  \left[w_i F_i + \beta D(\tilde{\alpha}_i)\right]}
{\lvert V\rvert + 10^{-6}}
$$

**白話讀法**　 先算每個有效像素的「分類損失 + KL 懲罰」，把它們加起來，除以有效像素數加上一個極小值，最後乘上任務權重。

~~~text
原版 loss = 任務權重 × 分子 / 分母
分子 = 對每個有效像素，加總 (該像素的類別權重 × 分類損失 + 退火權重 × 錯誤類別的 KL)
分母 = 這個 batch 的有效像素數 + 0.000001
~~~

| 式中的符號 | 怎麼讀、代表什麼 |
|---|---|
| $L_{\mathrm{old}}$ | 原版在一個 batch 上回傳的訓練損失，是一個純量。 |
| $i$、$V$、$\sum_{i\in V}$ | $i$ 是一個像素的索引。$V$ 收集整個 batch 中標成 land 或 water 的像素。$\sum_{i\in V}$ 就是走過這些像素，把方括號內的值各加一次。 |
| $\lvert V\rvert$、$10^{-6}$ | 集合兩旁的直線表示「有幾個像素」。$10^{-6}=0.000001$ 是避免除以 0 的保護值。以先前測試為例，分母是 $63{,}157+0.000001$。 |
| $w_{\mathrm{task}}$ | 整個 water-only 任務的權重，本次是 1，乘在整個有效像素平均外面。 |
| $w_i F_i$ | $F_i$ 是像素 $i$ 的分類損失，包括預測機率的平方誤差與變異數。$w_i$ 依真實標籤取值，land 為 1、water 為 6，乘法只作用在 $F_i$。 |
| $\beta$（beta） | 決定 KL 懲罰在這個 epoch 占多少權重。本次從 0 線性增加，epoch 5 是 0.5，epoch 10 起是 1。 |
| $D(\tilde{\alpha}_i)$ | $D$ 是 KL 散度。對任意兩類參數向量 $a$，它衡量以 $a$ 為參數的 Dirichlet 分布相對於均勻先驗 $\mathrm{Dir}(1,1)$ 的差異。$\tilde{\alpha}_i$ 上方的波浪號表示已將**真實類別**的參數設回 1，因此這一項只懲罰**錯誤類別**的 evidence。 |

**代入一個水體像素。** 假設 epoch 5、真實標籤為 water，模型給的兩類參數是 $(4,7)$，順序為 (land, water)。原版算 KL 時把 water 的 7 設回 1，得到 $(4,1)$。因此這個像素加進分子的值是「$6\times F_i+0.5\times D((4,1))$」。水體權重 6 只放大分類損失。

### 單一像素的分類損失與 evidence 怎麼算？

$$
F_i = \sum_{c\in\{\mathrm{land},\,\mathrm{water}\}}
\left[(t_{ic}-p_{ic})^2+\frac{p_{ic}(1-p_{ic})}{S_i+1}\right]
$$

**白話讀法**　 固定像素 $i$，對 land 和 water 各算一次，再相加。括號內第一項是「真實標籤與預測機率差多少」的平方，第二項是該類機率在 Dirichlet 分布下的變異數。

| 式中的符號 | 怎麼從模型或標籤得到 |
|---|---|
| $c$、$t_{ic}$ | $c$ 指 land 或 water。像素 $i$ 的真實類別等於 $c$ 時，$t_{ic}=1$，其餘為 0。例如 water 的標籤向量是 $(0,1)$。 |
| $z_{ic}$、$e_{ic}$ | $z_{ic}$ 是模型對該類輸出的原始分數（logit）。$e_{ic}=\max(0,z_{ic})$ 是 evidence，負分數經 ReLU 變成 0。 |
| $\alpha_{ic}$、$\alpha_i$ | $\alpha_{ic}=1+e_{ic}$ 是該類的 Dirichlet 參數。少了下標 $c$ 的 $\alpha_i$ 指同一像素的完整向量 $(\alpha_{i,\mathrm{land}},\alpha_{i,\mathrm{water}})$。 |
| $S_i$、$p_{ic}$ | $S_i$ 是兩類 $\alpha$ 的總和，$p_{ic}=\alpha_{ic}/S_i$ 是該類的預測機率。例如 $(4,7)$ 的總和是 11，預測水體機率是 $7/11$。 |
| $\tilde{\alpha}_{ic}$ | 原版 KL 用 $\tilde{\alpha}_{ic}=1+(1-t_{ic})(\alpha_{ic}-1)$。代入真實類別的 $t_{ic}=1$ 得到 1，代入另一類的 $t_{ic}=0$ 則保留 $\alpha_{ic}$。 |

### 新版多算了哪些像素？

$$
L_{\mathrm{new}} = L_{\mathrm{old}}+\beta L_{\mathrm{invalid}},
\qquad
L_{\mathrm{invalid}}=
\begin{cases}
\displaystyle\frac{\sum_{i\in M}D(\alpha_i)}{\lvert M\rvert}, & \lvert M\rvert>0\\
0, & \lvert M\rvert=0
\end{cases}
$$

**白話讀法**　 新版 loss 等於「原版 loss + 退火權重 × 選中 invalid 像素的平均 KL」。這些像素沒有 land／water 分類目標，只對完整的兩類 evidence 算 KL。

| 式中的符號 | 怎麼讀、代表什麼 |
|---|---|
| $M$、$\lvert M\rvert$ | $M$ 只收集 water GT 為 0 **且** KL mask 為 1 的像素。$\lvert M\rvert$ 是這一組的像素數。先前測試中有 2,379 個，因此新增項用 2,379 作分母。 |
| $L_{\mathrm{invalid}}$、$D(\alpha_i)$ | 對 $M$ 裡每個像素，用完整 $\alpha_i$ 算 KL 再取平均。例如仍用 $(4,7)$ 和均勻先驗比較。當兩類 evidence 都是 0，$\alpha_i=(1,1)$，此時 KL 為 0。若 $M$ 是空集合，此項直接回傳 0。 |
| $L_{\mathrm{new}}$ | 新版在一個 batch 上的總損失。兩組像素分別平均，再加總。如果只有 KL 像素，總損失是 $\beta L_{\mathrm{invalid}}$。如果兩組都空，回傳可反向傳播的 0。 |

**用已通過的 CPU 測試驗算。** 在 epoch 5，原版 loss 是 1.02331984，選中 invalid 像素的平均 KL 是 0.20506699。代入新版得到 $1.02331984+0.5\times0.20506699\approx1.125853$，與測試輸出一致。

### 參數與本次設定

| 程式參數 | 本次值與作用 |
|---|---|
| `task_mode`、`num_classes` | `water_only`、`1`，代表一個二元任務。每個像素仍有 land、water 兩個 evidence 通道，傳入 `kl_divergence` 的 `num_classes=2`。 |
| `logits`、`device` | 浮點張量 `[B,2,H,W]`，通道 0 是 land、1 是 water。`B` 是 batch 內 patch 數，`H,W` 是高與寬。此 run 設定 `B=32`、`H=W=256`、FP32。KL 在 `logits.device` 上計算。 |
| `batch["image"]`、`batch["mask"]`、`target` | 影像 `[B,6,H,W]` 先經模型產生 logits。GT `[B,2,H,W]` 的通道順序是 cloud、water。取第二通道得到 `target=[B,1,H,W]`。有效 GT 1、2 分別映射為 one-hot `[1,0]`、`[0,1]`。原版內部 `ignore_index=-1` 對應磁碟上的 invalid 值 0。 |
| `batch["kl_mask"]` | 新版輸入，二元張量 `[B,1,H,W]`。1 表示離線規則選中，0 表示未選或 padding。只有 water GT 同時為 0 才進入 $M$。 |
| `pos_weight` → `pos_weight_problem` → `class_weight` | `[[1,6]]` 對應 land 權重 1、water 權重 6，只乘在 $F_i$（誤差與變異數）。分母仍是有效像素數加 $10^{-6}$，兩個 KL 項都不乘類別權重。 |
| `weight_problem` | `[1.0]` 是水體任務的 $w_{\mathrm{task}}$，乘整個有效標籤 EDL。新增的 invalid KL 直接乘 $\beta$。 |
| `epoch_num`、`annealing_mode`、`annealing_step` | 訓練傳入從 0 起算的 `current_epoch`。本次為 `linear`、`10`，兩個 KL 共用 $\beta=\min(1,\max(0,\mathrm{epoch}/10))$。也就是把 epoch 除以 10，再將結果限制在 0 到 1。epoch 0 是 0，epoch 5 是 0.5，從 epoch 10 起維持 1。 |
| `annealing_coefficient` | 設定值 `0.5` 只在 `fixed` 模式用作固定 $\beta$。本次 epoch 5 的 0.5 來自線性排程 $5/10$。 |
| `accumulate_epoch` | `compute_loss` 預設 `False`，訓練傳 `True` 以累計各項的像素數與總和。回傳給反向傳播的都是同一個 batch 純量 loss。 |

讀 `metrics.csv` 時，`train_valid_edl_epoch` 已含有效標籤的 KL，`train_invalid_kl_epoch` 是尚未乘 $\beta$ 的 invalid KL 平均，`train_kl_pixels_epoch` 是該 epoch 選中的像素數。`train_loss_epoch` 將前兩項依上式相加。這項改動用於訓練，驗證沿用原版的有效標籤指標。[原版訓練入口](../flood_uncertainty/models/edl.py) · [EDL 公式與退火](../flood_uncertainty/losses/edl_loss.py) · [新版 loss](../flood_uncertainty/models/edl_water_kl.py) · [KL 實作](../mlguess/torch/class_losses.py) · [本次設定](../artifacts/models/edl_water_kl/20261123T162613333376Z/train_manifest.json)

## 從專案根目錄重跑

```bash
# 依序畫前 16 個保留 patch，同時比對來源 TIFF 和 DataLoader
uv run python -m scripts.preprocess.check_water_only_kl_dataloader_1stpatch --num-patches 16

# 重跑 loss 數值與梯度檢查，腳本目前存於 /tmp
PYTHONPATH="$PWD" /opt/venv/bin/python /tmp/water-only-loss-check.EAhALGlM/check.py
```

**若只補一項**，建議用第 11 張 patch 做一次模型 forward，在相同 logits、KL 權重大於 0 時，比較真實遮罩和全零遮罩的 loss 梯度。要求選區外的梯度差為 0，選區內與 PyTorch 參考梯度的方向和數值一致。這會把既有 loss 測試延伸到真實模型輸出。
