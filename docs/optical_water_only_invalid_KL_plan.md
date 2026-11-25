# Water-only EDL：難判讀像素的 KL 監督

**決策：先離線產生 KL mask，再由 Dataset 與影像、原始 GT 一起讀取並同步增強。**

## 選取規則

```text
KL_mask = input_valid AND inside_AOI AND (water_GT == 0)
```

- 水陸 GT：`0=invalid`、`1=land`、`2=water`。
- 有效輸入從原始影像判斷，排除 nodata、NaN、padding；在 normalization 前建立。
- AOI 取同事件 GeoJSON 中 `w_class="area_of_interest"` 的多邊形。缺少 AOI 的事件不以影像邊界代替：仍以有效 land／water 標註訓練，KL mask 視為全 0。
- 此交集是「難判讀區」的研究代理定義；目前確認雲遮蔽會造成 invalid，其他歧義原因尚未完整驗證。

## Loss

有效 land／water 使用既有 EDL loss；`KL_mask=1` 只計算完整 alpha 的 KL；其餘 invalid 忽略。

```text
alpha = evidence + 1
L_KL = KL(Dirichlet(alpha) || Dirichlet([1, 1]))
L_total = mean_valid(L_EDL) + lambda_inv(t) * mean_KL(L_KL)
```

兩組分開平均，空集合貢獻為 0。invalid 的 KL 不依類別修正 alpha。

採用 `lambda_inv = 0.02` 固定值，從 epoch 0 開始；`L_EDL` 內建的 misleading-evidence KL 維持 `beta(t)` 退火，與 baseline 相同。Ablation 比較 `lambda_inv = 0`、`beta(t)`、`0.02 * beta(t)` 與 `0.02`，設定在 `configurations/edl_water_kl_ablation/`，以 `scripts/train/run_edl_water_kl_ablation.sh` 依序執行。

## 資料流程

```text
離線：S2 + GT + AOI → 對齊 GT 網格 → 產生二元 KL GeoTIFF → 統計與疊圖檢查
訓練：讀取相同 window 的 S2、GT、KL mask
    → 暫時合併 [cloud_GT, water_GT, KL_mask]
    → 與影像執行同一次旋轉、翻轉
    → 拆回兩通道 GT 與獨立 KL mask
    → image 送模型；GT 與 KL mask 供 loss 使用
```

保留原始 GT；KL mask 另存，padding 填 `0`。訓練前確認 AOI 覆蓋、像素對齊、無效輸入排除，以及實際保留 patch 的 KL 像素數量。

評估有效標註區的準確度與不確定性品質；KL 區 evidence 下降本身不代表可靠性改善。

已提供離線產生與疊圖檢查程式，使用方式見 [離線產生 KL mask](generate_water_only_kl_masks.md)；預設只在包含候選像素的區塊讀取 S2。Dataset（[worldfloods_kl.py](../flood_uncertainty/data/worldfloods_kl.py)）、loss（[edl_water_kl.py](../flood_uncertainty/models/edl_water_kl.py)）與訓練腳本（[train_edl_water_kl.py](../scripts/train/train_edl_water_kl.py)）已接入，並以 smoke test 驗證；修改與驗證紀錄見 [審查待處理清單](water_only_kl_review_checklist.md)。產生器的 25 個合成資料測試，以及 loss 與 Dataset 的 25 個測試（[tests/test_edl_water_kl.py](../tests/test_edl_water_kl.py)），已於 2026-11-24 全部通過。
