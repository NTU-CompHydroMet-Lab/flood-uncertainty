# 離線產生 Water-only KL mask

程式：`scripts/preprocess/generate_water_only_kl_masks.py`。

```text
KL_mask = input_valid AND inside_AOI AND (water_GT == 0)
```

## 判定方式

- GT 固定為 WorldFloods v2 的兩通道格式，讀取第 2 通道 water GT：`0=invalid, 1=land, 2=water`。直接讀原始值，保留 TIFF nodata=0 對應的 invalid 標籤。
- 有效輸入檢查原始 S2 前 13 個光譜通道，與原始 GT 的產生流程一致；各通道須通過來源有效性 mask、全部為有限值，且不能全部為 0。QA60／雲機率等附加通道不參與判定。
- 來源有效性 mask 遵循 Rasterio 的 nodata／dataset mask 規則。若 S2 宣告 nodata=0，任一光譜通道被標為 nodata 都會排除；若沒有此宣告，單一通道為 0 不會直接排除。[Rasterio 說明](https://rasterio.readthedocs.io/en/stable/topics/masks.html)
- AOI 只取同事件 GeoJSON 的 `w_class="area_of_interest"`，轉換至 GT CRS，保留多邊形孔洞並沿用原始 GT 的 `all_touched=True`。
- S2 與 GT 的 CRS、尺寸、transform 必須一致；不自動重採樣。缺少配對檔、無效幾何、不合法 GT 值或 AOI 完全未覆蓋 GT 都會報錯。
- 缺少 AOI 預設報錯；明確指定 `--skip-missing-aoi` 才會記錄並跳過，不產生該事件的 mask。若該事件已有舊 mask，仍會報錯，需改用新的輸出目錄，避免誤用舊結果。

此規則是「難判讀區」的研究代理定義，沒有額外判斷厚雲、混合像素或標註者分歧。

## 輸入與輸出

```text
<data-root>/<split>/S2/<event>.tif
<data-root>/<split>/gt/<event>.tif
<data-root>/<split>/floodmaps/<event>.geojson

<output-root>/<split>/kl_mask/<event>.tif
<output-root>/kl_mask_summary_<UTC時間>.csv
```

掃描指定 split 的 `gt/` 目錄，以相同檔名配對 S2；不修改原始資料或訓練 split JSON。`--events` 可限定事件名稱。

輸出是單通道 `uint8` GeoTIFF，與 GT 共用 CRS、transform、尺寸；`1=只計算KL`、`0=不選取`。兩者都是有效 mask 值，因此不宣告 nodata。之後 Dataset 超出影像範圍的 padding 仍應填 `0`。

預設以 1024 × 1024 window 處理；`--block-size` 可調整記憶體用量。完整寫入後才發布該事件的 TIFF；預設遇到既有輸出會報錯。`--skip-existing` 可保留完成檔並續跑，`--overwrite` 則重新產生並替換既有結果，兩者不能同時使用。

## 候選區讀取

預設先讀 GT 並光柵化 AOI，只有存在候選像素的 window 才讀取 S2：

```text
candidate = inside_AOI AND (water_GT == 0)
沒有候選像素 → 寫入全 0 mask，跳過 S2 像素讀取
有候選像素   → 讀取原本的 13 bands，KL_mask = candidate AND input_valid
```

輸入檔仍會開啟以檢查 CRS、尺寸與 band 數，但不會為空候選區讀取 S2 像素陣列。候選 window 仍讀取完整 window；沒有改變有效輸入的定義或 KL 選取公式。

加上 `--full-input-statistics` 可切回全圖掃描，每個 window 都讀取 S2，保留原版全圖有效輸入統計。兩種模式的 KL mask 像素應完全一致，差別在讀取量與統計範圍。

## 執行

在專案根目錄先處理一個已確認具有 AOI 的事件：

```bash
uv run python scripts/preprocess/generate_water_only_kl_masks.py \
  --data-root /home/NAS/house/chlunchen-10030/datasets/cj/cj/data/worldfloods_v2/data \
  --output-root artifacts/kl_masks/worldfloods_v2_preview \
  --splits test \
  --events EMSR466_AOI01_DEL_PRODUCT
```

產生後，以同一事件檢查套疊：

```bash
uv run python -m scripts.preprocess.preview_water_only_kl_mask \
  --data-root /home/NAS/house/chlunchen-10030/datasets/cj/cj/data/worldfloods_v2/data \
  --mask-root artifacts/kl_masks/worldfloods_v2_preview \
  --split test \
  --event EMSR466_AOI01_DEL_PRODUCT
```

輸出到 `<mask-root>/test/preview/`：

- `<event>_overview.png`：全圖概覽，白色虛線框標出局部放大位置。
- `<event>_detail.png`：預設 512 × 512 原始像素的局部圖，不對來源陣列降採樣。
- `<event>_audit.json`：全解析度逐像素檢查結果。

兩張圖均包含 **S2 RGB、water GT、儲存後 KL mask、RGB＋KL 疊圖**。黃色線是 AOI 邊界，桃紅色是 KL 像素。四格使用相同來源像素座標；概覽採相同 nearest-neighbor 取樣網格，局部圖用來查看細節。

自動放大優先選擇規則不符的位置，其次是 KL 邊界；沒有選中像素時才顯示影像中心。也可用 `--window COL ROW WIDTH HEIGHT` 指定要檢查的原始像素區域。

預覽會先檢查 S2／GT／KL 的 CRS、transform、尺寸，再重新讀取已存的 mask，以既定公式逐像素比對。正常結果須有 `rule_matches=true`、`mismatch_pixels=0`，且 `kl_outside_aoi_pixels`、`kl_invalid_input_pixels`、`kl_valid_label_pixels` 都是 0。規則不符時仍輸出圖與報告，但以非零狀態結束；網格不一致則直接報錯。

檢查圖上有沒有整體偏移、KL 是否落在 water GT 的 invalid 區、是否越過 AOI 或蓋到無效輸入。自動檢查驗證的是既定規則，仍須人工判讀難辨識區的語意是否合理。

確認單一事件的檢查結果後，產生全部 split。以下指令也可在原目錄續跑：

```bash
uv run python scripts/preprocess/generate_water_only_kl_masks.py \
  --data-root /home/NAS/house/chlunchen-10030/datasets/cj/cj/data/worldfloods_v2/data \
  --output-root artifacts/kl_masks/worldfloods_v2 \
  --splits train val test \
  --skip-missing-aoi \
  --skip-existing
```

需要先處理具備 AOI 的事件時，可加 `--skip-missing-aoi`；CSV 會留下未處理清單。訓練時，缺 AOI 的事件仍會納入，KL mask 視為全 0，只計算有效 land／water 的 EDL；有 AOI 卻沒有 mask 的事件會讓訓練直接報錯，因此須等 generator 完成後再開始訓練。已有的 mask 在訓練前會以和 `--skip-existing` 相同的方式檢查（格式與網格、規則版本、來源路徑、來源修改時間與完成統計），不符時訓練報錯。

已啟動的舊程序不會自動採用新邏輯。要切換版本，先在原終端按 `Ctrl+C`，等舊程序結束後執行上面的續跑指令，沿用 `artifacts/kl_masks/worldfloods_v2`。

## 續跑行為

`--skip-existing` 只認 `<split>/kl_mask/<event>.tif` 的正式輸出；中斷留下的隱藏暫存目錄不算完成。已完成事件會跳過，未完成事件從頭產生，不從中斷的 window 接續。

跳過前檢查 TIFF 格式與網格、規則版本、來源路徑、完成統計，以及來源修改時間是否晚於 mask；來源時間檢查包含 GDAL 列出的附屬檔，例如有效性 mask。這些檢查只讀檔頭與 metadata，不重讀 S2 像素，也不是內容雜湊或逐像素驗證。資訊不符或來源已更新時會報錯，不自動覆寫；確認要重製後，可用 `--events <event>` 搭配 `--overwrite` 處理該事件。

最佳化前產生的 mask 也可沿用，其統計模式視為 `full`。每次執行建立新的 CSV，保留舊報表；沿用事件記為 `status=skipped_existing`，像素統計取自既有 TIFF tags，舊版未記錄的候選區欄位留空。`s2_windows_*`、`s2_pixels_read` 留空，避免把歷史讀取量算成本次；`elapsed_seconds` 為本次檢查既有檔案的時間。

搭配 `--full-input-statistics` 時，只會對新產生的 mask 計算全圖統計；沿用檔案仍保留原統計模式。如需補齊既有檔案的全圖統計，須明確重新產生。

## 檢查

CSV 的 `statistics_mode` 會標示 `candidate`（預設）或 `full`。新產生的 mask 在兩種模式下都精確計算全圖的總像素、AOI、invalid water GT、候選像素與 KL 像素數，滿足：

```text
water_invalid_pixels
  = water_invalid_outside_aoi_pixels + candidate_pixels

candidate_pixels
  = candidate_invalid_input_pixels + kl_pixels
```

- `candidate_pixels`：AOI 內的 invalid water GT。
- `candidate_invalid_input_pixels`：候選像素中，因輸入無效而排除的數量。
- `water_invalid_outside_aoi_pixels`：AOI 外的 invalid water GT，不要求輸入有效。

下列原版欄位只有 `full` 模式會計算；`candidate` 模式在 CSV 留空、TIFF tags 省略，表示未計算，不能解讀成 0：

- `input_valid_pixels`
- `excluded_invalid_input_pixels`
- `excluded_outside_aoi_pixels`

`full` 模式額外保留原版的分類方式，滿足：

```text
water_invalid_pixels
  = excluded_invalid_input_pixels
  + excluded_outside_aoi_pixels
  + kl_pixels
```

`excluded_outside_aoi_pixels` 只計算輸入有效、但在 AOI 外的 invalid water GT，與 `water_invalid_outside_aoi_pixels` 的範圍不同。

讀取量與時間另外記錄為：

- `s2_windows_total`、`s2_windows_read`、`s2_windows_skipped`：處理 window 總數、實際讀取 S2 的數量與跳過數量；總數等於讀取數加跳過數。
- `s2_pixels_read`：讀取 window 的空間像素數總和，不乘 band 數，也不代表磁碟實際傳輸量。
- `elapsed_seconds`：新產生事件從檔案檢查至 mask 完成儲存的耗時；沿用事件僅記錄既有檔案的檢查耗時。

`kl_pixels=0` 是合法結果。CSV 每完成一個事件即寫入；發生錯誤時已完成的事件仍保留。TIFF 以 `generator_version=2` 與 `statistics_mode` 記錄產生方式，KL 判定規則的 `mask_version` 維持 `1`。

合成資料測試涵蓋 nodata／NaN／Inf／全零輸入、AOI 外部與孔洞、CRS 轉換、分塊一致性、GT=0 保留、原始資料保護與失敗時不發布部分檔案；另檢查候選區讀取、兩種模式像素一致、未計算欄位留空，以及續跑時保留舊版 mask、忽略中斷暫存檔、拒絕不相容或來源已更新的結果：

```bash
uv run python -m pytest -c pyproject.toml -q tests/test_generate_water_only_kl_masks.py
```

上述 25 個測試已於 2026-11-24 執行並全部通過；候選區最佳化的加速幅度尚未量測。可用同一事件、相同 `--block-size`，分別在兩個輸出目錄執行預設模式及 `--full-input-statistics`，比較 CSV 的讀取量與時間；計時會受 NAS 負載與快取影響。

預覽程式仍逐像素檢查完整輸入，可用來驗證新產生的 mask，成本不等於候選區產生模式。接入訓練前仍需檢視疊圖；上述程式不變更 window 篩選。實際訓練 windows 保留的 KL 像素數由訓練腳本計算，記錄在 `train_manifest.json` 的 `kl_statistics`（總數取自 mask 的 `kl_pixels` tag，保留數只讀保留下來的 window）。
