# Water-only KL 程式審查：待處理清單

審查日期：2026-11-24。範圍是依 [KL 監督計畫](optical_water_only_invalid_KL_plan.md) 新增的程式。行號以審查當時的版本為準。

審查時只讀檔，未執行程式或測試。核心邏輯（KL mask 規則、loss、同步增強）符合計畫，沒有發現會讓 loss 算錯或 mask 錯位的 bug。

## 需要處理

- [x] **1. 訓練不應悄悄只用已有 mask 的事件**（已採用 B，2026-11-24 修改並驗證：run `20261124T063747420165Z` 為 475 事件、65,582 patch，`events_without_kl` 列出 7 個事件）
  - 位置：[train_edl_water_kl.py:91-101](../scripts/train/train_edl_water_kl.py#L91-L101)
  - 問題：`ready_pairs` 只保留 `kl_mask/` 已有 TIFF 的事件；S2/GT 不存在時也回傳 `None` 略過。缺 AOI 或還沒產生 mask 的事件會整個從訓練消失，包含它們有效的 land/water 標註。目前只會印出訊息，並記在 manifest 的 `excluded_events`。
  - 影響：KL 模型和 baseline 的訓練資料不同，比較結果會混入資料量差異。也和 [generate_water_only_kl_masks.md:98](generate_water_only_kl_masks.md#L98) 的「須補齊或明確排除，才能接入 Dataset」不一致。
  - 二選一：
    - [ ] A. 缺 mask 時報錯並列出事件；只有明確加上 `--allow-missing-kl-events` 才繼續。
    - [x] B. 缺 AOI 的事件仍放進訓練，KL mask 當作全 0，只計算 valid EDL。
  - S2/GT 檔案不存在時應直接報錯，不要回傳 `None`。
  - 處理結果：
    - Dataset 新增 `events_without_kl`，清單內事件回傳全 0 KL mask；不在清單上又缺 mask 檔時仍會報錯。
    - 訓練腳本把每個事件分成「有 mask」、「無 AOI」、「有 AOI 但缺 mask」三類，最後一類會報錯並列出事件。S2、GT 或 floodmap 不存在也會報錯。
    - Manifest 以 `kl_mask_events`、`events_without_kl` 取代 `ready_events`、`excluded_events`。
    - 上一次 run（`20261123T162613333376Z`）只用了 463/475 個事件：7 個無 AOI，另 5 個是 generator 尚未寫完就開始訓練。修改後會用到全部 475 個事件、65,582 個 patch（原本 60,828），其中無 AOI 的 7 個事件佔 2,389 個 patch。

- [x] **2. 在 manifest 記錄實際保留的 KL 像素數**（2026-11-24 修改並驗證：run `20261124T065143797802Z`，總數與 generator CSV 一致）
  - 位置：[train_edl_water_kl.py:110-142](../scripts/train/train_edl_water_kl.py#L110-L142)
  - 問題：計畫第 39 行要求確認保留 patch 的 KL 像素數，但 manifest 只記 patch 數。`filter_windows_v2` 會丟掉 water GT invalid ≥ 80% 的 patch；KL 像素是 water invalid 的子集，所以雲最多、KL 候選最集中的 patch 最先被濾掉。
  - 做法：每個事件讀一次整張 KL mask，分別加總全部 window 與保留 window 的 KL 像素，寫入 manifest（逐事件與總計）。
  - 完成條件：manifest 有全量 KL 像素、保留 KL 像素與保留比例。
  - 處理結果：
    - 實際做法改為：總數取自 mask 的 `kl_pixels` tag，保留數只讀保留下來的 window（每個 window 對應一個 256×256 tile），不整張讀入記憶體。
    - 新增 `kl_pixel_statistics`，結果寫入 manifest 的 `kl_statistics`：`total`、`retained`、`retained_fraction`、`patches_with_kl`，以及逐事件的 `by_event`；無 AOI 事件記為 0。
    - 若保留數大於 tag 的總數會報錯。
    - 訓練開始前多印一行保留比例與含 KL 的 patch 數。
  - 驗證結果：
    - KL 像素總數 836,149,571，保留 386,423,589（46.2%）；65,582 個 patch 中 27,983 個含 KL。
    - 逐事件保留比例中位數 87.1%，但雲多的大事件損失最多（例如 EMSR279_05ZARAGOZA_DEL_MONIT01_v2 只保留 7.8%），所以整體只剩不到一半。
    - 468 個有 mask 的事件中，63 個 KL 像素為 0；1 個有 KL 但保留的 patch 都沒有 KL。

- [x] **3. 訓練前檢查 KL mask 的來源與規則版本**（2026-11-24 修改並驗證：run `20261124T065516477547Z`，468 個 mask 全部通過，結果與上一次 run 相同）
  - 位置：[train_edl_water_kl.py:94-97](../scripts/train/train_edl_water_kl.py#L94-L97)
  - 問題：目前只比對格式與網格。`--mask-root` 若指到舊規則或其他資料目錄產生的 mask，只要網格相同就會被接受。
  - 做法：比對 TIFF tags 的 `rule`、`mask_version` 與 `source_gt`（需等於目前 GT 的解析後路徑）。可沿用 [generate_water_only_kl_masks.py](../scripts/preprocess/generate_water_only_kl_masks.py) 中 `read_existing_mask` 的比對邏輯。只讀 header，成本與現有檢查相同。
  - 處理結果：訓練腳本對已有的 mask 直接呼叫 `read_existing_mask`，取代原本只比對格式與網格的檢查；與 generator 的 `--skip-existing` 用同一套標準。floodmap 改為兩種情況都必須存在。修改當下確認 train 的 S2、GT、floodmap 沒有比最早的 mask 更新的檔案。

## 值得注意（設計與效能）

- [x] **4. 放寬 window cache 的重用條件**（2026-11-24 決定不需修改）
  - 位置：[train_edl_water_kl.py:122](../scripts/train/train_edl_water_kl.py#L122)
  - 問題：只要有一個事件在 baseline index 裡完全沒有保留任何 window，`issubset` 就不成立，程式會對所有事件重算 `filter_windows_v2`。結果相同，但要重新讀取全部 train GT，在約 10 MB/s 的 NAS 上很慢。
  - 做法：SHA256 已固定 cache 內容，只需確認路徑字串一致，例如檢查 `images` 是否都在原本的 train split 清單中。
  - 決定不需修改的理由：第 1 項修改後，訓練一定使用完整的 train split（缺 mask 會報錯）；SHA256 固定的 cache 涵蓋全部 475 個 train 事件，沒有 0 個 window 的事件，所以條件永遠成立，三次驗證 run 也都直接使用 cache。換用其他 split 檔時 SHA256 不會相符，程式照常重算，屬於正確行為。

- [x] **5. 決定 invalid KL 是否需要獨立權重**（2026-11-24 與 Codex 討論後採用 C 並實作；smoke test 通過，見下方驗證結果）
  - 位置：[edl_water_kl.py:23-29](../flood_uncertainty/models/edl_water_kl.py#L23-L29)
  - 現況：invalid KL 與 EDL 內建的 KL 共用同一個 β(t)，符合計畫公式。
  - 影響：
    - config 為 `linear`、`annealing_step=10`，epoch 0 的 β=0，第一個 epoch 沒有 invalid KL。
    - 無法單獨調整 invalid KL 強度；之後若要做不同強度的 ablation，需要另一個係數。
  - 決定後記下：維持現狀，或新增獨立係數。
  - 決定：採用 C，invalid KL 用獨立固定係數 0.02，從 epoch 0 開始；內建 KL 維持 beta(t)。Codex（gpt-6.2-astra、xhigh、priority）建議相同，理由是 0.02 × 1.97 ≈ 0.039，約為 valid EDL 0.120 的三分之一，而 A 在 epoch 1 就達 0.197。
  - 實作：
    - 模型新增 `invalid_kl_weight`、`invalid_kl_anneal`（預設 1.0、true，等同舊行為 A），並驗證型別與範圍。
    - 每步記錄 `loss` 與 `train_kl_fraction`；每個 epoch 記錄 `train_kl_weight_epoch`（invalid KL 係數）與 `train_beta_epoch`。
    - 訓練 DataLoader 使用專用的 seed 42 generator，讓四組的 shuffle 與 worker seed 一致。
    - 四個 config 在 `configurations/edl_water_kl_ablation/`：`kl_off`（0、false）、`kl_shared_beta`（1.0、true）、`kl_002_beta`（0.02、true）、`kl_002_const`（0.02、false）；只有這兩個參數與 `experiment_name` 和 `edl_water.json` 不同。
    - `scripts/train/run_edl_water_kl_ablation.sh` 依序執行 `$ARMS`，輸出在 `$OUTPUT_ROOT/<arm>`；拒絕轉傳 `--config`、`--output-dir`。
  - Codex review 修正兩點：逐步指標原本只在每 100 個 batch 記錄，會被 Lightning 的 50 步寫入間隔記到錯的 step，改為每步記錄；腳本拒絕會覆蓋各組設定的參數。
  - 驗證：`kl_off`、A、B 在 epoch 0 的 invalid KL 權重都是 0，第一個 epoch 的指標應一致；C 應不同。
  - Smoke test 結果（`artifacts/models/edl_water_kl_smoke/`，60 batch、8 workers、`--skip-val`）：
    - 四組抽到相同的 batch：`train_kl_pixels_epoch` 都是 21,932,903，第 49 步的 `train_kl_fraction` 都是 0.0621。
    - 係數正確：`kl_off`、A、B 的 `train_kl_weight_epoch` 為 0，C 為 0.02；`train_beta_epoch` 都是 0。
    - `kl_off`、A、B 的指標不是逐位元相同，彼此差約 0.1–0.2%（valid EDL 0.14272–0.14299）。三組的計算完全相同（權重都是 0.0），差異來自 GPU 運算的非決定性在 60 步中累積；正式比較時，小於這個幅度的差異不具意義。
    - C 明顯不同：invalid KL 從約 2.20 降到 0.337，valid EDL 0.1493（約高 4.6%），water IoU 0.7243（其他三組 0.7247–0.7253）。只有 60 步，僅供確認設定生效。

## 小問題

- [x] **6. 1stpatch 檢查腳本改用含斜線的 prefix**（2026-11-24 與 Codex 討論後修改並驗證：兩支檢查腳本對 EMSR260 都是 `passed`，完整版保留 KL 29,379 與 DataLoader 一致，180° 旋轉下 GT／KL 不符像素皆為 0）
  - 位置：[check_water_only_kl_dataloader_1stpatch.py:197](../scripts/preprocess/check_water_only_kl_dataloader_1stpatch.py#L197)
  - 問題：使用裸字串 `"S2"`、`"gt"`。base `__getitem__` 的 `str.replace(..., 1)` 只換第一個出現處，路徑或事件名稱含 `gt`（例如 Washin**gt**on）就會換錯。
  - 做法：與訓練腳本一致，改用 `/S2/`、`/gt/`。
  - 現況確認：train／val／test 共 509 個事件中，路徑前段都不含 `gt` 或 `S2`，事件名稱都不含 `gt`；3 個名稱含 `S2`，但資料夾 `/S2/` 先出現，替換結果仍正確。換錯時 rasterio 會找不到檔案而報錯，不會悄悄讀錯資料，所以屬於潛在風險。
  - 處理結果：兩支檢查腳本建立 KL Dataset 時都改用 `/S2/`、`/gt/`（[1stpatch:198](../scripts/preprocess/check_water_only_kl_dataloader_1stpatch.py#L198)、[完整版:157](../scripts/preprocess/check_water_only_kl_dataloader.py#L157)）。當作 split JSON key 或 `Path` 組成的資料夾名稱維持不帶斜線。
  - Codex 補充：完整版會先呼叫 ml4floods 的 `get_dataset`，window cache 不存在時，baseline DataModule 篩選 window 仍用不帶斜線的 prefix。這是 ml4floods 的程式，且 cache 已存在、以 SHA256 固定，不會觸發，所以不改。

- [x] **7. 1stpatch 檢查腳本讀 window cache 前驗證 SHA256**（2026-11-24 與 Codex 討論後修改並驗證：兩支檢查腳本對 EMSR260 皆 `passed` 且結果與修改前相同；強制重算與 cache 選出的 window 一致，`14 14 True`）
  - 位置：[check_water_only_kl_dataloader_1stpatch.py:57-75](../scripts/preprocess/check_water_only_kl_dataloader_1stpatch.py#L57-L75)
  - 問題：cache 若被替換，檢查的 patch 不一定是實際訓練用的 patch。
  - 做法：沿用訓練腳本的 `BASELINE_INDEX_SHA256` 檢查。
  - 決定（與 Codex 一致）：兩支檢查腳本都照抄訓練腳本的 window 選擇。cache 只在設定、SHA256 與「涵蓋全部 train 影像」都成立時使用，並以完整路徑選列、維持 cache 順序；否則只為該事件重算 `filter_windows_v2`（每個 window 獨立判斷，所以結果等於訓練）。完整版不再呼叫 ml4floods `get_dataset`，避免在 cache 不存在時寫入共用資料目錄。
  - 處理結果：1stpatch 腳本新增 `train_pairs`、`training_pair`、`event_training_windows`、`training_dataset`，完整版改用這四個函式。訓練腳本不改，只被 import `BASELINE_INDEX_SHA256` 與 `split_pair`。
  - Codex review 後修正：直接讀本地 split JSON（不經會切換到 GCS 的 ml4floods 函式），並用訓練的 `split_pair` 檢查 S2／GT 路徑；`add_mndwi_input` 比照訓練，缺少視為 false、為 true 時報錯。import 常數會連帶載入訓練腳本的相依套件，無害，不改。

- [x] **8. 開啟驗證時也記錄 β**（2026-11-24 隨第 5 項完成：模型一律記錄 `train_kl_weight_epoch` 與 `train_beta_epoch`）
  - 位置：[train_edl_water_kl.py:56-59](../scripts/train/train_edl_water_kl.py#L56-L59)
  - 問題：`train_kl_weight_epoch` 只在 `--skip-val` 時記錄。

- [x] **9. 更新計畫文件中過時的敘述**（2026-11-24 完成）
  - 位置：[optical_water_only_invalid_KL_plan.md:45](optical_water_only_invalid_KL_plan.md#L45)（原第 43 行，前面補充內容後下移）
  - 問題：仍寫「Dataset／loss 尚待接入」。
  - 處理結果：改為 Dataset、loss、訓練腳本已接入並經 smoke test 驗證，連結到本清單；第 10、11 項完成後已更新為「產生器 25 個測試，以及 loss 與 Dataset 的 25 個測試全部通過」。

## 測試

- [x] **10. 執行 KL mask 產生器的測試**（2026-11-24 執行：收集 25 個、全部通過；依 pytest 快取判斷，`nodeids` 有 25 個測試且沒有 `lastfailed` 紀錄）
  - 文件 [generate_water_only_kl_masks.md:159](generate_water_only_kl_masks.md#L159) 註明尚未執行。
  - 指令：
    ```bash
    uv run python -m pytest -c pyproject.toml -q tests/test_generate_water_only_kl_masks.py
    ```
  - 審查時已手算前兩個測試的預期值（KL 18、候選 22、AOI 外 12，讀取 6/9 個 window 等），數字正確。

- [x] **11. 新增 loss 回歸測試**（2026-11-24 與 Codex 討論後建立 [tests/test_edl_water_kl.py](../tests/test_edl_water_kl.py)；執行結果：收集 25 個、全部通過，依 pytest 快取判斷，沒有 `lastfailed` 紀錄）
  - Loss：無 KL 時等於 baseline（epoch 0、5、12）；混合 batch（3 valid、2 KL、1 個 logits 為正的忽略像素）驗證兩組各自平均、忽略像素梯度恰為 0、valid 像素上的 KL 旗標無效；整批忽略時 loss 為 0 且可 backward；只有 KL 時為係數 × KL 平均（float64 獨立公式）；四組在 epoch 0／5 的兩個係數；缺參數時預設等同 A；不合法參數被拒絕。
  - Dataset（tmp_path 合成 GeoTIFF）：無 AOI 事件回傳全 0 且不需 mask 檔，不在清單又缺檔則 `RasterioIOError`；真實訓練轉換在 8 個 seed 下影像、GT、KL 方向一致；超出邊緣的 window 補 0。
  - 注意：config 的 `AttrDict` 以 `self.__dict__ = self` 實作，deepcopy 後屬性與項目會分離，所以測試每次重新載入 config，不做 deepcopy。
  - [ ] `kl_mask` 全為 0 時，`compute_loss` 等於 baseline 的 `calc_edl_loss_multioutput_logistic_mask_invalid(logits, target)`。兩者都是 valid 像素平均，數學上相同。
  - [ ] water=0 且 KL=0 的像素，梯度為 0。
  - [ ] 整批像素都被忽略時，loss=0 且仍可 backward。

## 後續新增

- [ ] **12. 檢查腳本支援無 AOI 的事件**（處理第 6 項時發現，與 Codex 討論後延後）
  - 位置：[check_water_only_kl_dataloader_1stpatch.py:201](../scripts/preprocess/check_water_only_kl_dataloader_1stpatch.py#L201)、[check_water_only_kl_dataloader.py:159](../scripts/preprocess/check_water_only_kl_dataloader.py#L159)
  - 問題：兩支檢查腳本沒有傳入 `events_without_kl`，而且會直接開啟 KL mask 檔；拿 7 個無 AOI 的事件來檢查會因為找不到檔案而失敗。只傳 `events_without_kl` 不夠。
  - Codex 建議的做法：先確認事件確實沒有 AOI，再把事件名稱傳給 Dataset，並檢查回傳的 KL mask 全為 0；有 AOI 卻沒有 mask 的事件仍要報錯。

- [ ] **13. 決定驗證結果用的圖表與 W&B／本地輸出**（2026-11-24 新增；A 部分須在正式 ablation 前決定）
  - 現況：
    - 開啟驗證的訓練預設 `WANDB_MODE=offline`，資料只存在 run 目錄的 `wandb/`；要上傳須加 `--wandb-mode online` 或事後 `wandb sync`（帳號已登入）。
    - W&B 目前只收到數值指標與固定 patch 監看圖；模型的 batch 影像 log 要求第一個 logger 是 W&B，但第一個是 CSV，所以從未觸發。
    - 本地一定會存：`metrics.csv`、每個 epoch 的 `train_probe/*.png`、manifest、window 清單；checkpoint 只留最新與最好各一個；`train_summary.png` 只在 `--skip-val` 產生。
    - Claude 只能檢查本地檔案（W&B 離線檔是二進位，網頁儀表板也連不上）。原則：上 W&B 的圖表一律同時存本地 PNG／CSV，以本地為準。
  - A. 開始 ablation 前決定（事後補不回來）：
    - 是否每個 epoch 都保留 checkpoint：完整 checkpoint 93 MB，4 組 × 30 epoch 約 11 GB；只存權重約 31 MB，總計約 3.7 GB；NAS 尚有 52 TB。保留後，任何逐 epoch 指標都能事後補算。
    - W&B 直接 online 上傳，或維持離線、事後 sync。
  - A 部分決定與實作（2026-11-24 與 Codex 討論；smoke test `edl_water_kl_smoke/kl_002_const/20261124T080600038973Z` 通過：每 epoch 權重 checkpoint 31 MB × 2、`best`／`last` 正確，5 個程式副本與 SHA256 一致，`lr-Adam` 有記錄，W&B online 無錯誤且本地有副本；每個 epoch 的完整驗證約 1 分鐘以內）：
    - checkpoint：新增每個 epoch 只存權重的 checkpoint（`checkpoint/epochs/`，`monitor="epoch"`、`mode="max"`，以免和 `latest` 的 Lightning state key 衝突），保留完整的 `last.ckpt` 與 `best.ckpt`；4 組合計約 4.5 GB。
    - 新增 `LearningRateMonitor(logging_interval="epoch")`。
    - 出處：每個 run 把 5 個實驗程式檔複製到 `code/`，manifest 記錄它們、預訓練 checkpoint 與 split JSON 的 SHA256（split 只讀一次，解析與雜湊用同一份 bytes；checkpoint 在載入前一刻雜湊）。
    - W&B：使用者選 online。bash 腳本預設 `WANDB_MODE=online`（可設 `offline` 改回），run 名稱改用各組的 `experiment_name`；online 模式下每個 run 仍保留本地 `wandb/` 副本，`metrics.csv` 與 PNG 照常產生。
    - 四個 config 加入 `"num_workers_val": 8`（驗證轉換無隨機成分，不影響結果）。
    - Codex review 後修正：訓練腳本的參數解析關閉縮寫（`allow_abbrev=False`），避免 `--conf` 繞過 bash 腳本對 `--config` 的阻擋。
    - 第一次 online smoke test 失敗：W&B 回報 `entity flood_uncertainty not found`（HTTP 404），訓練在開始前停止。11/23 的 online 嘗試也是同樣錯誤，之前的 run 能完成是因為都在離線模式下跑。四個 ablation config 的 `wandb_entity` 改為 `null`，使用登入帳號的預設 entity；其他 config 不動。
  - B. 評估規格（2026-11-25 與 Codex 討論後固定；**在檢視任何 test 結果之前寫定，之後不再更動**）：
    - checkpoint：各組 `val_bce_land_water` 最佳的 epoch（kl_off 3；A／B／C 4，C 以訓練完成時為準）。`last` 只做事先講定的敏感度分析，且只重做主要指標表。
    - 比較：主要為 C 對 kl_off；A、B 為次要。
    - 資料：test 18 個事件（14 個有 KL mask，4 個無 AOI）。val 只用來選門檻與事前檢查。整體指標用全部 18 個事件；雲旁／遠離雲的比較只用 14 個有 AOI 的事件。雲旁 = 有效像素中，與 KL 像素的 Chebyshev 距離 ≤ 3 px 者（在整個事件上計算）。
    - 主要指標：water IoU（有效像素，P(water) > 0.5）、雲旁 water recall、Brier（P(water)）、AURC（以 u 排序錯誤）。
    - 次要指標：
      - 錯誤偵測 AUROC 與 AP：分全部／雲旁／遠離雲，附錯誤數、有效像素數、錯誤率；只在同一區域內跨組比較；只有單一類別的區域記為無法定義。
      - precision、固定分箱 ECE。
      - PAvPU 三個版本：val 固定 Q90（curve）、per-image 預測有效 Q90（實際使用版）、per-image GT 有效 Q90（診斷版）。patch IoU 門檻 0.9、patch size 3、retention 0.9，與既有結果相同。
    - KL 區：像素數、中位數 u、兩個 evidence 皆為 0（u = 1）的比例；依雲 GT 分成雲／晴空／invalid 三組。這些只描述 evidence 被壓低的程度，效果好壞以有效像素上的指標判斷。
    - 計算：u 以 10,000 個均勻分箱累計，u = 1 獨立一格。同分時 AUROC 給一半分數、AP 以同分組計算、AURC 以同分組為單位推進覆蓋率。Brier 用誤差平方和，ECE 用分箱內的實際機率總和。正式使用前，先在 val 上比對分箱與精確排序的結果。
    - 統計：全部合併的數值、對 kl_off 的逐事件配對差、event bootstrap 95% 區間（各組使用同一批抽樣事件）。區間含 0 時報「無法判定」。每組只有 1 個 seed，訓練的隨機性沒有量測。
    - case study：
      - 依資料挑選：每個 test 事件、每條規則挑 1 個 256×256 window（符合條件的像素數必須 > 0）。四條規則：KL 最多、雲旁 water 最多、晴空對照（雲 GT 為晴空、有 water、無 KL）、晴空 KL（KL 且雲 GT 為晴空）。全部放進總覽圖與 CSV；每類依數量由大到小排序（同數量時依事件、row、col）精選 2 個不同事件畫大圖；重複的 window 只畫一次，保留所有規則標籤。
      - 依結果挑選（另外標明）：以雲旁 water recall 的差值（C − kl_off）為指標，雲旁 water 像素至少 500 個，挑改善最多、退步最多各 1 個。
      - 版面：4 列 × 4 欄（原始資料｜u｜P(water)｜錯誤圖），共用 0–1 色階。錯誤圖的 FP 用藍色、FN 用橘色，判對的像素用淡色，忽略區用另一種底色；下方寫出雲旁 water recall 和分母。
    - 刪減：水體 AUPRC、全景概覽圖。
  - B 部分進度：
    - [x] 驗證集平面圖：[plot_val_kl_maps.py](../scripts/eval/plot_val_kl_maps.py)，用各組最佳的 epoch，輸出在 `artifacts/eval/edl_water_kl_ablation/val_maps/best_20261125T025306Z/`。觀察（6 張刻意挑選的 patch，只能描述現象）：
      - KL 三組在雲下的 u 接近 1、P(water) = 0.5，kl_off 在雲下很有把握；遠離雲的有效像素幾乎不受影響。
      - 雲旁有效像素的 u 會被拉高：A 最嚴重（例如 0.07 → 0.45），B、C 增加 0.02–0.06。
      - water IoU 有升有降；FARKADONA 下降約 0.09，推測是湖岸靠近雲的水被拉到 0.5 而判成 land。
      - EMSR279 ZARAGOZA 的 KL 區是明亮混濁的洪水、不是雲，四組的 u 都很低。推測是雲偵測誤判，尚未驗證。
    - [x] PAvPU：[compute_pavpu.py](../scripts/analysis/compute_pavpu.py) 新增預設關閉的 `--threshold-pixels gt-valid`；CSV 新增 `threshold_tie_fraction`、`threshold_flagged_pixel_fraction`、`threshold_pixels`、`diagnostic_pixels`。Codex review 後修正：同分比例改在 float64 下比較，curve 模式拒絕非有限門檻。`tests/test_pavpu_adaptive.py` 10 passed。
    - 範圍精簡（使用者決定）：不寫報表產生器，英文 Markdown 報表由 Claude 從 CSV 與 PNG 手動組成，放在 `docs/edl_water_kl_ablation_report/`（`report.md` 與 `figures/`）。正文：Summary、Setup、主要指標表、PAvPU 小表、risk–coverage＋reliability 圖、2 個精選 case；其餘移到附錄。
    - [x] 步驟 1–2 腳本：[run_kl_ablation_eval.sh](../scripts/analysis/run_kl_ablation_eval.sh)。只接受完整跑完的正式 run（30 epoch、開啟驗證、未限制 batch、split SHA256 相符），推論、`eval_metrics.py`、`analysis_S2.py`（出現 `Error processing` 即停止）、`compute_pavpu.py` 三個版本；核對預測與混淆圖都和 split 一致。（2026-11-25 已執行：權重 kl_off epoch 3、A／B／C epoch 4；val 16、test 18 張與 split 一致；`analysis_S2` 無錯誤；四組評分像素總數相同。輸出 `artifacts/results/edl_water_kl_ablation/best/`）
    - [x] 步驟 3 腳本：[evaluate_kl_ablation.py](../scripts/analysis/evaluate_kl_ablation.py)。輸出 `events.csv`、`event_metrics.csv`（逐事件與配對差）、`pooled_metrics.csv`（含 bootstrap 區間）、`kl_region.csv`、`pavpu_summary.csv`、`risk_coverage_reliability.png`；`--check-exact` 在 val 上比對，超過 1e-3 即以錯誤結束。（2026-11-25 已執行：val 精確比對最大差 5.3e-4，PASS；左圖標題改為依 `--split` 顯示。test 評分像素 167,122,003，和 `eval_metrics` 一致。輸出 `best/evaluation/{val,test}/`）
    - [x] 步驟 3b：擴充 [plot_val_kl_maps.py](../scripts/eval/plot_val_kl_maps.py)，不另寫腳本。`--split test --pred-root` 讀已存預測；4×4 精選圖（含錯誤圖與雲旁 recall）、分頁總覽圖、依結果挑選、`patch_stats.csv`。（2026-11-25 已執行：8 張精選圖、35 個 window 的 6 頁總覽；test 沒有晴空下的 KL 像素，所以 `clear_kl` 沒有選到 window。輸出 `artifacts/eval/edl_water_kl_ablation/case_studies/test_best_saved_20261125T082357Z/`）
    - [x] 整張事件的平面圖（使用者要求）：新增 [plot_kl_event_maps.py](../scripts/eval/plot_kl_event_maps.py)，每個事件一張 4×4 圖，版面和 case study 相同，讀已存預測。標題和 `event_stats.csv` 的數字用全解析度計算，只有顯示時每 s 個像素取一個。`plot_featured` 改為由呼叫者傳入第一行標題，並可指定 `figsize`、`dpi`；預設值不變，所以現有 case study 的輸出不變。（尚待執行）
    - test 主要結果（C − kl_off，95% CI）：IoU −0.008 [−0.016, +0.004] 看不出差異；recall −0.006 [−0.017, −0.001] 顯著下降（A −0.022、B −0.028）；整體錯誤偵測 AP −0.022 [−0.067, −0.007]；雲旁 AP 三組都顯著下降；雲旁水的 recall 沒有改善；雲下 u 中位數 kl_off 0.075、C 0.95。
    - Case study 觀察：VATOMANDRY 是預期效果（雲下 u≈1、IoU 不變）；NORTHNORMANTON 的水 GT 位於薄雲下，KL 組在這裡把證據壓到 P≈0.5，又因判定規則是 P > 0.5，所以被判成 land（A 的 recall 0.18、C 0.84）；SOUTHNORMANTON 是沒有 KL 的晴空混濁水，KL 組的 FN 也增加，而且 u 很低。
    - Codex review 後修正：依結果挑選改用四組共同有效像素；新增逐事件指標；單一類別時 AP 為 NA；KL 區只用四組皆有效的像素、中位數取中間兩筆平均；精確比對失敗即結束；晴空對照要求範圍內全為晴空；無 AOI 的雲旁／遠離雲記為 NA；重複 window 合併；讀已存預測時檢查網格。
    - [x] 步驟 4：`last` 的主要指標表（`CHECKPOINT=last` 重跑步驟 1–3）。（2026-11-25 已執行：四組都用 epoch 29。整體 IoU 的差異消失：C − kl_off +0.001 [−0.001, +0.003]，`best` 時 recall 的下降也不再出現；雲旁錯誤偵測 AP 仍然顯著下降，C −0.141。所以分割上的差異取決於權重的選擇，雲旁 u 變差在兩種權重下都成立。）
    - [x] 英文報告：[report.md](edl_water_kl_ablation_report/report.md)，圖在 `figures/`（4.8 MB）。依 Codex 審稿修改（gpt-6.2-astra、xhigh、priority）：
      - 摘要改為先講主要結果（`best` 時 recall 下降），再講 `last` 的敏感度分析。
      - 雲旁 AP 補充「6 個區間都不含 0」以及錯誤率；A 改寫成「觀察到的下降最大」。
      - 機制改用保守的說法。
      - PAvPU 的比例改為逐張平均的像素比例。
      - 依原始 CSV 修正 9 個四捨五入的數字。
      - 依使用者要求精簡：每張圖表只寫「顯示什麼」與「怎麼解讀」。
    - [x] 中文版：[report_zh.md](edl_water_kl_ablation_report/report_zh.md)，專有名詞保留英文。
      - Codex 審查翻譯（gpt-6.2-astra、xhigh、priority）：數值、正負號、†、粗體都和英文版一致。
      - 措辭改為：「95% CI 包含 0，尚無法判定」；「KL 鄰近像素（near-KL）」；「各影像的 Q90」；機制改用「與……的解釋相符」。
      - 同時修正英文版：表 2 的「仍然較低」只對 AP 成立；摘要改為 invalid-pixel KL，以和 EDL 本身的 KL 區分。

## 已確認、不需處理

- 選取規則與 AOI（`w_class == "area_of_interest"`、`all_touched=True`）和 ml4floods `create_gt.py` 一致。
- 同步增強：rot90、flip 對 3 通道 mask 使用同一組參數；`Normalize` 只作用在影像；padding 填 0。
- `seed_worker_transform` 是必要的：albumentations 2.0.8 的 `Compose` 有自己的隨機數產生器，少了它每個 worker 會重複同一串增強。
- Loss 兩組分開平均、空集合為 0 且保留梯度、KL 使用完整 alpha、其餘 invalid 像素被忽略。
- LR scheduler 的 `frequency` 與驗證頻率對齊；固定 patch 監看圖能取得同一 epoch 的數值。
- Generator 只在完整寫完後發布 mask，失敗不留部分檔案，續跑時檢查既有檔案。
