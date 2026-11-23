# flood-uncertainty

Flood segmentation uncertainty 專案（`v2` / `EDL` / `ensemble` / `MC dropout` / `EDL-TERRAMIND`）。

## 1. 專案結構

- `flood_uncertainty/`: 核心套件（models/losses/metrics/inference/utils）
- `scripts/`: train / infer / eval 入口
- `configurations/`: 每個方法一份 config（`shared/train/infer/validate_only`）
- `artifacts/`: 權重與輸出（已在 `.gitignore`）
- `ml4floods/`, `mlguess/`: 本地依賴

## 2. 環境

目前流程以 `uv` 環境執行：

```bash
uv sync --python 3.10
uv run python -V
```

## 3. Config mode

每份 `configurations/*.json` 都使用相同 mode 架構：

- `train`: 訓練
- `infer`: 推論
- `validate_only`: 只跑 validation（目前主要用於 EDL）

## 4. 常用指令

### 4.1 訓練

```bash
uv run python scripts/train/train_v2.py --config configurations/v2.json --mode train
uv run python scripts/train/train_edl.py --config configurations/edl.json --mode train
uv run python scripts/train/train_dropout.py --config configurations/dropout.json --mode train
uv run python scripts/train/train_ensemble.py --config configurations/ensemble.json --mode train
uv run python scripts/train/train_edl_sar.py --config configurations/edl_sar.json --mode train
uv run python scripts/train/train_edl_terramind.py --config configurations/edl_terramind.json --mode train   # EDL + TerraMind v1 encoder
```

EDL-TERRAMIND 訓練 log 預設走 **W&B offline**（`wandb_mode` in config，或 `WANDB_MODE=online|offline|disabled` 覆寫），
run 與 `metrics.csv` 都在 `artifacts/models/<experiment_name>/`，事後上傳：`wandb sync artifacts/models/<experiment_name>/wandb/offline-run-*`。

可選覆寫資料根目錄：

```bash
uv run python scripts/train/train_edl.py --config configurations/edl.json --mode train --data_root /path/to/worldfloods_v2/data
```

#### 使用 10 組 ensemble 權重初始化 EDL

`configurations/edl_ensemble.json` 指向已確認存在的
`/home/NAS/house/chlunchen-10030/datasets/cj/cj/ML4FloodsUncertainty/models/ensemble/`
底下第 0～9 組 `WF2_unetv2_bgriswirs_test_*`。資料根目錄使用同一個 NAS 下的
`datasets/cj/cj/data/worldfloods_v2/data`，可用 `--data_root` 覆寫。

```bash
# 只檢查來源權重、輸入通道與參數尺寸，列出選中的 checkpoint；不寫檔或訓練
uv run python scripts/train/train_edl_ensemble.py --dry_run

# 一次啟動，依序訓練 10 組（使用 configurations/edl_ensemble.json）
uv run python scripts/train/train_edl_ensemble.py

# 只訓練指定的來源成員（目前 offset=10，以下輸出編號為 15～19）
uv run python scripts/train/train_edl_ensemble.py --members 5 6 7 8 9
```

每組從仍存在的 checkpoint 中選擇 `val_bce_land_water` 最低的權重；讀取各檔案的
`best_k_models`，並以檔名重定位舊 NAS 路徑。若完全沒有可用分數，會明確提示並改選最大 step。
沿用 ensemble 的 `seed + i`，目前基準 seed 為 7，因此各組使用 7～16。
`shared.member_index_offset` 目前為 10，來源 `i=0～9` 對應輸出編號 `10～19`；
此偏移量只改輸出編號，不改來源權重或 seed 的遞增規則。未設定偏移量時從 0 開始。
以現有 `train_edl.py` 分別建立新的 EDL 訓練。
UNet 主體載入來源權重；2 通道分類 head 改成隨機初始化的 4 通道 EDL head。
會先檢查全部指定成員的權重相容性，再啟動第一組訓練。

輸出為 `artifacts/models/edl_ensemble/edl_from_ensemble_{10..19}/`，保留上一輪的 0～9 組。
各自包含 `config.json`、`pretraining.json`（輸出及來源編號、來源權重、分數、seed、
重新初始化的參數）、`batch_plot.png` 與 `checkpoint/`。
目前沿用 EDL 的 batch size 64、最多 30 epochs。EDL 的本機 CSV logger 一律啟用，
每次訓練的原始指標寫到成員目錄的 `logs/version_N/metrics.csv`。
每組訓練結束後會合併續訓紀錄，輸出 `loss_history.csv` 與 `loss_curve.png`；
從較早 checkpoint 續訓時，會以新紀錄取代該 epoch 起的舊曲線。
`loss_history.csv` 的 epoch 從 0 開始，圖上從 1 開始。
圖中分開顯示 training loss 與 validation loss，因為現有兩者的 class/task weights 與 KL 設定不同。
`val_bce_*` 欄位實際是 EDL loss，圖上也會以 EDL loss 標示。
批次訓練完成後，會將本輪已有紀錄的成員畫成
`artifacts/models/edl_ensemble/edl_from_ensemble_10_19_loss_curves.png`，保留上一輪總覽圖，
比較各組 training loss 與
checkpoint monitor `val_bce_land_water`。圖直接存成 PNG，適合沒有桌面的 NAS 環境。

W&B 的離線功能仍保留：`shared.wandb_mode` 支援 `disabled`（預設）與 `offline`，
也可用環境變數覆寫。兩種模式都會保留 CSV 與 loss 圖：

```bash
# 只記錄本機 CSV 與 loss 圖
WANDB_MODE=disabled uv run python scripts/train/train_edl_ensemble.py

# 額外保存離線 W&B 紀錄及原有影像記錄，無須登入
WANDB_MODE=offline uv run python scripts/train/train_edl_ensemble.py
```

離線 W&B 紀錄位於各成員目錄的 `wandb/`；W&B 不另存 model checkpoint artifact。
其他訓練入口（例如 EDL-TERRAMIND）原有的 W&B 模式維持原設定。
`--dry_run` 只檢查 CPU 上的權重與設定，不驗證完整資料集或 GPU 訓練。
Checkpoint 選擇、權重尺寸、各組設定及 loss 紀錄合併的測試可執行：

```bash
uv run python -m pytest -c pyproject.toml -q tests/test_edl_ensemble.py tests/test_loss_plotting.py
```

批次入口會拒絕寫入非空的成員輸出目錄；已開始的成員可用保存的設定單獨續訓：

```bash
uv run python scripts/train/train_edl.py \
  --config artifacts/models/edl_ensemble/edl_from_ensemble_0/config.json \
  --mode train \
  --resume_ckpt artifacts/models/edl_ensemble/edl_from_ensemble_0/checkpoint/last.ckpt
```

### 4.2 只驗證（EDL）

```bash
uv run python scripts/train/train_edl.py --config configurations/edl.json --mode validate_only
```

### 4.3 推論

以下指令從專案根目錄執行。EDL 使用 EpochKL epoch 8（測試指定的 trusted checkpoint）；權重不隨 Git 提供，請先自行放到 `--checkpoint` 指定的路徑。

```bash
uv run python scripts/inference/run_inference.py --model_type v2 --config_v2 configurations/v2.json
uv run python scripts/inference/run_inference.py \
  --model_type EDL \
  --config_edl configurations/edl.json \
  --checkpoint artifacts/models/edl_EpochKL_20260819/checkpoint/epoch=8-step=9225.ckpt
uv run python scripts/inference/run_inference.py --model_type EDL-TERRAMIND --config_edl_terramind configurations/edl_terramind.json --checkpoint /path/to/last.ckpt
uv run python scripts/inference/run_inference_ensemble.py --mode ensemble --config configurations/ensemble.json
uv run python scripts/inference/run_inference_ensemble.py --mode mcdropout --config configurations/dropout.json
uv run python scripts/inference/run_inference_sar.py input.tif output.tif --config configurations/edl_sar.json --weights /path/to/sar.ckpt
```
如果需要實際權重路徑，請洽jerrychlun

### 4.4 評估

```bash
uv run python scripts/eval/eval_metrics.py --model_type v2 --subset val
uv run python scripts/eval/eval_metrics.py --model_type EDL --subset val
uv run python scripts/eval/eval_metrics.py --model_type EDL-TERRAMIND --subset val
```

## 5. Smoke Scripts

```bash
bash scripts/smoke/smoke_infer_eval.sh
bash scripts/smoke/smoke_train.sh
bash scripts/smoke/smoke_analysis.sh
bash scripts/smoke/smoke_all.sh
```

常用覆寫參數（不帶就用預設）：

```bash
SMOKE_INFER_MODEL_TYPE=EDL SMOKE_INFER_SUBSET=val SMOKE_INFER_MAX_FILES=1 bash scripts/smoke/smoke_infer_eval.sh
SMOKE_TRAIN_MODEL=v2 bash scripts/smoke/smoke_train.sh
SMOKE_TRAIN_MODEL=EDL_TERRAMIND bash scripts/smoke/smoke_train.sh          # 或 bash scripts/smoke/smoke_train_edl_terramind.sh
SMOKE_INFER_MODEL_TYPE=EDL-TERRAMIND SMOKE_INFER_CHECKPOINT=/path/to/last.ckpt bash scripts/smoke/smoke_infer_eval.sh
SMOKE_ANALYSIS_MODEL_TYPE=EDL SMOKE_ANALYSIS_PREPARE_INFER=1 SMOKE_ANALYSIS_MAX_FILES=1 bash scripts/smoke/smoke_analysis.sh
SMOKE_ALL_RUN_TRAIN=0 bash scripts/smoke/smoke_all.sh
```

## 6. Tests

### 6.1 執行測試

執行全部專案測試：

```bash
uv run python -m pytest -c pyproject.toml -q tests
```

只執行 optical EDL numerical regression tests：

```bash
uv run python -m pytest -c pyproject.toml -q tests/test_optical_regression.py
```

Optical regression tests 使用以下 trusted checkpoint：

```text
artifacts/models/edl_EpochKL_20260819/checkpoint/epoch=8-step=9225.ckpt
```

測試包含：

- Fixed-checkpoint inference regression：使用固定 synthetic optical tensor 執行正式 inference pipeline，比較 classification mask、probability、evidence、DST uncertainty、aleatoric uncertainty 與 epistemic uncertainty。
- Fixed-batch training-step regression：還原固定 model checkpoint 與 Adam optimizer state，在固定 synthetic optical batch 上執行一次 forward、EDL loss、backward 與 optimizer update，再比較 loss、gradient 與更新後 logits。

若 trusted checkpoint 不存在，optical regression tests 會顯示為 skipped。Checkpoint 不會複製到 Git repository。

### 6.2 更新 optical regression baseline

目前保存的 numerical baseline 位於：

```text
tests/fixtures/optical_edl_epoch8_regression.pt
```

只有在確認模型的新計算行為正確、並且決定接受該變更後，才可重新產生 baseline：

```bash
uv run python scripts/tests/generate_optical_regression_baseline.py
uv run python -m pytest -c pyproject.toml -q tests/test_optical_regression.py
```

不要在一般 pytest 執行過程中自動更新 baseline，否則錯誤結果可能被接受為新的標準。重新產生後應檢查程式變更原因、baseline diff，並重新執行 regression tests。

## 7. Analysis 指令

### 7.1 單支執行

執行 PAvPU 前，先將 `PAVPU_ACC_THRESHOLD` 設為預先決定的 patch Water IoU 門檻（0～1），
所有模型使用相同數值。以下指令透過必填的 `--acc-threshold` 傳入該門檻。

```bash
uv run python scripts/analysis/analysis_S2.py --model-type EDL --subset val --output-dir artifacts/results/analysis_S2/val --max-files 1
uv run python scripts/analysis/compute_pavpu.py --model-type EDL --subset val --acc-threshold "$PAVPU_ACC_THRESHOLD" --max-files 1
uv run python scripts/analysis/plot_retention_curve_compare.py --group all
uv run python scripts/analysis/plot_epistemic_fp_fn_compare.py --subset val --max-files 1
```

`compute_pavpu.py` 預設使用 **val curve 設定 uncertainty 門檻、評估 test 影像**（`--threshold-mode curve`）。
準確度門檻獨立由 `--acc-threshold` 指定，不再採用 curve CSV 的 IoU。
有效區域由預測 TIFF 的 Classification（排除 0）、來源 NoData 與有限值建立，不使用 GT；
計分時再與混淆圖的有效條件取交集，每個區塊（預設 3×3）內任一像素無效就排除。
`--threshold-mode per-image` 會對每個模型、事件及 uncertainty 類型，
使用整張影像所有有效像素（包含邊界像素）的 uncertainty 計算一個 Q90。
此模式不讀 retention CSV，也不使用 GT 選擇門檻；`--retention-target` 可調整分位數，預設 0.9。
PAvPU 仍使用區塊平均 uncertainty 與此門檻比較，判定規則為 `mean_unc >= threshold`。
像素與區塊的單位不同，加上相同值及評分範圍篩選，不保證最後恰有 10% 區塊判為不確定。

```bash
# 每個事件的有效像素 Q90 → test
uv run python scripts/analysis/compute_pavpu.py \
  --model-type EDL \
  --subset test \
  --threshold-mode per-image \
  --retention-target 0.9 \
  --acc-threshold "$PAVPU_ACC_THRESHOLD"
```

自適應結果另存至 `artifacts/results/analysis_S2/<subset>/<model>/pavpu_per_image_q90/pavpu_*.csv`；
資料夾尾碼會隨分位數變更。每個事件記錄自己的 `unc_threshold`，`OVERALL`／`AVERAGE` 的門檻留空，
並保留共同的 `acc_threshold` 與 `threshold_mode=per-image`；`curve_subset`／`curve_csv` 留空。

兩種模式的 CSV 都記錄 `n_valid_patches`（`ac + au + ic + iu`，有效評分區塊數）與
`uncertain_fraction`（`(au + iu) / n_valid_patches`，0～1）。這是套用評分範圍後的區塊統計，
不是用來計算 Q90 的像素數。終端會逐事件顯示檔名、Q90／uncertainty 門檻、準確度門檻、
有效評分區塊數、不確定比例（百分比）及 PAvPU。
`OVERALL` 使用總計數計算比例；`AVERAGE` 記錄各事件區塊數及比例的算術平均。
沒有有效評分區塊的事件，其不確定比例及 PAvPU 在 CSV 留空、終端顯示 `N/A`，
且不納入這兩個比例欄位的事件平均。

在 `curve` 模式中，
`--curve-subset` 選擇 uncertainty 門檻來源，`--subset` 選擇待評估影像；兩者可獨立指定：

```bash
# val curve → test（預設）
uv run python scripts/analysis/compute_pavpu.py --model-type EDL --subset test --curve-subset val --acc-threshold "$PAVPU_ACC_THRESHOLD"

# test curve → test
uv run python scripts/analysis/compute_pavpu.py --model-type EDL --subset test --curve-subset test --acc-threshold "$PAVPU_ACC_THRESHOLD"
```

`curve` 模式從 `artifacts/results/analysis_S2/<curve-subset>/<model>/retention_*.csv` 讀取門檻，
結果寫入 `artifacts/results/analysis_S2/<subset>/<model>/pavpu_from_<curve-subset>/pavpu_*.csv`。
CSV 會記錄 threshold mode、curve 來源、門檻值、retention target 和 patch size。
預設取最接近 90% retention 那列的 `threshold` 作為不確定性門檻；patch IoU 門檻使用指定的固定值。

如果 curve 位於其他目錄，用 `--curve-root` 指定**包含模型子目錄**的資料夾，
並以 `--curve-subset` 正確標記該批 curve 的來源。`--output-dir` 可獨立指定輸出根目錄，
程式會依模式附加 `<model>/pavpu_from_<curve-subset>/` 或 `<model>/pavpu_per_image_q<百分位>/`。
舊的 `--analysis-root` 保留為 `--output-dir` 的別名，**不再用來選擇門檻 CSV**。
批次與 smoke 腳本會明確指定使用該次執行產生的 curve，並要求環境變數 `PAVPU_ACC_THRESHOLD`。
批次腳本也可從 `scripts/analysis/analysis.env` 讀取此設定；未設定時會在分析開始前停止。

完成 EDL、ensemble、mcdropout 的自適應 PAvPU 後，可匯出 Table 4（整體）與 Table 5（指定事件）：

```bash
uv run python CJ_scripts/export_pavpu_tables.py \
  --subset test \
  --threshold-mode per-image \
  --retention-target 0.9 \
  --event EMSR466_AOI01_DEL_PRODUCT
```

程式會讀取三個模型的 `pavpu_per_image_q90/pavpu_*.csv`，
確認評估子集、模式、分位數、區塊大小及固定準確度門檻一致。
每個事件的 `unc_threshold` 可不同；自適應結果不需要 `curve_subset`／`curve_csv`。
輸出預設位於 `artifacts/results/figures/pavpu_tables/test_per_image_q90/`，包含
`pavpu_tables.png`、`table4_pavpu_overall.csv` 與 `table5_pavpu_event.csv`。
圖說會標示每個事件的有效像素分位數、區塊大小及固定準確度門檻。
表格程式預設仍為 `--threshold-mode curve --curve-subset val`，可讀取既有曲線門檻結果；
此舊模式不強制不同模型共用準確度門檻，以保留先前結果的讀取方式。

自適應 PAvPU 的小型驗證使用合成 TIFF 與手算結果，涵蓋整張影像 Q90、有效範圍、
四格計數、GT 變動不影響門檻，以及 CSV 的整體／逐事件平均：

```bash
uv run python -m pytest -c pyproject.toml -q tests/test_pavpu_adaptive.py
```

測試中的 `acc_threshold=0.5` 僅為合成案例設定，不是正式實驗的建議值。
測試通過後，可使用預先選定的 `PAVPU_ACC_THRESHOLD`，先評估 val 按檔名排序的第一張影像：

```bash
uv run python scripts/analysis/compute_pavpu.py \
  --model-type EDL \
  --subset val \
  --threshold-mode per-image \
  --retention-target 0.9 \
  --acc-threshold "$PAVPU_ACC_THRESHOLD" \
  --max-files 1 \
  --output-dir artifacts/results/analysis_S2_validation/val
```

單張驗證結果位於 `artifacts/results/analysis_S2_validation/val/EDL/pavpu_per_image_q90/`。
檢查四種 uncertainty 的事件門檻、有效評分區塊數、不確定比例及 PAvPU，再進行完整評估。

### 7.2 一鍵執行全部 analysis（含時間與步驟提示）

先編輯：

`scripts/analysis/analysis.env`

再執行：

```bash
bash scripts/analysis/run_all_analysis.sh
```

可選：指定其他 env 檔

```bash
ANALYSIS_ENV_FILE=/path/to/analysis.env bash scripts/analysis/run_all_analysis.sh
```

此腳本執行順序為：

`analysis_S2 (all models) -> compute_pavpu (all models) -> plot_retention_curve_compare -> plot_epistemic_fp_fn_compare`

執行 log 會寫到：

`artifacts/results/analysis_S2/logs/`

## 8. TerraMind encoder（EDL-TERRAMIND）

`EDL_TerraMind_ML4FloodsModel` 用 TerraMind v1 base（ViT-B/16）取代 EDL 模型的 UNet encoder，decoder / EDL head / loss 不變。
encoder 原始碼 vendor 自 terratorch 1.2.11（`flood_uncertainty/models/backbones/terramind_vendored/`，Apache-2.0），
權重由 `huggingface_hub` 取得（`ibm-esa-geospatial/TerraMind-1.0-base`）。config 欄位 `backbone_*` 說明與設計理由見
`docs/terramind_edl_integration_log.md`；整合計畫見 `docs/foundation_model_integration.md`。

- 預設 config `configurations/edl_terramind.json` = `tm_l1c13`（S2L1C 13 band、encoder 不凍結）。
- 比較實驗的 config 在 `configurations/terramind/`（`tm_l2a6`、`tm_l1c13`、`tm_l1c13_tim`、`tm_l1c13_inskip`、`tm_frozen_l2a6`），
  結果見 `docs/edl_vs_terramind_comparison.md`。
- 推論 tile：`run_inference.py --max_tile_size N`；未指定時 `EDL-TERRAMIND` 用 256（訓練 window），其他 model_type 用 1024。
- 推論只從 checkpoint 載入權重，不會下載 HF 權重（可在離線節點執行）。

## 9. 路徑說明

- 資料根目錄：預設由 config 的 `data_params.path_to_splits` 決定（可用 `--data_root` 覆寫）
- 推論與評估輸出：預設寫到 `artifacts/results/val_test_inference`
- analysis 輸出：預設寫到 `artifacts/results/analysis_S2` 與 `artifacts/results/figures`
- 權重與 checkpoints：預設放在 `artifacts/models`
