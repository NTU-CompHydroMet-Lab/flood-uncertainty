# flood-uncertainty

Flood segmentation uncertainty 專案（`v2` / `EDL` / `ensemble` / `MC dropout`）。

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
```

可選覆寫資料根目錄：

```bash
uv run python scripts/train/train_edl.py --config configurations/edl.json --mode train --data_root /path/to/worldfloods_v2/data
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
uv run python scripts/inference/run_inference_ensemble.py --mode ensemble --config configurations/ensemble.json
uv run python scripts/inference/run_inference_ensemble.py --mode mcdropout --config configurations/dropout.json
uv run python scripts/inference/run_inference_sar.py input.tif output.tif --config configurations/edl_sar.json --weights /path/to/sar.ckpt
```
如果需要實際權重路徑，請洽jerrychlun

### 4.4 評估

```bash
uv run python scripts/eval/eval_metrics.py --model_type v2 --subset val
uv run python scripts/eval/eval_metrics.py --model_type EDL --subset val
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

```bash
uv run python scripts/analysis/analysis_S2.py --model-type EDL --subset val --max-files 1
uv run python scripts/analysis/compute_pavpu.py --model-type EDL --subset val --max-files 1
uv run python scripts/analysis/plot_retention_curve_compare.py --group all
uv run python scripts/analysis/plot_epistemic_fp_fn_compare.py --subset val --max-files 1
```

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

## 8. 路徑說明

- 資料根目錄：預設由 config 的 `data_params.path_to_splits` 決定（可用 `--data_root` 覆寫）
- 推論與評估輸出：預設寫到 `artifacts/results/val_test_inference`
- analysis 輸出：預設寫到 `artifacts/results/analysis_S2` 與 `artifacts/results/figures`
- 權重與 checkpoints：預設放在 `artifacts/models`
