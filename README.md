# flood-uncertainty

Flood segmentation uncertainty 專案（`v2` / `EDL` / `ensemble` / `MC dropout`）。

## 1. 專案結構

- `flood_uncertainty/`: 核心套件（models/losses/metrics/inference/utils）
- `scripts/`: train / infer / eval 入口
- `configurations/`: 每個方法一份 config（`shared/train/infer/validate_only`）
- `artifacts/`: 權重與輸出（已在 `.gitignore`）
- `ml4floods/`, `mlguess/`: 本地依賴

## 2. 環境

目前流程以既有 conda 環境執行（例如 `ml4floods`）：

```bash
conda activate ml4floods
```

## 3. Config mode

每份 `configurations/*.json` 都使用相同 mode 架構：

- `train`: 訓練
- `infer`: 推論
- `validate_only`: 只跑 validation（目前主要用於 EDL）

## 4. 常用指令

### 4.1 訓練

```bash
python scripts/train_v2.py --config configurations/v2.json --mode train
python scripts/train_edl.py --config configurations/edl.json --mode train
python scripts/train_dropout.py --config configurations/dropout.json --mode train
python scripts/train_ensemble.py --config configurations/ensemble.json --mode train
```

可選覆寫資料根目錄：

```bash
python scripts/train_edl.py --config configurations/edl.json --mode train --data_root /path/to/worldfloods_v2/data
```

### 4.2 只驗證（EDL）

```bash
python scripts/train_edl.py --config configurations/edl.json --mode validate_only
```

### 4.3 推論

```bash
python scripts/run_inference.py --model_type v2 --config_v2 configurations/v2.json
python scripts/run_inference.py --model_type EDL --config_edl configurations/edl.json
python scripts/run_inference_ensemble.py --mode ensemble --config configurations/ensemble.json
python scripts/run_inference_ensemble.py --mode mcdropout --config configurations/dropout.json
```

### 4.4 評估

```bash
python scripts/eval_metrics.py --model_type v2 --subset val
python scripts/eval_metrics.py --model_type EDL --subset val
```

## 5. 一鍵 Smoke（v2 infer + eval）

```bash
bash scripts/smoke.sh
```

可選環境變數（不帶就用預設）：

```bash
SMOKE_SUBSET=val SMOKE_MAX_FILES=1 SMOKE_CONFIG=configurations/v2.json bash scripts/smoke.sh
```

## 6. Analysis 指令

```bash
python analysis/analysis_S2.py --model-type EDL --subset val --max-files 1
python analysis/compute_pavpu.py --model-type EDL --subset val --max-files 1
python analysis/plot_retention_curve_compare.py --group all
python analysis/plot_epistemic_fp_fn_compare.py --subset val --max-files 1
```

## 7. 路徑說明

- 資料根目錄：預設由 config 的 `data_params.path_to_splits` 決定（可用 `--data_root` 覆寫）
- 推論與評估輸出：預設寫到 `artifacts/results/val_test_inference`
- analysis 輸出：預設寫到 `artifacts/results/analysis_S2` 與 `artifacts/results/figures`
- 權重與 checkpoints：預設放在 `artifacts/models`
