#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${SMOKE_CONFIG:-configurations/v2.json}"
SUBSET="${SMOKE_SUBSET:-val}"
MAX_FILES="${SMOKE_MAX_FILES:-1}"
OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$PROJECT_ROOT/artifacts/results/val_test_inference}"
DATA_ROOT="${SMOKE_DATA_ROOT:-}"
PYTHON_BIN="${SMOKE_PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

INFER_ARGS=(
  --model_type v2
  --config_v2 "$CONFIG_PATH"
  --output_dir "$OUTPUT_DIR"
  --subsets "$SUBSET"
  --max_files "$MAX_FILES"
  --no_save_plot
  --save_pred_tif
)
if [[ -n "$DATA_ROOT" ]]; then
  INFER_ARGS+=(--data_root "$DATA_ROOT")
fi

EVAL_ARGS=(
  --model_type v2
  --config "$CONFIG_PATH"
  --config_mode infer
  --output_dir "$OUTPUT_DIR"
  --subset "$SUBSET"
  --max_files "$MAX_FILES"
  --no_plot_png
  --no_save_tif
)
if [[ -n "$DATA_ROOT" ]]; then
  EVAL_ARGS+=(--data_root "$DATA_ROOT")
fi

"$PYTHON_BIN" scripts/run_inference.py "${INFER_ARGS[@]}"
"$PYTHON_BIN" scripts/eval_metrics.py "${EVAL_ARGS[@]}"

METRICS_CSV="$OUTPUT_DIR/$SUBSET/v2/metrics_v2.csv"
if [[ ! -f "$METRICS_CSV" ]]; then
  echo "SMOKE FAIL: metrics file not found: $METRICS_CSV"
  exit 1
fi

echo "SMOKE PASS: $METRICS_CSV"
