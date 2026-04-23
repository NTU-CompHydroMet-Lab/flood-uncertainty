#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="${SMOKE_INFER_MODEL_TYPE:-v2}"
CONFIG_PATH="${SMOKE_INFER_CONFIG:-}"
SUBSET="${SMOKE_INFER_SUBSET:-val}"
MAX_FILES="${SMOKE_INFER_MAX_FILES:-1}"
OUTPUT_DIR="${SMOKE_INFER_OUTPUT_DIR:-$PROJECT_ROOT/artifacts/results/val_test_inference}"
DATA_ROOT="${SMOKE_INFER_DATA_ROOT:-}"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH"
  exit 1
fi

case "$MODEL_TYPE" in
  v2)
    CONFIG_PATH="${CONFIG_PATH:-configurations/v2.json}"
    INFER_MODEL_ARGS=(--model_type v2 --config_v2 "$CONFIG_PATH")
    ;;
  EDL)
    CONFIG_PATH="${CONFIG_PATH:-configurations/edl.json}"
    INFER_MODEL_ARGS=(--model_type EDL --config_edl "$CONFIG_PATH")
    ;;
  *)
    echo "Unsupported SMOKE_INFER_MODEL_TYPE: $MODEL_TYPE (allowed: v2, EDL)"
    exit 1
    ;;
esac

INFER_ARGS=(
  "${INFER_MODEL_ARGS[@]}"
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
  --model_type "$MODEL_TYPE"
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

uv run python scripts/inference/run_inference.py "${INFER_ARGS[@]}"
uv run python scripts/eval/eval_metrics.py "${EVAL_ARGS[@]}"

METRICS_CSV="$OUTPUT_DIR/$SUBSET/$MODEL_TYPE/metrics_${MODEL_TYPE}.csv"
if [[ ! -f "$METRICS_CSV" ]]; then
  echo "SMOKE FAIL: metrics file not found: $METRICS_CSV"
  exit 1
fi

echo "SMOKE PASS: $METRICS_CSV"
