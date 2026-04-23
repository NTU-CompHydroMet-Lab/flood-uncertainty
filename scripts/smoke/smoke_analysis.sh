#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH"
  exit 1
fi

MODEL_TYPE="${SMOKE_ANALYSIS_MODEL_TYPE:-EDL}"   # EDL | ensemble | mcdropout
SUBSET="${SMOKE_ANALYSIS_SUBSET:-val}"
MAX_FILES="${SMOKE_ANALYSIS_MAX_FILES:-1}"
PRED_ROOT="${SMOKE_ANALYSIS_PRED_ROOT:-$PROJECT_ROOT/artifacts/results/val_test_inference}"
ANALYSIS_ROOT="${SMOKE_ANALYSIS_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/analysis_S2}"
DATA_ROOT="${SMOKE_ANALYSIS_DATA_ROOT:-/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data}"
RETENTION_STEPS="${SMOKE_ANALYSIS_RETENTION_STEPS:-50}"
RETENTION_TARGET="${SMOKE_ANALYSIS_RETENTION_TARGET:-0.9}"
PATCH_SIZE="${SMOKE_ANALYSIS_PATCH_SIZE:-3}"
PREPARE_INFER="${SMOKE_ANALYSIS_PREPARE_INFER:-1}"
RUN_CROSS_MODEL_PLOTS="${SMOKE_ANALYSIS_RUN_CROSS_MODEL_PLOTS:-0}"

if [[ "$PREPARE_INFER" == "1" && "$MODEL_TYPE" == "EDL" ]]; then
  INFER_CONFIG="${SMOKE_ANALYSIS_INFER_CONFIG:-configurations/edl.json}"
  echo "Preparing prediction artifacts for analysis (EDL, subset=$SUBSET, max_files=$MAX_FILES)"
  uv run python scripts/inference/run_inference.py \
    --model_type EDL \
    --config_edl "$INFER_CONFIG" \
    --subsets "$SUBSET" \
    --max_files "$MAX_FILES" \
    --output_dir "$PRED_ROOT" \
    --data_root "$DATA_ROOT" \
    --no_save_plot \
    --save_pred_tif

  uv run python scripts/eval/eval_metrics.py \
    --model_type EDL \
    --config "$INFER_CONFIG" \
    --config_mode infer \
    --subset "$SUBSET" \
    --max_files "$MAX_FILES" \
    --output_dir "$PRED_ROOT" \
    --data_root "$DATA_ROOT" \
    --no_plot_png \
    --save_tif
elif [[ "$PREPARE_INFER" == "1" ]]; then
  echo "SMOKE_ANALYSIS_PREPARE_INFER=1 is only supported for MODEL_TYPE=EDL."
  echo "MODEL_TYPE=$MODEL_TYPE requires existing prediction artifacts under:"
  echo "  $PRED_ROOT/$SUBSET/$MODEL_TYPE"
  echo "Set SMOKE_ANALYSIS_PREPARE_INFER=0 to run analysis only."
  exit 1
fi

echo "Running analysis_S2 (model=$MODEL_TYPE)"
uv run python scripts/analysis/analysis_S2.py \
  --model-type "$MODEL_TYPE" \
  --subset "$SUBSET" \
  --pred-root "$PRED_ROOT" \
  --output-dir "$ANALYSIS_ROOT" \
  --retention-steps "$RETENTION_STEPS" \
  --max-files "$MAX_FILES"

echo "Running compute_pavpu (model=$MODEL_TYPE)"
uv run python scripts/analysis/compute_pavpu.py \
  --model-type "$MODEL_TYPE" \
  --subset "$SUBSET" \
  --pred-root "$PRED_ROOT" \
  --analysis-root "$ANALYSIS_ROOT" \
  --patch-size "$PATCH_SIZE" \
  --retention-target "$RETENTION_TARGET" \
  --max-files "$MAX_FILES"

if [[ "$RUN_CROSS_MODEL_PLOTS" == "1" ]]; then
  FIGURE_ROOT="${SMOKE_ANALYSIS_FIGURE_ROOT:-$PROJECT_ROOT/artifacts/results/figures}"
  COMPARE_OUTPUT="${SMOKE_ANALYSIS_COMPARE_OUTPUT:-$ANALYSIS_ROOT/compare}"

  echo "Running cross-model retention plot"
  uv run python scripts/analysis/plot_retention_curve_compare.py \
    --analysis-root "$ANALYSIS_ROOT" \
    --output-dir "$FIGURE_ROOT" \
    --group all

  echo "Running cross-model epistemic FP/FN compare plot"
  uv run python scripts/analysis/plot_epistemic_fp_fn_compare.py \
    --subset "$SUBSET" \
    --data-root "$DATA_ROOT" \
    --pred-root "$PRED_ROOT" \
    --output-root "$COMPARE_OUTPUT" \
    --max-files "$MAX_FILES"
fi

if ! ls "$ANALYSIS_ROOT/$MODEL_TYPE"/retention_*.csv >/dev/null 2>&1; then
  echo "SMOKE ANALYSIS FAIL: no retention CSV found in $ANALYSIS_ROOT/$MODEL_TYPE"
  exit 1
fi

echo "SMOKE ANALYSIS PASS: $ANALYSIS_ROOT/$MODEL_TYPE"
