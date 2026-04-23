#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ANALYSIS_ENV_FILE:-$SCRIPT_DIR/analysis.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "[WARN] analysis env file not found: $ENV_FILE"
  echo "[WARN] Using fallback defaults from script."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found in PATH."
  exit 1
fi

ANALYSIS_SUBSET="${ANALYSIS_SUBSET:-val}"
ANALYSIS_MODELS="${ANALYSIS_MODELS:-EDL,ensemble,mcdropout}"
ANALYSIS_MAX_FILES="${ANALYSIS_MAX_FILES:-}"
ANALYSIS_RETENTION_STEPS="${ANALYSIS_RETENTION_STEPS:-50}"
ANALYSIS_RETENTION_TARGET="${ANALYSIS_RETENTION_TARGET:-0.9}"
ANALYSIS_PATCH_SIZE="${ANALYSIS_PATCH_SIZE:-3}"
ANALYSIS_RETENTION_GROUP="${ANALYSIS_RETENTION_GROUP:-all}"
ANALYSIS_RUN_RETENTION_PLOT="${ANALYSIS_RUN_RETENTION_PLOT:-1}"
ANALYSIS_RUN_EPI_FP_FN_PLOT="${ANALYSIS_RUN_EPI_FP_FN_PLOT:-1}"

ANALYSIS_DATA_ROOT="${ANALYSIS_DATA_ROOT:-$PROJECT_ROOT/data/worldfloods_v2/data}"
ANALYSIS_PRED_ROOT="${ANALYSIS_PRED_ROOT:-$PROJECT_ROOT/artifacts/results/val_test_inference}"
ANALYSIS_OUTPUT_ROOT="${ANALYSIS_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/analysis_S2}"
ANALYSIS_FIGURE_ROOT="${ANALYSIS_FIGURE_ROOT:-$PROJECT_ROOT/artifacts/results/figures}"
ANALYSIS_COMPARE_OUTPUT_ROOT="${ANALYSIS_COMPARE_OUTPUT_ROOT:-$ANALYSIS_OUTPUT_ROOT/compare}"
ANALYSIS_LOG_DIR="${ANALYSIS_LOG_DIR:-$ANALYSIS_OUTPUT_ROOT/logs}"

mkdir -p "$ANALYSIS_OUTPUT_ROOT" "$ANALYSIS_FIGURE_ROOT" "$ANALYSIS_COMPARE_OUTPUT_ROOT" "$ANALYSIS_LOG_DIR"

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$ANALYSIS_LOG_DIR/run_all_analysis_${ANALYSIS_SUBSET}_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_step() {
  local step_name="$1"
  shift
  local start_ts
  local end_ts
  start_ts="$(date +%s)"

  echo "[$(timestamp)] [START] $step_name"
  if "$@"; then
    end_ts="$(date +%s)"
    echo "[$(timestamp)] [DONE ] $step_name (${end_ts-start_ts}s)"
  else
    local exit_code=$?
    end_ts="$(date +%s)"
    echo "[$(timestamp)] [FAIL ] $step_name (${end_ts-start_ts}s, exit=$exit_code)"
    exit "$exit_code"
  fi
  echo "--------------------------------------------------"
}

IFS=',' read -r -a raw_models <<< "$ANALYSIS_MODELS"
models=()
for raw_model in "${raw_models[@]}"; do
  model="${raw_model//[[:space:]]/}"
  [[ -z "$model" ]] && continue
  case "$model" in
    EDL|ensemble|mcdropout)
      models+=("$model")
      ;;
    *)
      echo "[ERROR] Unsupported model in ANALYSIS_MODELS: $model"
      exit 1
      ;;
  esac
done

if [[ "${#models[@]}" -eq 0 ]]; then
  echo "[ERROR] ANALYSIS_MODELS is empty after parsing."
  exit 1
fi

cd "$PROJECT_ROOT"

pipeline_start="$(date +%s)"
echo "[$(timestamp)] [INFO ] run_all_analysis started"
echo "[$(timestamp)] [INFO ] env file: $ENV_FILE"
echo "[$(timestamp)] [INFO ] log file: $LOG_FILE"
echo "[$(timestamp)] [INFO ] models: ${models[*]}"
echo "[$(timestamp)] [INFO ] subset: $ANALYSIS_SUBSET"
echo "[$(timestamp)] [INFO ] max_files: ${ANALYSIS_MAX_FILES:-ALL}"
echo "[$(timestamp)] [INFO ] pred_root: $ANALYSIS_PRED_ROOT"
echo "[$(timestamp)] [INFO ] analysis_root: $ANALYSIS_OUTPUT_ROOT"
echo "[$(timestamp)] [INFO ] figure_root: $ANALYSIS_FIGURE_ROOT"
echo "[$(timestamp)] [INFO ] compare_root: $ANALYSIS_COMPARE_OUTPUT_ROOT"
echo "--------------------------------------------------"

for model in "${models[@]}"; do
  analysis_args=(
    scripts/analysis/analysis_S2.py
    --model-type "$model"
    --subset "$ANALYSIS_SUBSET"
    --pred-root "$ANALYSIS_PRED_ROOT"
    --output-dir "$ANALYSIS_OUTPUT_ROOT"
    --retention-steps "$ANALYSIS_RETENTION_STEPS"
  )
  if [[ -n "$ANALYSIS_MAX_FILES" ]]; then
    analysis_args+=(--max-files "$ANALYSIS_MAX_FILES")
  fi
  run_step "analysis_S2 model=$model" uv run python "${analysis_args[@]}"
done

for model in "${models[@]}"; do
  pavpu_args=(
    scripts/analysis/compute_pavpu.py
    --model-type "$model"
    --subset "$ANALYSIS_SUBSET"
    --pred-root "$ANALYSIS_PRED_ROOT"
    --analysis-root "$ANALYSIS_OUTPUT_ROOT"
    --patch-size "$ANALYSIS_PATCH_SIZE"
    --retention-target "$ANALYSIS_RETENTION_TARGET"
  )
  if [[ -n "$ANALYSIS_MAX_FILES" ]]; then
    pavpu_args+=(--max-files "$ANALYSIS_MAX_FILES")
  fi
  run_step "compute_pavpu model=$model" uv run python "${pavpu_args[@]}"
done

if [[ "$ANALYSIS_RUN_RETENTION_PLOT" == "1" ]]; then
  run_step \
    "plot_retention_curve_compare group=$ANALYSIS_RETENTION_GROUP" \
    uv run python \
    scripts/analysis/plot_retention_curve_compare.py \
    --analysis-root "$ANALYSIS_OUTPUT_ROOT" \
    --output-dir "$ANALYSIS_FIGURE_ROOT" \
    --group "$ANALYSIS_RETENTION_GROUP"
else
  echo "[$(timestamp)] [SKIP ] plot_retention_curve_compare (ANALYSIS_RUN_RETENTION_PLOT=$ANALYSIS_RUN_RETENTION_PLOT)"
  echo "--------------------------------------------------"
fi

if [[ "$ANALYSIS_RUN_EPI_FP_FN_PLOT" == "1" ]]; then
  compare_args=(
    scripts/analysis/plot_epistemic_fp_fn_compare.py
    --subset "$ANALYSIS_SUBSET"
    --data-root "$ANALYSIS_DATA_ROOT"
    --pred-root "$ANALYSIS_PRED_ROOT"
    --output-root "$ANALYSIS_COMPARE_OUTPUT_ROOT"
  )
  if [[ -n "$ANALYSIS_MAX_FILES" ]]; then
    compare_args+=(--max-files "$ANALYSIS_MAX_FILES")
  fi
  run_step "plot_epistemic_fp_fn_compare" uv run python "${compare_args[@]}"
else
  echo "[$(timestamp)] [SKIP ] plot_epistemic_fp_fn_compare (ANALYSIS_RUN_EPI_FP_FN_PLOT=$ANALYSIS_RUN_EPI_FP_FN_PLOT)"
  echo "--------------------------------------------------"
fi

pipeline_end="$(date +%s)"
echo "[$(timestamp)] [INFO ] run_all_analysis finished (${pipeline_end-pipeline_start}s)"
echo "[$(timestamp)] [INFO ] full log: $LOG_FILE"
