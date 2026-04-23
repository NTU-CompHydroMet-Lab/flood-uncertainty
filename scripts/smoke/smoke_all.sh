#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_INFER_EVAL="${SMOKE_ALL_RUN_INFER_EVAL:-1}"
RUN_TRAIN="${SMOKE_ALL_RUN_TRAIN:-1}"
RUN_ANALYSIS="${SMOKE_ALL_RUN_ANALYSIS:-1}"
ANALYSIS_MODEL_TYPE="${SMOKE_ANALYSIS_MODEL_TYPE:-EDL}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_step() {
  local name="$1"
  shift
  local start
  start="$(date +%s)"
  echo "[$(timestamp)] [START] $name"
  "$@"
  local end
  end="$(date +%s)"
  echo "[$(timestamp)] [DONE ] $name ($((end - start))s)"
  echo "--------------------------------------------------"
}

if [[ "$RUN_INFER_EVAL" == "1" ]]; then
  run_step "smoke_infer_eval" bash scripts/smoke/smoke_infer_eval.sh
else
  echo "[$(timestamp)] [SKIP ] smoke_infer_eval"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  run_step "smoke_train" bash scripts/smoke/smoke_train.sh
else
  echo "[$(timestamp)] [SKIP ] smoke_train"
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  if [[ "$ANALYSIS_MODEL_TYPE" == "EDL" ]]; then
    run_step "smoke_analysis" bash scripts/smoke/smoke_analysis.sh
  else
    run_step "smoke_analysis" env SMOKE_ANALYSIS_PREPARE_INFER="${SMOKE_ANALYSIS_PREPARE_INFER:-0}" bash scripts/smoke/smoke_analysis.sh
  fi
else
  echo "[$(timestamp)] [SKIP ] smoke_analysis"
fi

echo "SMOKE ALL PASS"
