#!/usr/bin/env bash
# Run the water-only invalid-KL ablation arms one after another on one GPU.
#
#   bash scripts/train/run_edl_water_kl_ablation.sh                  # all four arms
#   ARMS="kl_002_const" bash scripts/train/run_edl_water_kl_ablation.sh
#   OUTPUT_ROOT=artifacts/models/edl_water_kl_smoke bash scripts/train/run_edl_water_kl_ablation.sh \
#     --skip-val --max-epochs 1 --limit-train-batches 20 --num-workers 0
#
# Extra arguments are passed unchanged to train_edl_water_kl.py for every arm.
# W&B uploads online by default (each run still keeps a local wandb/ copy); set
# WANDB_MODE=offline to sync later. --skip-val runs keep W&B disabled.
set -euo pipefail
cd "$(dirname "$0")/../.."

ARMS="${ARMS:-kl_off kl_shared_beta kl_002_beta kl_002_const}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/models/edl_water_kl_ablation}"
export WANDB_MODE="${WANDB_MODE:-online}"

# A forwarded --config or --output-dir would override every arm's own setting.
for arg in "$@"; do
  case "$arg" in
    --config*|--output-dir*)
      echo "Do not pass ${arg}; choose arms with ARMS and the parent directory with OUTPUT_ROOT." >&2
      exit 2
      ;;
  esac
done

for arm in $ARMS; do
  echo "=== ${arm} ==="
  uv run python scripts/train/train_edl_water_kl.py \
    --config "configurations/edl_water_kl_ablation/${arm}.json" \
    --output-dir "${OUTPUT_ROOT}/${arm}" "$@"
done
