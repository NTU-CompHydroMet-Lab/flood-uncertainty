#!/bin/bash
# One training chunk on Nano5. Submit through scripts/nchc/submit.sh, which sets
# --account, --chdir (repo root) and the variables below.
#
# Env:
#   CONFIG        configurations/... json (required)
#   TRAIN_SCRIPT  default scripts/train/train_edl_terramind.py
#   MODE          train | validate_only (default train)
#   DATA_ROOT     optional --data_root override (default: config path, remapped)
#   RESUME_CKPT   optional; default resumes from <experiment>/checkpoint/last.ckpt if present
#
# A chunk that finishes training writes <experiment>/.nchc_done; later chunks of the
# same experiment then exit immediately (delete the marker to retrain from scratch).
#SBATCH --job-name=flood-train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=artifacts/nchc_logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run via sbatch}"
source scripts/nchc/env.sh
nchc_runtime_env
nchc_offline_env

: "${CONFIG:?CONFIG is required}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/train_edl_terramind.py}"
MODE="${MODE:-train}"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

EXP_DIR="$("$PY" - "$CONFIG" "$MODE" <<'PY'
import sys
from flood_uncertainty.utils.config_loader import load_mode_config
config = load_mode_config(sys.argv[1], mode=sys.argv[2])
print(f"{config.model_params.model_folder}/{config.experiment_name}")
PY
)"

echo "job=$SLURM_JOB_ID node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
echo "config=$CONFIG script=$TRAIN_SCRIPT mode=$MODE"
echo "experiment dir=$EXP_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "$MODE" == train && -f "$EXP_DIR/.nchc_done" ]]; then
  echo "already finished ($EXP_DIR/.nchc_done); nothing to do"
  exit 0
fi

if [[ -z "${RESUME_CKPT:-}" && "$MODE" == train && -f "$EXP_DIR/checkpoint/last.ckpt" ]]; then
  RESUME_CKPT="$EXP_DIR/checkpoint/last.ckpt"
fi

args=(--config "$CONFIG" --mode "$MODE")
[[ -n "${DATA_ROOT:-}" ]] && args+=(--data_root "$DATA_ROOT")
[[ -n "${RESUME_CKPT:-}" ]] && args+=(--resume_ckpt "$RESUME_CKPT") && echo "resuming from $RESUME_CKPT"

start=$(date +%s)
"$PY" "$TRAIN_SCRIPT" "${args[@]}"
echo "wall time: $(( ($(date +%s) - start) / 60 )) min"

if [[ "$MODE" == train ]]; then
  touch "$EXP_DIR/.nchc_done"
fi
