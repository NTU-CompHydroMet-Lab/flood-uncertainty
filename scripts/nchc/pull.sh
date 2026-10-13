#!/usr/bin/env bash
# Download finished experiments from Nano5 into the lab checkout ($LAB_REPO/artifacts).
#
# Usage (run on the lab machine):
#   bash scripts/nchc/pull.sh EXPERIMENT_NAME [...]               # whole experiment dir (all ckpts)
#   bash scripts/nchc/pull.sh --logs-only EXPERIMENT_NAME [...]   # lightning_logs + wandb only
#   bash scripts/nchc/pull.sh --file REL_PATH                     # one file under $NCHC_REPO/artifacts/models,
#                                                                 #   e.g. <exp>/checkpoint/epoch=12-step=....ckpt
# Slurm logs (artifacts/nchc_logs) are always fetched.
# Offline W&B runs can then be uploaded with:
#   wandb sync artifacts/models/<exp>/wandb/offline-run-*
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

read -r -a SSH_OPTS <<<"$(nchc_ssh_opts "$NCHC_XFER_PORT")"
SFTP_OPTS=("-P" "$NCHC_XFER_PORT" "${SSH_OPTS[@]:2}")
REMOTE="$NCHC_USER@$NCHC_HOST"
REMOTE_MODELS="$NCHC_REPO/artifacts/models"
LOCAL_MODELS="$LAB_REPO/artifacts/models"

logs_only=false
files=()
exps=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --logs-only) logs_only=true; shift ;;
    --file) files+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) exps+=("$1"); shift ;;
  esac
done
[[ ${#exps[@]} -gt 0 || ${#files[@]} -gt 0 ]] || { sed -n '2,12p' "$0"; exit 2; }

ssh "${SSH_OPTS[@]}" -O check "$REMOTE" 2>/dev/null || ssh "${SSH_OPTS[@]}" -fN "$REMOTE"

mkdir -p "$LOCAL_MODELS" "$LAB_REPO/artifacts/nchc_logs"
batch="$(mktemp)"
trap 'rm -f "$batch"' EXIT
{
  echo "lcd $LAB_REPO/artifacts"
  echo "-get -r $NCHC_REPO/artifacts/nchc_logs"
  for exp in "${exps[@]}"; do
    if $logs_only; then
      mkdir -p "$LOCAL_MODELS/$exp"
      echo "lcd $LOCAL_MODELS/$exp"
      echo "-get -r $REMOTE_MODELS/$exp/lightning_logs"
      echo "-get -r $REMOTE_MODELS/$exp/wandb"
    else
      echo "lcd $LOCAL_MODELS"
      # '-': files already complete locally make `-a` report an error; keep going.
      echo "-get -r -a $REMOTE_MODELS/$exp"
    fi
  done
  for rel in "${files[@]}"; do
    mkdir -p "$LOCAL_MODELS/$(dirname "$rel")"
    echo "get -a $REMOTE_MODELS/$rel $LOCAL_MODELS/$rel"
  done
} >"$batch"
sftp "${SFTP_OPTS[@]}" -b "$batch" "$REMOTE"
echo "pulled into $LAB_REPO/artifacts"
