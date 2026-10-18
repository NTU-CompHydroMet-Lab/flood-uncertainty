#!/usr/bin/env bash
# Download finished experiments from Nano5 into $PULL_DEST (default $LAB_REPO/artifacts/nchc),
# kept apart from artifacts/models so runs with the same name on the lab GPU are never overwritten.
#
# Usage (run on the lab machine):
#   bash scripts/nchc/pull.sh EXPERIMENT [...]               # whole experiment dir (all ckpts), e.g.
#                                                            #   terramind_sen1/tm_l1c13_wf_s1f11
#   bash scripts/nchc/pull.sh --logs-only EXPERIMENT [...]   # lightning_logs + wandb only
#   bash scripts/nchc/pull.sh --file REL_PATH                # one file under $NCHC_REPO/artifacts/models
#   bash scripts/nchc/pull.sh --results NAME                 # $NCHC_REPO/artifacts/results/NAME
# Slurm logs (artifacts/nchc_logs) are always fetched.
# Offline W&B runs can then be uploaded with:
#   wandb sync artifacts/nchc/models/<exp>/wandb/offline-run-*
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

read -r -a SSH_OPTS <<<"$(nchc_ssh_opts "$NCHC_XFER_PORT")"
SFTP_OPTS=("-P" "$NCHC_XFER_PORT" "${SSH_OPTS[@]:2}")
REMOTE="$NCHC_USER@$NCHC_HOST"
REMOTE_MODELS="$NCHC_REPO/artifacts/models"
PULL_DEST="${PULL_DEST:-$LAB_REPO/artifacts/nchc}"
LOCAL_MODELS="$PULL_DEST/models"

logs_only=false
files=()
results=()
exps=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --logs-only) logs_only=true; shift ;;
    --file) files+=("$2"); shift 2 ;;
    --results) results+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) exps+=("$1"); shift ;;
  esac
done
[[ ${#exps[@]} -gt 0 || ${#files[@]} -gt 0 || ${#results[@]} -gt 0 ]] || { sed -n '2,14p' "$0"; exit 2; }

ssh "${SSH_OPTS[@]}" -O check "$REMOTE" 2>/dev/null || ssh "${SSH_OPTS[@]}" -fN "$REMOTE"

mkdir -p "$LOCAL_MODELS" "$PULL_DEST/results"
batch="$(mktemp)"
trap 'rm -f "$batch"' EXIT
{
  echo "lcd $PULL_DEST"
  echo "-get -r $NCHC_REPO/artifacts/nchc_logs"
  for exp in "${exps[@]}"; do
    if $logs_only; then
      mkdir -p "$LOCAL_MODELS/$exp"
      echo "lcd $LOCAL_MODELS/$exp"
      echo "-get -r $REMOTE_MODELS/$exp/lightning_logs"
      echo "-get -r $REMOTE_MODELS/$exp/wandb"
    else
      mkdir -p "$LOCAL_MODELS/$(dirname "$exp")"
      echo "lcd $LOCAL_MODELS/$(dirname "$exp")"
      # '-': files already complete locally make `-a` report an error; keep going.
      echo "-get -r -a $REMOTE_MODELS/$exp"
    fi
  done
  for rel in "${files[@]}"; do
    mkdir -p "$LOCAL_MODELS/$(dirname "$rel")"
    echo "get -a \"$REMOTE_MODELS/$rel\" \"$LOCAL_MODELS/$rel\""
  done
  for name in "${results[@]}"; do
    echo "lcd $PULL_DEST/results"
    echo "-get -r $NCHC_REPO/artifacts/results/$name"
  done
} >"$batch"
sftp "${SFTP_OPTS[@]}" -b "$batch" "$REMOTE"
echo "pulled into $PULL_DEST"
