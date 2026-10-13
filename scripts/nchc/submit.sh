#!/usr/bin/env bash
# Submit training jobs on the Nano5 login node, strictly one at a time.
#
# Usage:
#   bash scripts/nchc/submit.sh --smoke
#   bash scripts/nchc/submit.sh [options] CONFIG [CONFIG ...]
#
# Options:
#   --chunks N     submit N 48 h chunks per config; each resumes from last.ckpt (default 1)
#   --after JOBID  start the chain after an existing job
#   --script PATH  training script (default scripts/train/train_edl_terramind.py)
#   --mode MODE    train | validate_only (default train)
#   --data-root D  pass --data_root D to the training script
#   --dev          use the dev partition (2 h limit) instead of normal
#   --h200         use normal2 (H200) instead of normal (H100)
#   --dry-run      print the sbatch commands only
#
# Every job depends on the previous one with afterany, so runs never overlap and a
# failed or timed-out job does not block the rest of the queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/env.sh"

chunks=1
after=""
train_script="scripts/train/train_edl_terramind.py"
mode="train"
data_root=""
partition_args=()
dry_run=false
smoke=false
configs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke=true; shift ;;
    --chunks) chunks="$2"; shift 2 ;;
    --after) after="$2"; shift 2 ;;
    --script) train_script="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --dev) partition_args=(--partition=dev --time=02:00:00); shift ;;
    --h200) partition_args=(--partition=normal2); shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) configs+=("$1"); shift ;;
  esac
done

[[ -n "$NCHC_ACCOUNT" ]] || { echo "set NCHC_ACCOUNT in scripts/nchc/nchc.env" >&2; exit 1; }
cd "$PROJECT_ROOT"
mkdir -p artifacts/nchc_logs

submit() {
  local cmd=(sbatch --parsable --account="$NCHC_ACCOUNT" --chdir="$PROJECT_ROOT" "$@")
  if $dry_run; then
    echo "${cmd[*]}" >&2
    echo "DRYRUN$RANDOM"
  else
    "${cmd[@]}"
  fi
}

if $smoke; then
  job="$(submit scripts/nchc/smoke.sbatch)"
  echo "smoke job $job -> artifacts/nchc_logs/flood-smoke-$job.out"
  exit 0
fi

[[ ${#configs[@]} -gt 0 ]] || { sed -n '2,20p' "$0"; exit 2; }

prev="$after"
for config in "${configs[@]}"; do
  [[ -f "$config" ]] || { echo "config not found: $config" >&2; exit 1; }
  name="$(basename "$config" .json)"
  for ((i = 1; i <= chunks; i++)); do
    dep=()
    [[ -n "$prev" ]] && dep=(--dependency="afterany:$prev")
    job="$(submit "${partition_args[@]}" "${dep[@]}" --job-name="$name" \
      --export="ALL,CONFIG=$config,TRAIN_SCRIPT=$train_script,MODE=$mode,DATA_ROOT=$data_root" \
      scripts/nchc/train.sbatch)"
    echo "$name chunk $i/$chunks: job $job${prev:+ (after $prev)}"
    prev="$job"
  done
done
