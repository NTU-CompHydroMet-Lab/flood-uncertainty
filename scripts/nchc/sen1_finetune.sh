#!/usr/bin/env bash
# NCHC version of artifacts/runs/sen1_finetune_chain.sh: the 5 TerraMind Sen1Floods11
# fine-tunes (configurations/terramind_sen1) and then the paired DEUF evaluation.
# Run on the Nano5 login node after `push.sh code`, `push.sh hf` and `push.sh sen1`.
#
# Usage:
#   bash scripts/nchc/sen1_finetune.sh [options] [RUN ...]     # default: all 5 runs + eval
#
# Options:
#   --serial     chain the runs one after another (default: independent jobs; Slurm still
#                runs at most 2 per user at a time, each on its own GPU)
#   --no-eval    do not submit the evaluation job
#   --eval-only  submit only the evaluation job
#   --time T     walltime per fine-tune (default 04:00:00; an unfinished run resumes from
#                last.ckpt when resubmitted)
#   --h200       use normal2 (H200) instead of normal (H100)
#   --out NAME   eval output under artifacts/results/ (default sen1_fusion_compare)
#   --dry-run    print the sbatch commands only
#
# TiM runs (tm_l1c13_tim_wf_s1f11, tm_l1c13_tim_s1f11) are not in the default list; name them,
# e.g. `--out sen1_fusion_compare_tim tm_l1c13_tim_wf_s1f11 tm_l1c13_tim_s1f11`. The eval job
# adds the TiM pairs (bef_tim, direct_tim) when those runs have finished.
#
# The runs do not depend on each other: BEF runs 1-2 start from stage-1 checkpoints that
# already exist; the eval job waits (afterok) for every submitted run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/env.sh"

ALL_RUNS=(tm_l1c13_wf_s1f11 tm_s1grd_kuro_s1f11 tm_l1c13_s1f11 tm_s1grd_s1f11 tm_l1c13_s1grd_s1f11)
TRAIN_SCRIPT=scripts/train/train_edl_terramind_sen1.py
CONFIG_DIR=configurations/terramind_sen1

serial=false
do_eval=true
do_train=true
walltime=04:00:00
partition_args=()
dry_run=false
out=""
runs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) serial=true; shift ;;
    --no-eval) do_eval=false; shift ;;
    --eval-only) do_train=false; shift ;;
    --time) walltime="$2"; shift 2 ;;
    --h200) partition_args=(--partition=normal2); shift ;;
    --out) out="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) runs+=("$1"); shift ;;
  esac
done
[[ ${#runs[@]} -gt 0 ]] || runs=("${ALL_RUNS[@]}")

[[ -n "$NCHC_ACCOUNT" ]] || { echo "set NCHC_ACCOUNT in scripts/nchc/nchc.env" >&2; exit 1; }
cd "$PROJECT_ROOT"
mkdir -p artifacts/nchc_logs

# Fail before queueing anything if an input is missing on this machine
# (the stage-1 checkpoint of every requested run is read from its config).
if $do_train && ! $dry_run; then
  nchc_runtime_env
  missing=0
  inputs=("$NCHC_SEN1/v1.1/splits/flood_handlabeled/flood_train_data.txt")
  for name in "${runs[@]}"; do
    pre="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shared"]["model_params"].get("pretrained_path") or "")' \
      "$CONFIG_DIR/$name.json")"
    [[ -n "$pre" ]] && inputs+=("$pre")
  done
  for path in "${inputs[@]}"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; missing=1; }
  done
  compgen -G "$HF_HOME/hub/models--ibm-esa-geospatial--TerraMind-1.0-base/snapshots/*/TerraMind_v1_base.pt" >/dev/null \
    || { echo "missing: TerraMind weights under $HF_HOME (push.sh hf)" >&2; missing=1; }
  [[ $missing -eq 0 ]] || { echo "upload the missing inputs first (push.sh sen1 / hf)" >&2; exit 1; }
fi

submit() {
  local cmd=(sbatch --parsable --account="$NCHC_ACCOUNT" --chdir="$PROJECT_ROOT" "$@")
  if $dry_run; then
    echo "${cmd[*]}" >&2
    echo "DRYRUN$RANDOM"
  else
    "${cmd[@]}"
  fi
}

if ! $dry_run; then
  summary=""
  if $do_train; then summary="${#runs[@]} fine-tune job(s) (1 GPU, <= $walltime each): ${runs[*]}"; fi
  if $do_eval; then summary="${summary:+$summary; }1 eval job (1 GPU, <= 2 h)"; fi
  nchc_confirm_account "$summary" || exit 1
fi

train_jobs=()
prev=""
if $do_train; then
  for name in "${runs[@]}"; do
    config="$CONFIG_DIR/$name.json"
    [[ -f "$config" ]] || { echo "config not found: $config" >&2; exit 1; }
    dep=()
    $serial && [[ -n "$prev" ]] && dep=(--dependency="afterany:$prev")
    job="$(submit "${partition_args[@]}" "${dep[@]}" --job-name="$name" --time="$walltime" --cpus-per-task=8 \
      --export="ALL,CONFIG=$config,TRAIN_SCRIPT=$TRAIN_SCRIPT,MODE=train,DATA_ROOT=" \
      scripts/nchc/train.sbatch)"
    echo "$name: job $job${dep:+ (after $prev)}"
    train_jobs+=("$job")
    prev="$job"
  done
fi

if $do_eval; then
  dep=()
  if [[ ${#train_jobs[@]} -gt 0 ]]; then
    dep=(--dependency="afterok:$(IFS=:; echo "${train_jobs[*]}")" --kill-on-invalid-dep=yes)
  fi
  job="$(submit "${partition_args[@]}" "${dep[@]}" ${out:+--export=ALL,OUT=artifacts/results/$out} \
    scripts/nchc/sen1_eval.sbatch)"
  echo "eval: job $job${dep:+ (after all fine-tunes succeed)}"
fi
echo "logs: artifacts/nchc_logs/"
