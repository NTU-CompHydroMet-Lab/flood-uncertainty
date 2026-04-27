#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH"
  exit 1
fi

MODEL="${SMOKE_TRAIN_MODEL:-v2}" # v2 | EDL | dropout
SOURCE_DATA_ROOT="${SMOKE_TRAIN_SOURCE_DATA_ROOT:-/home/NAS/homes/cjchen-10025/data/worldfloods_v2/data}"
WORK_ROOT="${SMOKE_TRAIN_WORK_ROOT:-$PROJECT_ROOT/artifacts/smoke/train}"
MINI_DATA_ROOT="${SMOKE_TRAIN_DATA_ROOT:-$WORK_ROOT/data_min}"
CONFIG_DIR="${SMOKE_TRAIN_CONFIG_DIR:-$WORK_ROOT/configs}"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
SMOKE_CONFIG_PATH="${SMOKE_TRAIN_CONFIG_PATH:-$CONFIG_DIR/${MODEL}_smoke_${RUN_ID}.json}"
SMOKE_MODEL_ROOT="${SMOKE_TRAIN_MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/smoke}"
SMOKE_GPUS="${SMOKE_TRAIN_GPUS:-0}"
SMOKE_NUM_WORKERS="${SMOKE_TRAIN_NUM_WORKERS:-2}"
SMOKE_BATCH_SIZE="${SMOKE_TRAIN_BATCH_SIZE:-2}"
export WANDB_MODE="${SMOKE_TRAIN_WANDB_MODE:-offline}"

case "$MODEL" in
  v2)
    BASE_CONFIG="configurations/v2.json"
    TRAIN_SCRIPT="scripts/train/train_v2.py"
    ;;
  EDL)
    BASE_CONFIG="configurations/edl.json"
    TRAIN_SCRIPT="scripts/train/train_edl.py"
    ;;
  dropout)
    BASE_CONFIG="configurations/dropout.json"
    TRAIN_SCRIPT="scripts/train/train_dropout.py"
    ;;
  ensemble)
    echo "SMOKE_TRAIN_MODEL=ensemble is not supported in smoke_train.sh (train script loops 20 models)."
    exit 1
    ;;
  *)
    echo "Unsupported SMOKE_TRAIN_MODEL: $MODEL (allowed: v2, EDL, dropout)"
    exit 1
    ;;
esac

mkdir -p "$WORK_ROOT" "$CONFIG_DIR" "$SMOKE_MODEL_ROOT"

echo "Preparing minimal smoke dataset at: $MINI_DATA_ROOT"
uv run python - <<'PY' "$SOURCE_DATA_ROOT" "$MINI_DATA_ROOT"
import csv
import os
import sys

source_root = sys.argv[1]
mini_root = sys.argv[2]

metadata_path = os.path.join(source_root, "dataset_metadata.csv")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"dataset_metadata.csv not found: {metadata_path}")

with open(metadata_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

selected_by_split = {}
for row in rows:
    split = row.get("split")
    if split in ("train", "val", "test") and split not in selected_by_split:
        selected_by_split[split] = row
    if len(selected_by_split) == 3:
        break

missing = [split for split in ("train", "val", "test") if split not in selected_by_split]
if missing:
    raise RuntimeError(f"Missing split(s) in dataset_metadata.csv: {missing}")

os.makedirs(mini_root, exist_ok=True)
header = rows[0].keys()
mini_metadata_path = os.path.join(mini_root, "dataset_metadata.csv")
with open(mini_metadata_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for split in ("train", "val", "test"):
        writer.writerow(selected_by_split[split])

for split, row in selected_by_split.items():
    event_id = row["event id"]
    for folder in ("S2", "gt"):
        src = os.path.join(source_root, split, folder, f"{event_id}.tif")
        dst = os.path.join(mini_root, split, folder, f"{event_id}.tif")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Required source file not found: {src}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

print(f"Mini metadata: {mini_metadata_path}")
for split in ("train", "val", "test"):
    print(f"{split}: {selected_by_split[split]['event id']}")
PY

echo "Generating smoke config: $SMOKE_CONFIG_PATH"
uv run python - <<'PY' "$BASE_CONFIG" "$SMOKE_CONFIG_PATH" "$MODEL" "$RUN_ID" "$MINI_DATA_ROOT" "$SMOKE_MODEL_ROOT" "$SMOKE_GPUS" "$SMOKE_NUM_WORKERS" "$SMOKE_BATCH_SIZE"
import json
import os
import sys

base_config_path = sys.argv[1]
output_config_path = sys.argv[2]
model_name = sys.argv[3]
run_id = sys.argv[4]
mini_data_root = sys.argv[5]
smoke_model_root = sys.argv[6]
smoke_gpus = int(sys.argv[7])
num_workers = int(sys.argv[8])
batch_size = int(sys.argv[9])

with open(base_config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

shared = cfg.setdefault("shared", {})
data_params = shared.setdefault("data_params", {})
model_params = shared.setdefault("model_params", {})
hyper = model_params.setdefault("hyperparameters", {})

base_experiment = shared.get("experiment_name", f"{model_name}_experiment")
shared["experiment_name"] = f"{base_experiment}_smoke_{run_id}"
shared["gpus"] = smoke_gpus
shared["resume_from_checkpoint"] = False

model_params["model_folder"] = smoke_model_root
data_params["path_to_splits"] = mini_data_root
data_params["train_test_split_file"] = os.path.join(mini_data_root, "train_test_split_from_csv.json")
data_params["num_workers"] = num_workers
data_params["batch_size"] = batch_size

hyper["max_epochs"] = 1
hyper["val_every"] = 1
if "early_stopping_patience" in hyper:
    hyper["early_stopping_patience"] = 1

os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
with open(output_config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

print(f"smoke config written: {output_config_path}")
PY

echo "Running smoke train: model=$MODEL script=$TRAIN_SCRIPT"
uv run python "$TRAIN_SCRIPT" --config "$SMOKE_CONFIG_PATH" --data_root "$MINI_DATA_ROOT"

echo "SMOKE TRAIN PASS"
echo "model outputs root: $SMOKE_MODEL_ROOT"
