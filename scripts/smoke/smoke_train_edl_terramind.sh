#!/usr/bin/env bash
# Smoke-train the EDL + TerraMind model on a 1-tile-per-split dataset.
#
# Thin wrapper around smoke_train.sh (SMOKE_TRAIN_MODEL=EDL_TERRAMIND) that also
# copes with a data root whose *train* split has no S2 imagery (the copy under
# /home/NAS/homes/cjchen-10025 only ships S2 for val/test). In that case one
# val tile is borrowed as the train sample; that is fine for a plumbing check,
# and obviously not for a real experiment.
#
# Usage:
#   bash scripts/smoke/smoke_train_edl_terramind.sh
# Env (all optional):
#   SMOKE_TRAIN_SOURCE_DATA_ROOT  full WorldFloods v2 root (default: ~/data/worldfloods_v2 symlink root)
#   SMOKE_TRAIN_WORK_ROOT         where mini datasets / configs go (default: artifacts/smoke/train)
#   SMOKE_TRAIN_GPUS, SMOKE_TRAIN_BATCH_SIZE, SMOKE_TRAIN_NUM_WORKERS, SMOKE_TRAIN_WANDB_MODE
#                                 passed through to smoke_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_DATA_ROOT="${SMOKE_TRAIN_SOURCE_DATA_ROOT:-/home/NAS/house/ycchen-10014/data/worldfloods_v2}"
WORK_ROOT="${SMOKE_TRAIN_WORK_ROOT:-$PROJECT_ROOT/artifacts/smoke/train}"
SRC_MINI_ROOT="$WORK_ROOT/data_src_edl_terramind"

mkdir -p "$WORK_ROOT"

echo "Preparing smoke source dataset at: $SRC_MINI_ROOT"
uv run python - <<'PY' "$SOURCE_DATA_ROOT" "$SRC_MINI_ROOT"
import csv, os, sys

source_root, mini_root = sys.argv[1], sys.argv[2]
meta = os.path.join(source_root, "dataset_metadata.csv")
if not os.path.exists(meta):
    raise FileNotFoundError(meta)

with open(meta, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def has_files(row):
    eid = row["event id"]
    return all(os.path.exists(os.path.join(source_root, row["split"], m, f"{eid}.tif")) for m in ("S2", "gt"))

chosen = {}
for row in rows:
    s = row["split"]
    if s in ("train", "val", "test") and s not in chosen and has_files(row):
        chosen[s] = dict(row)

borrowed = []
for s in ("train", "val", "test"):
    if s not in chosen:
        donor = chosen.get("val") or chosen.get("test")
        if donor is None:
            raise RuntimeError(f"No usable tile for split '{s}' and no donor split available")
        r = dict(donor); r["split"] = s; r["_src_split"] = donor["split"]
        chosen[s] = r
        borrowed.append(f"{s}<-{donor['split']}")

os.makedirs(mini_root, exist_ok=True)
fields = list(rows[0].keys())
with open(os.path.join(mini_root, "dataset_metadata.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for s in ("train", "val", "test"):
        w.writerow(chosen[s])

for s in ("train", "val", "test"):
    row = chosen[s]; eid = row["event id"]; src_split = row.get("_src_split", s)
    for m in ("S2", "gt"):
        src = os.path.join(source_root, src_split, m, f"{eid}.tif")
        dst = os.path.join(mini_root, s, m, f"{eid}.tif")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f"{s}: {eid}" + (f"  (borrowed from {src_split})" if src_split != s else ""))
if borrowed:
    print(f"WARNING: borrowed tiles for splits {borrowed}; plumbing check only, not a real experiment")
PY

echo "Delegating to smoke_train.sh (model=EDL_TERRAMIND)"
SMOKE_TRAIN_MODEL=EDL_TERRAMIND \
SMOKE_TRAIN_SOURCE_DATA_ROOT="$SRC_MINI_ROOT" \
SMOKE_TRAIN_WORK_ROOT="$WORK_ROOT" \
bash "$SCRIPT_DIR/smoke_train.sh"
