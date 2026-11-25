#!/usr/bin/env bash
# Item 13 B, steps 1-2 for the water-only invalid-KL ablation arms:
#   full-image inference on val and test -> confusion maps -> val retention curves -> three test PAvPU variants.
#
#   bash scripts/analysis/run_kl_ablation_eval.sh                   # each arm's best validation epoch
#   CHECKPOINT=last bash scripts/analysis/run_kl_ablation_eval.sh   # sensitivity analysis
#   ARMS="kl_off kl_002_const" bash scripts/analysis/run_kl_ablation_eval.sh
#
# Outputs: $EVAL_ROOT/<checkpoint>/<arm>/{predictions,analysis}, one directory per arm and checkpoint choice.
# PAvPU settings match the earlier EDL / ensemble / MC dropout tables: patch IoU 0.9, patch 3, retention 0.9.
set -euo pipefail
cd "$(dirname "$0")/../.."

ARMS="${ARMS:-kl_off kl_shared_beta kl_002_beta kl_002_const}"
CHECKPOINT="${CHECKPOINT:-best}"
ABLATION_ROOT="artifacts/models/edl_water_kl_ablation"
EVAL_ROOT="${EVAL_ROOT:-artifacts/results/edl_water_kl_ablation}"
PAVPU_ACC_THRESHOLD=0.9
PATCH_SIZE=3
RETENTION=0.9

for arm in $ARMS; do
  echo "=== ${arm} (${CHECKPOINT}) ==="
  out="${EVAL_ROOT}/${CHECKPOINT}/${arm}"
  pred_root="${out}/predictions"
  analysis_root="${out}/analysis"
  config="configurations/edl_water_kl_ablation/${arm}.json"

  # Resolve the checkpoint only from a finished, full ablation run; record it next to the outputs.
  ckpt=$(uv run python - "$ABLATION_ROOT" "$arm" "$CHECKPOINT" "$config" <<'EOF'
import hashlib, json, sys
from pathlib import Path
from scripts.eval.plot_val_kl_maps import checkpoint_path, latest_run

root, arm, choice, config_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
config = json.loads(config_path.read_text())["shared"]
run = latest_run(root, arm)
manifest = json.loads((run / "train_manifest.json").read_text())
finished = len(list((run / "checkpoint" / "epochs").glob("epoch=*.ckpt")))
split_sha = hashlib.sha256(Path(config["data_params"]["train_test_split_file"]).read_bytes()).hexdigest()
problems = {
    "epochs finished": (finished, config["model_params"]["hyperparameters"]["max_epochs"]),
    "manifest max_epochs": (manifest["max_epochs"], config["model_params"]["hyperparameters"]["max_epochs"]),
    "validation enabled": (manifest["validation_enabled"], True),
    "limit_train_batches": (manifest["limit_train_batches"], None),
    "split file sha256": (manifest["split_file_sha256"], split_sha),
}
problems = {key: pair for key, pair in problems.items() if pair[0] != pair[1]}
if problems:  # e.g. still training, or a shortened smoke run
    sys.exit(f"{arm}: {run.name} is not a finished full run: " + "; ".join(f"{k} {a} != {b}" for k, (a, b) in problems.items()))
print(checkpoint_path(run, manifest, choice))
EOF
)
  ckpt=$(printf '%s\n' "$ckpt" | tail -n 1)  # Keep only the path if anything else reached stdout.
  mkdir -p "$out"
  echo "$ckpt" > "${out}/checkpoint.txt"
  echo "checkpoint: ${ckpt}"

  uv run python scripts/inference/run_inference.py --model_type EDL --config_edl "$config" \
    --checkpoint "$ckpt" --output_dir "$pred_root" --subsets val,test --no_save_plot

  for subset in val test; do
    uv run python scripts/eval/eval_metrics.py --model_type EDL --config "$config" --subset "$subset" \
      --output_dir "$pred_root" --no_plot_png --save_tif
  done

  # Every split event must have exactly one prediction and one confusion map (cm_* files are not predictions).
  uv run python - "$config" "$pred_root" <<'EOF'
import json, sys
from pathlib import Path

config, pred_root = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
split = json.loads(Path(config["shared"]["data_params"]["train_test_split_file"]).read_text())
for subset in ("val", "test"):
    expected = {Path(p).stem for p in split[subset]["S2"]}
    files = list((pred_root / subset / "EDL").glob("*_output_EDL.tif"))
    found = {
        "predictions": {p.name.removesuffix("_output_EDL.tif") for p in files if not p.name.startswith("cm_")},
        "confusion maps": {p.name[3:].removesuffix("_output_EDL.tif") for p in files if p.name.startswith("cm_")},
    }
    for kind, events in found.items():
        if events != expected:
            sys.exit(f"{subset} {kind} do not match the split "
                     f"(missing {sorted(expected - events)}, extra {sorted(events - expected)})")
    print(f"{subset}: {len(expected)} predictions and confusion maps match the split")
EOF

  # analysis_S2 skips a failing event and continues; any such error invalidates the val threshold.
  mkdir -p "${analysis_root}"
  uv run python scripts/analysis/analysis_S2.py --model-type EDL --subset val \
    --pred-root "$pred_root" --output-dir "${analysis_root}/val" 2>&1 | tee "${analysis_root}/analysis_S2_val.log"
  if grep -q "Error processing" "${analysis_root}/analysis_S2_val.log"; then
    echo "analysis_S2 skipped an event after an error; see ${analysis_root}/analysis_S2_val.log" >&2
    exit 1
  fi

  common=(--model-type EDL --subset test --pred-root "$pred_root" --output-dir "${analysis_root}/test"
          --acc-threshold "$PAVPU_ACC_THRESHOLD" --patch-size "$PATCH_SIZE" --retention-target "$RETENTION")
  uv run python scripts/analysis/compute_pavpu.py "${common[@]}" --threshold-mode curve \
    --curve-subset val --curve-root "${analysis_root}/val"
  uv run python scripts/analysis/compute_pavpu.py "${common[@]}" --threshold-mode per-image
  uv run python scripts/analysis/compute_pavpu.py "${common[@]}" --threshold-mode per-image --threshold-pixels gt-valid
done
