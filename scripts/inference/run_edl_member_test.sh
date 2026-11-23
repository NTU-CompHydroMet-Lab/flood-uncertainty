#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory; all paths below belong to this repository.
EDL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$EDL_PROJECT_ROOT"

if (( $# > 1 )); then
  printf '用法：bash scripts/inference/run_edl_member_test.sh [模型編號]\n' >&2
  exit 2
fi

edl_member="${1:-}"
if [[ -z "$edl_member" ]]; then
  printf '可用的模型編號：0、4、5、6、7、8、9\n'
  read -r -p '請輸入要執行的模型編號：' edl_member
fi

# Checkpoints selected by minimum validation val_bce_land_water, not test scores.
case "$edl_member" in
  0) edl_checkpoint_name="epoch=8-step=9225.ckpt" ;;
  4) edl_checkpoint_name="epoch=2-step=3075.ckpt" ;;
  5) edl_checkpoint_name="epoch=10-step=11275.ckpt" ;;
  6) edl_checkpoint_name="epoch=14-step=15375.ckpt" ;;
  7) edl_checkpoint_name="epoch=15-step=16400.ckpt" ;;
  8|9) edl_checkpoint_name="epoch=6-step=7175.ckpt" ;;
  *)
    printf '不支援的編號：%s。請使用 0、4、5、6、7、8、9。\n' "$edl_member" >&2
    exit 2
    ;;
esac

edl_model_dir="$EDL_PROJECT_ROOT/artifacts/models/edl_ensemble/edl_from_ensemble_${edl_member}"
edl_config="$edl_model_dir/config.json"
edl_checkpoint="$edl_model_dir/checkpoint/$edl_checkpoint_name"
edl_output_dir="$EDL_PROJECT_ROOT/artifacts/results/edl_ensemble_members/member_${edl_member}"
edl_metrics="$edl_output_dir/test/EDL/metrics_EDL.csv"

# Preserve completed results when selecting the same member again.
if [[ -f "$edl_metrics" ]]; then
  printf 'Member %s 已有評估結果：\n%s\n' "$edl_member" "$edl_metrics"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  printf '找不到 uv，請先確認 uv 已安裝並加入 PATH。\n' >&2
  exit 1
fi

for edl_required_file in "$edl_config" "$edl_checkpoint"; do
  if [[ ! -f "$edl_required_file" ]]; then
    printf '找不到必要檔案：%s\n' "$edl_required_file" >&2
    exit 1
  fi
done

printf 'Member：%s\nCheckpoint：%s\n輸出目錄：%s\n' \
  "$edl_member" "$edl_checkpoint" "$edl_output_dir"

printf '\n[1/2] 推論 test dataset\n'
uv run python scripts/inference/run_inference.py \
  --model_type EDL \
  --config_edl "$edl_config" \
  --checkpoint "$edl_checkpoint" \
  --subsets test \
  --output_dir "$edl_output_dir" \
  --max_tile_size 256 \
  --save_pred_tif \
  --no_save_plot

printf '\n[2/2] 計算 test 評估指標\n'
uv run python scripts/eval/eval_metrics.py \
  --model_type EDL \
  --config "$edl_config" \
  --subset test \
  --output_dir "$edl_output_dir" \
  --no_plot_png \
  --no_save_tif

if [[ ! -f "$edl_metrics" ]]; then
  printf '未產生預期的 metrics CSV：%s\n' "$edl_metrics" >&2
  exit 1
fi

printf '\nMember %s 完成。結果：\n%s\n' "$edl_member" "$edl_metrics"
printf '查看 OVERALL 列的 iou 欄位，即為整個 test dataset 的 Water IoU。\n'
