#!/bin/bash
# Paired Sen1Floods11 test evaluation (BEF / direct fine-tune / multimodal / UNet DEUF),
# NCHC version of the eval step in artifacts/runs/sen1_finetune_chain.sh.
# Submitted by scripts/nchc/sen1_finetune.sh after the 5 fine-tunes finish.
#
# Env: OUT (default artifacts/results/sen1_fusion_compare), SPLIT (default test)
#SBATCH --job-name=sen1-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/nchc_logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run via sbatch}"
source scripts/nchc/env.sh
nchc_runtime_env
nchc_offline_env

M=artifacts/models/terramind_sen1
C=configurations/terramind_sen1
U=artifacts/models/ref_unet_deuf          # uploaded by `push.sh sen1`
OUT="${OUT:-artifacts/results/sen1_fusion_compare}"
SPLIT="${SPLIT:-test}"

for run in tm_s1grd_kuro_s1f11 tm_l1c13_wf_s1f11 tm_s1grd_s1f11 tm_l1c13_s1f11 tm_l1c13_s1grd_s1f11; do
  [[ -f "$M/$run/.nchc_done" ]] || echo "WARNING: $M/$run has no .nchc_done (training unfinished?)"
done

# TiM optical runs pair with the same SAR models as their non-TiM counterparts.
tim=()
if [[ -f "$M/tm_l1c13_tim_wf_s1f11/.nchc_done" ]]; then
  tim+=(--pair bef_tim "$C/tm_s1grd_kuro_s1f11.json" "$M/tm_s1grd_kuro_s1f11" "$C/tm_l1c13_tim_wf_s1f11.json" "$M/tm_l1c13_tim_wf_s1f11")
fi
if [[ -f "$M/tm_l1c13_tim_s1f11/.nchc_done" ]]; then
  tim+=(--pair direct_tim "$C/tm_s1grd_s1f11.json" "$M/tm_s1grd_s1f11" "$C/tm_l1c13_tim_s1f11.json" "$M/tm_l1c13_tim_s1f11")
fi
echo "TiM pairs: $(( ${#tim[@]} / 6 ))"

"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/analysis/eval_sen1_fusion_compare.py \
  --split "$SPLIT" \
  --pair bef "$C/tm_s1grd_kuro_s1f11.json" "$M/tm_s1grd_kuro_s1f11" "$C/tm_l1c13_wf_s1f11.json" "$M/tm_l1c13_wf_s1f11" \
  --pair direct "$C/tm_s1grd_s1f11.json" "$M/tm_s1grd_s1f11" "$C/tm_l1c13_s1f11.json" "$M/tm_l1c13_s1f11" \
  --single multimodal "$C/tm_l1c13_s1grd_s1f11.json" "$M/tm_l1c13_s1grd_s1f11" \
  --pair unet_deuf configurations/edl_sar_sen1.json "$U/edl_sar_sen1floods11/checkpoint/epoch=17-step=4536.ckpt" \
         configurations/edl_optical_sen1.json "$U/edl_optical_sen1floods11/checkpoint/epoch=9-step=2520-v1.ckpt" \
  "${tim[@]}" \
  --out "$OUT"

echo "results in $OUT (pull with: bash scripts/nchc/pull.sh --results $(basename "$OUT"))"
