#!/bin/bash
# Plumbing check on the `dev` partition before any real run: GPU visible, venv
# imports, TerraMind weights loadable offline, 1-epoch EDL-TerraMind train on
# one tile per split. Submit with: bash scripts/nchc/submit.sh --smoke
#SBATCH --job-name=flood-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/nchc_logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run via sbatch}"
source scripts/nchc/env.sh
nchc_runtime_env
nchc_offline_env

nvidia-smi
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import torch
from huggingface_hub import hf_hub_download
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
path = hf_hub_download("ibm-esa-geospatial/TerraMind-1.0-base", "TerraMind_v1_base.pt")
print("TerraMind weights (offline):", path)
PY

SMOKE_TRAIN_SOURCE_DATA_ROOT="$NCHC_DATA/worldfloods_v2" \
SMOKE_TRAIN_NUM_WORKERS=4 \
bash scripts/smoke/smoke_train_edl_terramind.sh
