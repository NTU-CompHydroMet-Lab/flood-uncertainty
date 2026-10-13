#!/usr/bin/env bash
# One-time Python environment setup on the Nano5 *login node* (needs internet).
# Builds $NCHC_REPO/.venv from uv.lock (Python 3.10, torch wheels ship their own CUDA).
#
# Usage (on Nano5, after `push.sh code`):
#   bash $NCHC_REPO/scripts/nchc/setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
nchc_runtime_env

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$NCHC_DATA" "$NCHC_HF_HOME" \
  "$NCHC_REPO/artifacts/models" "$NCHC_REPO/artifacts/nchc_logs"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

cd "$NCHC_REPO"
uv python install 3.10
uv sync --frozen --python 3.10

"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import torch, pytorch_lightning, rasterio
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("lightning", pytorch_lightning.__version__, "rasterio", rasterio.__version__)
PY

echo "workspace usage:"
du -sh "$UV_PROJECT_ENVIRONMENT" "$UV_CACHE_DIR" 2>/dev/null || true
echo "setup done. Next: sbatch smoke test -> bash scripts/nchc/submit.sh --smoke"
