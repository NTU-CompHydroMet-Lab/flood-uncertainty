#!/usr/bin/env bash

set -euo pipefail

# Case-study figure export helper
#
# Usage:
#   bash CJ_scripts/plot/run_case_compare_exports.sh
#
# What this script does:
# 1. Re-export full-image confusion compare figures
# 2. Re-export full-image uncertainty compare figures
# 3. Re-export 512x512 tiled confusion compare figures
# 4. Re-export 512x512 tiled uncertainty compare figures
#
# Default event IDs:
# - EMSR466_AOI01_DEL_PRODUCT
# - EMSR264_08VATOMANDRY_DEL_v2
#
# Runtime:
# - Uses `uv run python`
# - Run this script from the repository root

PYTHON_CMD=(uv run python)
TILE_SIZE=512

echo "== Full-image confusion compare figures =="
"${PYTHON_CMD[@]}" CJ_scripts/plot/plot_case_confusion_compare.py --event-id EMSR466_AOI01_DEL_PRODUCT
"${PYTHON_CMD[@]}" CJ_scripts/plot/plot_case_confusion_compare.py --event-id EMSR264_08VATOMANDRY_DEL_v2

echo "== Full-image uncertainty compare figures =="
"${PYTHON_CMD[@]}" CJ_scripts/plot/plot_case_uncertainty_compare.py --event-id EMSR466_AOI01_DEL_PRODUCT
"${PYTHON_CMD[@]}" CJ_scripts/plot/plot_case_uncertainty_compare.py --event-id EMSR264_08VATOMANDRY_DEL_v2

echo "== 512x512 tiled confusion compare figures =="
"${PYTHON_CMD[@]}" CJ_scripts/plot/export_case_compare_tiles.py --tile-size "${TILE_SIZE}" --kind confusion

echo "== 512x512 tiled uncertainty compare figures =="
"${PYTHON_CMD[@]}" CJ_scripts/plot/export_case_compare_tiles.py --tile-size "${TILE_SIZE}" --kind uncertainty

echo "All exports completed."
