#!/usr/bin/env bash
# 重新產生 docs/architecture.md 內的 Mermaid 圖。
#
# 圖由 pyreverse 從程式碼靜態分析產生，手寫說明保留在標記區塊之外。
# 需求：pylint（提供 pyreverse）
#   uv tool install pylint
#
# 用法：bash scripts/dev/gen_diagrams.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOC="${PROJECT_ROOT}/docs/architecture.md"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

if ! command -v pyreverse >/dev/null 2>&1; then
    echo "找不到 pyreverse，請先執行：uv tool install pylint" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

pyreverse -o mmd -p classes -d "${TMP}" flood_uncertainty/models/edl.py >/dev/null
pyreverse -o mmd -p pkg     -d "${TMP}" flood_uncertainty              >/dev/null

# 把產生的 .mmd 內容塞回 DOC 的標記區塊之間
inject() {
    local marker="$1" src="$2"
    python3 - "${DOC}" "${marker}" "${src}" <<'PY'
import re, sys
doc, marker, src = sys.argv[1], sys.argv[2], sys.argv[3]
body = open(src, encoding="utf-8").read().strip()
text = open(doc, encoding="utf-8").read()
begin, end = f"<!-- BEGIN {marker} -->", f"<!-- END {marker} -->"
block = f"{begin}\n\n```mermaid\n{body}\n```\n\n{end}"
pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
if not pattern.search(text):
    raise SystemExit(f"在 {doc} 找不到標記 {marker}")
open(doc, "w", encoding="utf-8").write(pattern.sub(lambda _: block, text))
PY
}

inject CLASS_DIAGRAM   "${TMP}/classes_classes.mmd"
inject PACKAGE_DIAGRAM "${TMP}/packages_pkg.mmd"

echo "已更新 ${DOC}"
