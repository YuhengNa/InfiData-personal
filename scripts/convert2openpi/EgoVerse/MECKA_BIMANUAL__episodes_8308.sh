#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_egoverse_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/InfiData/EgoVerse}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/EgoVerse}"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key MECKA_BIMANUAL \
  --camera-set front_1 \
  --data-dir "$RLDS_ROOT/mecka_bimanual_front_1__episodes_8308" \
  --overwrite \
  "$@"
