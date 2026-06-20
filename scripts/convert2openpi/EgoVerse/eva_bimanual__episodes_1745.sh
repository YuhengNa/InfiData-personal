#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_egoverse_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/InfiData/EgoVerse}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/EgoVerse}"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key eva_bimanual \
  --camera-set front_1,left_wrist,right_wrist \
  --data-dir "$RLDS_ROOT/eva_bimanual_front_1_left_wrist_right_wrist__episodes_1745" \
  --overwrite \
  "$@"
