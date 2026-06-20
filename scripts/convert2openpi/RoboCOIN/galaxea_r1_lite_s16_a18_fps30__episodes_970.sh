#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_robocoin_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/InfiData/RoboCOIN}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/RoboCOIN}"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key galaxea_r1_lite_s16_a18_fps30 \
  --camera-set cam_high,cam_left_wrist,cam_right_wrist \
  --data-dir "$RLDS_ROOT/galaxea_r1_lite_s16_a18_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_970" \
  --overwrite \
  "$@"
