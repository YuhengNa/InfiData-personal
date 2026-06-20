#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_robocoin_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/InfiData/RoboCOIN}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/RoboCOIN}"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key unitree_g1_s28_a28_fps30 \
  --camera-set cam_high \
  --data-dir "$RLDS_ROOT/unitree_g1_s28_a28_fps30_cam_high__episodes_227" \
  --overwrite \
  "$@"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key unitree_g1_s28_a28_fps30 \
  --camera-set cam_high,cam_left_wrist,cam_right_wrist \
  --data-dir "$RLDS_ROOT/unitree_g1_s28_a28_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_931" \
  --overwrite \
  "$@"
