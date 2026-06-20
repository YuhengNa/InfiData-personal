#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_robocoin_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/InfiData/RoboCOIN}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/RoboCOIN}"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key discover_robotics_aitbot_mmk2_s36_a36_fps30 \
  --camera-set cam_high,cam_left_wrist,cam_right_wrist \
  --data-dir "$RLDS_ROOT/discover_robotics_aitbot_mmk2_s36_a36_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_5747" \
  --overwrite \
  "$@"
