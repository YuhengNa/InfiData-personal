#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_agibot_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/InfiData/AgiBotWorld-Beta-ModelScope-500h}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/AgiBot}"
SCHEMA_KEY="agibot_world_robot_agibot_world_beta_mobile_dual_arm_joint_absolute_position_real_s20_a20_fps30_cam_high_cam_left_wrist_cam_right_wrist"
OUT_DIR="$RLDS_ROOT/${SCHEMA_KEY}__episodes_22986"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key "$SCHEMA_KEY" \
  --data-dir "$OUT_DIR" \
  --overwrite \
  "$@"
