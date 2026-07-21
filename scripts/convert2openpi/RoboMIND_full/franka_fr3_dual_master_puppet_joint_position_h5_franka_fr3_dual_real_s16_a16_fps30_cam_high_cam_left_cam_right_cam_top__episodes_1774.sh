#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_robomind_full_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/wudi/InfiData}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/RoboMIND_full}"
SCHEMA_KEY="franka_fr3_dual_master_puppet_joint_position_h5_franka_fr3_dual_real_s16_a16_fps30_cam_high_cam_left_cam_right_cam_top"
OUT_DIR="$RLDS_ROOT/${SCHEMA_KEY}__episodes_1774"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key "$SCHEMA_KEY" \
  --data-dir "$OUT_DIR" \
  --overwrite \
  "$@"
