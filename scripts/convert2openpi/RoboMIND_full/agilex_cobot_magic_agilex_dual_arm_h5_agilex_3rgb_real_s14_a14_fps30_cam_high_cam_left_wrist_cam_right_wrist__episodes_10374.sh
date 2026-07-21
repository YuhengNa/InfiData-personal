#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_robomind_full_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/wudi/InfiData}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/RoboMIND_full}"
SCHEMA_KEY="agilex_cobot_magic_agilex_dual_arm_h5_agilex_3rgb_real_s14_a14_fps30_cam_high_cam_left_wrist_cam_right_wrist"
OUT_DIR="$RLDS_ROOT/${SCHEMA_KEY}__episodes_10374"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key "$SCHEMA_KEY" \
  --data-dir "$OUT_DIR" \
  --overwrite \
  "$@"
