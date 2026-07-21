#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_robomind_full_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/wudi/InfiData}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/RoboMIND_full}"
SCHEMA_KEY="franka_sim_simulation_franka_joint_position_h5_sim_franka_3rgb_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external"
OUT_DIR="$RLDS_ROOT/${SCHEMA_KEY}__episodes_14488"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key "$SCHEMA_KEY" \
  --data-dir "$OUT_DIR" \
  --overwrite \
  "$@"
