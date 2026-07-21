#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../convert_infidata_robomind_full_to_rlds.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFIDATA_ROOT="${INFIDATA_ROOT:-/mnt/workspace/wudi/InfiData}"
RLDS_ROOT="${RLDS_ROOT:-/mnt/workspace/RLDS/RoboMIND_full}"
SCHEMA_KEY="tienkung_humanoid_tiangong_joint_position_none_real_s38_a38_fps30_cam_chest_cam_head"
OUT_DIR="$RLDS_ROOT/${SCHEMA_KEY}__episodes_146"

"$PYTHON_BIN" "$CONVERTER" \
  --infidata-root "$INFIDATA_ROOT" \
  --schema-key "$SCHEMA_KEY" \
  --data-dir "$OUT_DIR" \
  --overwrite \
  "$@"
