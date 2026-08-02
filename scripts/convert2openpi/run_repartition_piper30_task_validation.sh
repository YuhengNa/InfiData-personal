#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/wudi/.venvs/infidata-quality/bin/python}"

SOURCE_DIR="${SOURCE_DIR:-/data/wudi/RLDS/realworld_piper/piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist/realworld_piper_infidata/1.0.0}"
EPISODE_MANIFEST="${EPISODE_MANIFEST:-$REPO_ROOT/quality_runs/piper30_task_audit_20260802/episodes.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wudi/RLDS/realworld_piper_task_split/piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist/realworld_piper_infidata/1.1.0}"

"$PYTHON_BIN" "$SCRIPT_DIR/repartition_realworld_piper_rlds.py" \
  --source-dir "$SOURCE_DIR" \
  --episode-manifest "$EPISODE_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --output-version 1.1.0 \
  --unseen-task "place the cola into the paper box" \
  --unseen-task "find the green block and place it into the basket" \
  --seen-ratio 0.05 \
  --seed 0 \
  "$@"
