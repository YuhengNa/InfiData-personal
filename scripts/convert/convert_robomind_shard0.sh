#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOMIND_ROOT="${ROBOMIND_ROOT:-/mnt/workspace/wudi/ELUBrain/RoboMIND}"
OUT_ROOT="${OUT_ROOT:-/mnt/workspace/wudi/InfiData/RoboMIND_shard_0}"
EXTRACT_ROOT="${EXTRACT_ROOT:-/mnt/workspace/wudi/ELUBrain/RoboMIND_extracted_shards/shard_0}"

extra_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
if [[ "${NO_VIDEOS:-0}" == "1" ]]; then
  extra_args+=(--no_videos)
fi

python "${SCRIPT_DIR}/convert_robomind.py" \
  --robomind_root "${ROBOMIND_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --extract_root "${EXTRACT_ROOT}" \
  --num_shards 4 \
  --shard_index 0 \
  "${extra_args[@]}" \
  "$@"
