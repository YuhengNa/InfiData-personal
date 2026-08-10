#!/usr/bin/env bash

set -euo pipefail

# 1. Rebuild and verify the final P0-P5 episode decisions.
/data/wudi/.venvs/infidata-quality/bin/python /data/wudi/InfiData-personal/scripts/quality/build_final_rlds_decisions.py

# 2. Create/resume the independent filtered RLDS copy.
/data/wudi/.venvs/infidata-quality/bin/python \
  /data/wudi/InfiData-personal/scripts/quality/materialize_filtered_rlds.py \
  --source-root /data/wudi/RLDS \
  --output-root /data/wudi/RLDS_State_Action_Filtered_20260807 \
  --execute
