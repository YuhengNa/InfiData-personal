#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi

python convert_infidata_egoverse_to_rlds.py \
  --infidata-roots \
    /mnt/workspace/InfiData/EgoVerse_full_1 \
    /mnt/workspace/InfiData/EgoVerse_full_2 \
    /mnt/workspace/InfiData/EgoVerse_full_3 \
  --schema-key scale \
  --camera-set front_1 \
  --data-dir /mnt/workspace/RLDS/EgoVerse_full/scale_front_1 \
  --overwrite
