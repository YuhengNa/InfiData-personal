export HF_LEROBOT_HOME=/mnt/data/lerobot
INFI_ROOT=/mnt/data/InfiData/Rmbench_infi/
ORG=wudi

for task_dir in "$INFI_ROOT"/*; do
  [ -d "$task_dir" ] || continue
  task_name=$(basename "$task_dir")

  python scripts/convert2openpi/convert_rmbench.py \
    --infidata_root "$task_dir" \
    --repo_id "$ORG/rmbench_${task_name}" \
    --output_root "$HF_LEROBOT_HOME/$ORG/rmbench_${task_name}" \
    --fps 30 \
    --use_images \
    --overwrite
done