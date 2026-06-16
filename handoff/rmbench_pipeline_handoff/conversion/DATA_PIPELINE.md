# 数据转换流程

## 1. RMBench -> InfiData

输入：

```text
<rmbench_root>/observe_and_pickup/demo_clean/
├── data/episode0.hdf5
├── instructions/episode0.json
└── language_annotation.json
```

转换规则：

| RMBench | InfiData |
|---|---|
| `/joint_action/vector[t]` | `observation.state[t]` |
| `/joint_action/vector[t+1]` | `action[t]` |
| 三路相机 `/observation/*_camera/rgb` | 三个 mp4 |
| `instructions/episode*.json` | episode `task` |
| `language_annotation.json` | 逐帧 `subtask` 和 `segments.jsonl` |

由于 action 使用下一时刻，149 个 joint vector 转为 148 个 timestep。

运行：

```bash
python conversion/code/rmbench_to_infidata/convert_rmbench_mini.py \
  --rmbench_root /path/to/RMBench/data/data \
  --task_name observe_and_pickup \
  --out_root /path/to/InfiData/Rmbench_infi/observe_and_pickup \
  --num_episodes 999999 \
  --strict
```

旧数据若存在红蓝通道互换，可额外运行：

```bash
python conversion/code/rmbench_to_infidata/fix_rmbench_infidata_video_colors.py \
  --infidata_root /path/to/InfiData/Rmbench_infi/observe_and_pickup \
  --output_root /path/to/InfiData/Rmbench_infi_fixrgb/observe_and_pickup \
  --overwrite
```

当前转换脚本已经包含 RMBench RGB 到 OpenCV BGR 的处理，新转换数据不要重复修复。

## 2. InfiData memory 自标注

在生成 LeRobot 前运行标注，将结果写回：

```text
meta/memory_segments.jsonl
episode parquet 的 summary 列
```

运行方式见 [AUTO_ANNOTATION.md](AUTO_ANNOTATION.md)。

## 3. InfiData -> LeRobot v2.1

输入 memory 时，转换器读取 `meta/memory_segments.jsonl`，按 frame 区间将 summary 写入每一行的 `memory`。

```bash
python conversion/code/infidata_to_lerobot/convert_rmbench.py \
  --infidata_root /path/to/InfiData/Rmbench_infi_fixrgb/observe_and_pickup \
  --repo_id wudi/observe_and_pickup \
  --output_root /path/to/lerobot/wudi/observe_and_pickup \
  --fps 30 \
  --use_images \
  --strict \
  --max_memory_tail_extension_frames 0 \
  --overwrite
```

关键选项：

- `--use_images`：三路图像以内嵌 PNG 存入 parquet。
- `--strict`：遇到非法 episode 立即失败。
- `--max_memory_tail_extension_frames 0`：不自动补尾帧，要求 memory 原始覆盖完整。

最终每帧包含：

```text
observation.state
action
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
subtask
memory
timestamp / frame_index / episode_index / index / task_index
```

