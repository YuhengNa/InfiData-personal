# RMBench LeRobot Mini

这是由 `examples/rmbench_mini` 转换得到的 LeRobot/openpi 示例数据，用于验证 InfiData 到 openpi 训练格式的对齐。

## Contents

- 格式：LeRobot dataset
- episode 数量：3
- 总帧数：1995
- 来源：`examples/rmbench_mini`
- `repo_id`：`wudi/rmbench_battery_try_demo3`
- robot_type：`robotwin_dual_arm_sim`
- FPS：30
- 图像格式：LeRobot `image` feature，写入 parquet
- 图像尺寸：`3 x 240 x 320`

## Features

| 字段 | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `observation.state` | `float32` | `[14]` | 机器人状态 |
| `action` | `float32` | `[14]` | 动作 |
| `observation.images.cam_high` | `image` | `[3, 240, 320]` | 头部相机图像 |
| `observation.images.cam_left_wrist` | `image` | `[3, 240, 320]` | 左腕相机图像 |
| `observation.images.cam_right_wrist` | `image` | `[3, 240, 320]` | 右腕相机图像 |
| `task` | metadata task | - | LeRobot episode task |
| `subtask` | `string` | `[1]` | InfiData 片段级语义目标 |
| `memory` | `string` | `[1]` | 当前与 `subtask` 相同，后续可替换为人工长期记忆标注 |

## Reproduce

建议在安装了 LeRobot/openpi 依赖的环境中运行：

```bash
export HF_LEROBOT_HOME=/home/wudi/wudi_data/lerobot
python scripts/convert2openpi/convert_rmbench.py \
  --infidata_root examples/rmbench_mini \
  --repo_id wudi/rmbench_battery_try_demo3 \
  --output_root examples/rmbench_lerobot_mini \
  --num_episodes 3 \
  --fps 30 \
  --use_images \
  --overwrite
```

## Quick Check

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "wudi/rmbench_battery_try_demo3",
    root="examples/rmbench_lerobot_mini",
)
sample = ds[0]
print(len(ds))
print(sample["observation.state"].shape)
print(sample["observation.images.cam_high"].shape)
print(sample["subtask"], sample["memory"])
```

该样例是 image-based LeRobot 数据集，所以 `meta/info.json` 中 `total_videos` 为 0；图像数据在 `data/chunk-000/episode_*.parquet` 中。
