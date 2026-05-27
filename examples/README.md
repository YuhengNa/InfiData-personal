# Examples

本目录保存可直接检查的数据样例，用来验证转换脚本、schema 字段和下游训练格式。

| 目录 | 格式 | 来源 | 规模 | 说明 |
| --- | --- | --- | --- | --- |
| `aloha_mini` | InfiData | ALOHA-Cosmos-Policy | 3 episodes | ALOHA 最小转换样例，subtask 为自动占位分段 |
| `droid_mini` | InfiData | DROID | 3 episodes | DROID 最小转换样例，包含 real_future 子目标帧指针 |
| `rmbench_mini` | InfiData | RMBench `battery_try/demo_clean` | 3 episodes | RMBench 最小转换样例，subtask 来自 `language_annotation.json` |
| `rmbench_lerobot_mini` | LeRobot/openpi | `rmbench_mini` | 3 episodes | 面向 openpi 的 LeRobot 样例，额外包含 `subtask` 和 `memory` |

## RMBench 转换链路

当前 RMBench 示例分两步生成：

1. RMBench HDF5 转 InfiData：

```bash
python scripts/convert/convert_rmbench_mini.py \
  --rmbench_root /path/to/RMBench/data \
  --task_name battery_try \
  --out_root examples/rmbench_mini \
  --num_episodes 3
```

2. InfiData 转 LeRobot/openpi：

```bash
python scripts/convert2openpi/convert_rmbench.py \
  --infidata_root examples/rmbench_mini \
  --repo_id wudi/rmbench_battery_try_demo3 \
  --output_root examples/rmbench_lerobot_mini \
  --num_episodes 3 \
  --fps 30 \
  --use_images \
  --overwrite
```

## 注意事项

`rmbench_mini` 中的 `subtask` 和 `meta/segments.jsonl` 来自 RMBench 原始 `language_annotation.json`，不是 fake data。`rmbench_lerobot_mini` 中的 `memory` 当前与 `subtask` 保持一致，只是为了先打通 openpi 数据路径；后续可以替换为更长时序的人工记忆标注。
