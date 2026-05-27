# RMBench Mini

这是从 RMBench `battery_try/demo_clean` 转换得到的 InfiData 示例数据。

## Contents

- 格式：InfiData robot episode
- episode 数量：3
- 总帧数：1995
- 来源数据集：RMBench
- RMBench task：`battery_try`
- 原始配置：`demo_clean`
- 机器人：RoboTwin dual-arm simulation
- 控制模式：joint
- FPS：30
- 相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`

## Directory Layout

```text
rmbench_mini/
├── data/rmbench/chunk-000/episode_*.parquet
├── meta/
│   ├── episodes.jsonl
│   ├── segments.jsonl
│   ├── tasks.json
│   ├── robots.json
│   └── stats.json
└── videos/rmbench/*_cam_*.mp4
```

## Field Notes

每个 parquet 行对应一个 timestep，主要字段包括：

- `observation.state`：来自 RMBench `/joint_action/vector` 的当前关节状态。
- `action`：下一帧关节向量，即 one-step shifted joint target。
- `task`：来自 RMBench `instructions/episode*.json`。
- `subtask`：来自 RMBench `language_annotation.json`。
- `video.cam_*.path` 与 `video.cam_*.frame_index`：指向导出的三路相机 mp4。
- `subgoal.real_future.cam_*.*`：未来帧指针，不是人工语义目标文本。

## Data Quality

本样例没有为 task、subtask、state、action 或相机数据填充 fake data。缺失必要 RMBench 文件、相机帧不足或语言分段无法覆盖 episode 的样本不会被静默补齐；默认会跳过异常 episode，使用 `--strict` 时会直接报错。

`meta/segments.jsonl` 的 `annotation_status` 为 `human_labeled`，并额外记录 `source_annotation_status=rmbench_language_annotation`，表示语义分段来自 RMBench 原始语言标注。

## Reproduce

```bash
python scripts/convert/convert_rmbench_mini.py \
  --rmbench_root /home/wudi/src/RMBench/data \
  --task_name battery_try \
  --out_root examples/rmbench_mini \
  --num_episodes 3
```
