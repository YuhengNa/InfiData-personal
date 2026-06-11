# Video2Tasks + InfiData 自动标注管线使用说明

本文档面向需要在实验室服务器上运行、检查和交接自动标注任务的成员。

当前管线支持：

- 从机器人操作视频自动划分 `subtask`
- 为视频生成 long-term memory `summary`
- 将结果保存到独立 run 目录
- 将结果回填到 InfiData 数据集的 `meta/*.jsonl`
- 将 `subtask` 或 `summary` 按帧写回 episode parquet
- 为每个 episode 生成带时间轴的可视化视频

## 1. 管线结构

管线由两个进程组成：

```text
InfiData dataset
      |
      v
v2t-server
  - 解析 episode 和视频路径
  - 分窗、抽帧、PNG 编码
  - 创建推理任务
  - 聚合窗口结果
  - 写入 run 目录
  - 回填 JSONL / parquet
  - 生成可视化视频
      |
      | HTTP: 127.0.0.1:<port>
      v
v2t-worker
  - 获取任务
  - 解码视频帧
  - 调用 Qwen3-VL
  - 返回 subtask / memory JSON
```

如果 server 和 worker 都运行在同一台 SSH 云服务器上，推荐使用：

```yaml
server:
  host: "127.0.0.1"

worker:
  server_url: "http://127.0.0.1:8131"
```

不需要浏览器，也不需要从个人笔记本访问该 HTTP 地址。

## 2. 环境安装

进入服务器上的 `video2tasks` 目录：

```bash
cd /mnt/data/szeluresearch/InfiData/video2tasks
pip install -e ".[qwen3vl]"
```

`-e` 表示 editable install。安装后，代码仍然指向当前仓库：

```bash
python -c "import video2tasks; print(video2tasks.__file__)"
which v2t-server
which v2t-worker
```

预期 Python 包路径类似：

```text
/mnt/data/szeluresearch/InfiData/video2tasks/src/video2tasks/__init__.py
```

## 3. InfiData 输入结构

单个 subset 的典型结构：

```text
<root>/<subset>/
├── data/
│   └── .../episode_000000.parquet
├── meta/
│   ├── episodes.jsonl
│   ├── segments.jsonl
│   └── memory_segments.jsonl
└── videos/
    └── .../episode_000000_cam_high.mp4
```

### 3.1 视频

推荐在配置中显式指定视角：

```yaml
video_key: "cam_high"
```

不要优先使用 `video_key: "auto"`。当 `episodes.jsonl` 或 parquet 中没有符合预期的视频路径字段时，`auto` 可能导致：

```text
[Resume] Already done: 0/0 (computed_total=0)
[Dataset] Completed <subset>. Switching to next...
```

### 3.2 已有 subtask

已有 subtask 从以下文件读取：

```text
<root>/<subset>/meta/segments.jsonl
```

示例：

```json
{"episode_index": 0, "segment_index": 0, "start_frame": 0, "end_frame": 266, "task": "...", "subtask": "Pick up the batteries"}
```

RMBench 的 memory-only 标注会保留该文件，不会修改已有 subtask。

## 4. 标注模式

### 4.1 仅划分 subtask

```yaml
annotation:
  targets:
    - "subtask"

segmentation:
  mode: "fine"
```

`segmentation.mode`：

- `coarse`：主要在操作对象切换时划分
- `fine`：按 reach、grasp、lift、move、place、release 等动作阶段细分

### 4.2 仅标注 memory

适用于 RMBench 等已经有 subtask 的数据集：

```yaml
annotation:
  targets:
    - "memory"

memory:
  use_subtask_context: true
  align_to_subtasks: false
```

- `use_subtask_context: true`：将已有 subtask 文本作为 VLM 的辅助上下文
- `align_to_subtasks: false`：memory 按自身状态变化独立切分
- `align_to_subtasks: true`：最终 memory 边界对齐到已有 subtask 边界

推荐先使用 `align_to_subtasks: false` 检查真实 memory 状态变化；需要“一段 subtask 对应一段 memory”时再设为 `true`。

`align_to_subtasks: true` 会改变 memory 任务的分段与聚合方式。全量运行前，应先用少量 episode 检查生成数量、边界和 `.DONE` 状态是否符合预期。

### 4.3 同时划分 subtask 和 memory

适用于 ALOHA 等需要同时生成两类标注的数据集：

```yaml
annotation:
  targets:
    - "subtask"
    - "memory"

segmentation:
  mode: "fine"

memory:
  use_subtask_context: false
  align_to_subtasks: true
```

同一窗口会要求 VLM 同时返回：

```json
{
  "subtask": {
    "transitions": [],
    "instructions": []
  },
  "memory": {
    "transitions": [],
    "summaries": [],
    "change_event_types": []
  }
}
```

## 5. Write-back 行为

### 5.1 推荐配置

memory-only 标注并写回 JSONL 和 parquet：

```yaml
infidata:
  write_back: true
  update_parquet_subtasks: false
  update_parquet_memory_summaries: true
  parquet_memory_column: "summary"
```

### 5.2 开关行为表

| 配置 | 行为 |
|---|---|
| `write_back: false` | 不修改原数据集，只保存 run 目录结果 |
| `write_back: true` + target `subtask` | 更新 `meta/segments.jsonl` |
| `write_back: true` + target `memory` | 更新 `meta/memory_segments.jsonl` |
| `update_parquet_subtasks: true` | 将生成的 subtask 按帧写入 parquet 的 `subtask` 列 |
| `update_parquet_memory_summaries: true` | 将 memory summary 按帧写入 parquet |
| `parquet_memory_column: "summary"` | 指定 parquet 中的 memory 列名 |

`update_parquet_subtasks` 和 `update_parquet_memory_summaries` 是两个独立开关。

对于 memory-only 任务，保持：

```yaml
update_parquet_subtasks: false
```

这样不会修改已有 subtask。

### 5.3 JSONL 替换规则

回填 `meta/segments.jsonl` 或 `meta/memory_segments.jsonl` 时，管线会：

1. 读取原 JSONL
2. 删除当前 `episode_index` 的旧记录
3. 写入本次生成的新记录
4. 保留其他 episode 的记录

因此，当前 episode 中原有的占位 summary 会被新结果替换。

### 5.4 Parquet 写入规则

开启 `update_parquet_memory_summaries` 后：

- 如果 parquet 中没有 `summary`，会在 `subtask` 后创建该列
- 优先使用 `frame_index` 匹配 memory 的 `start_frame` / `end_frame`
- 如果没有 `frame_index`，则使用 `timestamp`
- 对应时间范围内的旧 summary 会被覆盖
- parquet 文件原地写回

正式全量运行前，建议备份：

```bash
cp -a /mnt/workspace/InfiData/Rmbench_infi \
      /mnt/workspace/InfiData/Rmbench_infi_backup_before_v2t
```

## 6. Run 目录与断点续跑

配置：

```yaml
run:
  base_dir: "/mnt/data/szeluresearch/InfiData/video2tasks/Rmbench_v2t_runs"
  run_id: "rmbench_memory_v1"
```

实际输出目录：

```text
<base_dir>/<subset>/<run_id>/
├── memory_segments.jsonl
├── segments_infidata.jsonl
├── visualizations/
│   └── episode_000000_annotated.mp4
└── samples/
    └── episode_000000/
        ├── windows.jsonl
        ├── memory_segments.jsonl
        ├── segments.json
        └── .DONE
```

其中：

- `windows.jsonl`：逐窗口 VLM 原始输出，适合排查模型是否看到视频
- `memory_segments.jsonl`：当前 run 聚合后的 memory 标注
- `.DONE`：该 episode 已完成的断点标记
- `visualizations/`：带 subtask 和 memory 时间轴的视频

更换配置或重新实验时应使用新的 `run_id`。否则管线会根据 `.DONE` 跳过已完成 episode。

## 7. 完整 RMBench Memory 配置模板

以下模板展示单个 subset。全量运行时，在 `datasets` 中继续添加其他 subset。

```yaml
datasets:
  - root: "/mnt/workspace/InfiData/Rmbench_infi"
    subset: "battery_try"
    format: "infidata"
    video_key: "cam_high"

run:
  base_dir: "/mnt/data/szeluresearch/InfiData/video2tasks/Rmbench_v2t_runs"
  run_id: "rmbench_memory_v1"

server:
  host: "127.0.0.1"
  port: 8131
  max_queue: 8
  inflight_timeout_sec: 600.0
  max_retries_per_job: 5
  auto_exit_after_all_done: false

worker:
  server_url: "http://127.0.0.1:8131"
  backend: "qwen3vl"

  qwen3vl:
    model_path: "Qwen/Qwen3-VL-8B-Instruct"
    device_map: "auto"

  remote_api:
    api_url: "http://127.0.0.1:8080/infer"
    api_key: ""
    timeout_sec: 60.0
    headers: {}

windowing:
  window_sec: 16.0
  step_sec: 8.0
  frames_per_window: 16
  target_width: 720
  target_height: 480
  png_compression: 0

annotation:
  targets:
    - "memory"

segmentation:
  mode: "fine"

infidata:
  write_back: true
  update_parquet_subtasks: false
  update_parquet_memory_summaries: true
  parquet_memory_column: "summary"

memory:
  use_subtask_context: true
  align_to_subtasks: false

visualization:
  enabled: true
  output_dir: ""
  panel_height: 180

progress:
  total_override: 0

logging:
  level: "INFO"
```

## 8. 完整 ALOHA Subtask + Memory 配置模板

```yaml
datasets:
  - root: "/mnt/data/szeluresearch/InfiData/examples"
    subset: "aloha_full"
    format: "infidata"
    video_key: "cam_high"

run:
  base_dir: "/mnt/data/szeluresearch/InfiData/video2tasks/aloha_full_runs"
  run_id: "aloha_subtask_memory_v1"

server:
  host: "127.0.0.1"
  port: 8132
  max_queue: 8
  inflight_timeout_sec: 600.0
  max_retries_per_job: 5
  auto_exit_after_all_done: false

worker:
  server_url: "http://127.0.0.1:8132"
  backend: "qwen3vl"
  qwen3vl:
    model_path: "Qwen/Qwen3-VL-8B-Instruct"
    device_map: "auto"
  remote_api:
    api_url: "http://127.0.0.1:8080/infer"
    api_key: ""
    timeout_sec: 60.0
    headers: {}

windowing:
  window_sec: 16.0
  step_sec: 8.0
  frames_per_window: 16
  target_width: 720
  target_height: 480
  png_compression: 0

annotation:
  targets:
    - "subtask"
    - "memory"

segmentation:
  mode: "fine"

infidata:
  write_back: true
  update_parquet_subtasks: true
  update_parquet_memory_summaries: true
  parquet_memory_column: "summary"

memory:
  use_subtask_context: false
  align_to_subtasks: true

visualization:
  enabled: true
  output_dir: ""
  panel_height: 180

progress:
  total_override: 0

logging:
  level: "INFO"
```

## 9. 运行步骤

### 9.1 校验配置

```bash
cd /mnt/data/szeluresearch/InfiData/video2tasks
v2t-validate --config config.rmbench.memory.yaml
```

重点检查：

```text
Annotation targets: ['memory']
InfiData write back: True
Update parquet subtasks: False
Update parquet memory summaries: True
Parquet memory column: summary
Visualization enabled: True
```

### 9.2 启动 server

终端 1：

```bash
cd /mnt/data/szeluresearch/InfiData/video2tasks
v2t-server --config config.rmbench.memory.yaml
```

正常启动时应看到非零 episode 数量：

```text
[Resume] Already done: 0/40 (computed_total=40)
```

### 9.3 启动 worker

终端 2：

```bash
cd /mnt/data/szeluresearch/InfiData/video2tasks
v2t-worker --config config.rmbench.memory.yaml
```

新版 worker 应显示：

```text
[Worker] Annotation targets: ['memory']
[Done] ... -> Subtask cuts: []; Memory cuts: [...]
```

### 9.4 并行运行第二套任务

可以在 GPU 仍有余量时启动另一组 server/worker，但必须使用不同的：

- `server.port`
- `worker.server_url`
- `run_id`
- 最好使用不同的 `base_dir`

## 10. 结果检查

### 10.1 检查段级 memory

```bash
head -n 5 \
  /mnt/workspace/InfiData/Rmbench_infi/battery_try/meta/memory_segments.jsonl
```

示例：

```json
{"episode_index": 0, "segment_index": 0, "start_frame": 0, "end_frame": 266, "task": "...", "summary": "The robot is holding the left battery. The positive slot remains empty.", "change_event_type": ["object_picked"], "annotation_status": "vlm_pseudo"}
```

### 10.2 检查 parquet summary

```bash
python - <<'PY'
import pandas as pd

p = "/mnt/workspace/InfiData/Rmbench_infi/battery_try/data/rmbench/chunk-000/episode_000000.parquet"
df = pd.read_parquet(p)
print(df.columns.tolist())
print(df[["frame_index", "timestamp", "subtask", "summary"]].head(20).to_string())
PY
```

### 10.3 查找可视化视频

```bash
find /mnt/data/szeluresearch/InfiData/video2tasks/Rmbench_v2t_runs \
  -name "*annotated.mp4" | head
```

可视化底部包含：

- Subtask 时间轴
- Memory 时间轴
- 当前 subtask
- 当前 memory summary

## 11. 常见问题

### 11.1 `computed_total=0`

现象：

```text
[Resume] Already done: 0/0 (computed_total=0)
```

处理：

1. 将 `video_key: "auto"` 改为 `video_key: "cam_high"`
2. 检查 `meta/episodes.jsonl`、`meta/segments.jsonl` 和视频目录
3. 检查 subset 名称和 root 路径

### 11.2 更换 run_id 后仍然跳过

检查实际目录：

```bash
find <base_dir>/<subset>/<run_id> -name ".DONE" | head
```

同时确认启动命令使用的是修改后的配置文件。

### 11.3 VLM 输出 `No visual information is available`

先检查 worker 的图片解码函数：

```bash
python - <<'PY'
import inspect
import video2tasks.worker.runner as runner
print(inspect.getsource(runner.decode_b64_to_numpy))
PY
```

函数中必须存在：

```python
base64.b64decode(...)
Image.open(...)
```

再检查 OpenCV 是否能读取真实帧：

```bash
python - <<'PY'
import cv2

p = "/mnt/workspace/InfiData/Rmbench_infi/battery_try/videos/rmbench/episode_000000_cam_high.mp4"
cap = cv2.VideoCapture(p)
ok, frame = cap.read()
print("opened:", cap.isOpened())
print("read:", ok)
print("shape:", None if frame is None else frame.shape)
print("mean:", None if frame is None else float(frame.mean()))
cap.release()
PY
```

### 11.4 Worker 只显示 `Cuts: [...]`

这通常表示运行的是旧版 worker。检查：

```bash
python - <<'PY'
import inspect
from video2tasks.prompt import prompt_for_annotations
import video2tasks.worker.runner as runner

print("runner:", runner.__file__)
print("new log:", "Subtask cuts" in inspect.getsource(runner.run_worker))
print("memory prompt:", "long-term memory" in prompt_for_annotations(16, "fine", ["memory"]))
PY
```

### 11.5 DROID 视频无法解码

部分 DROID 视频使用 AV1 编码，而服务器 OpenCV/FFmpeg 可能缺少可用解码器。先用：

```bash
ffprobe -hide_banner -i <video.mp4>
```

如果显示 `Video: av1` 且 OpenCV 无法稳定抽帧，需要先转码为 H.264，或安装支持 AV1 的 FFmpeg/OpenCV 环境。

### 11.6 修改代码后是否需要重新安装

如果：

```bash
python -c "import video2tasks; print(video2tasks.__file__)"
```

已经指向当前仓库的 `video2tasks/src/video2tasks`，说明是 editable install。普通 Python 文件修改后通常无需重新安装，但必须重启正在运行的 server/worker 进程。

如果更新了依赖、命令入口或环境不确定，重新执行：

```bash
pip install -e ".[qwen3vl]"
```

## 12. 交接检查清单

开始全量任务前确认：

- 使用的是最新仓库代码
- `decode_b64_to_numpy` 能真实解码图片
- `video_key` 能找到正确视角
- `run_id` 为全新名称
- server 和 worker 的端口一致
- memory-only 时 `update_parquet_subtasks: false`
- 写回原数据前已备份数据集
- 先对 1 至 3 个 episode 检查 `windows.jsonl` 和可视化
- 检查 summary 没有未来信息泄漏、空文本或明显幻觉
- 确认无误后再运行全部 subset
