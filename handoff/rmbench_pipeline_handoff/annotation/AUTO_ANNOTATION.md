# Memory 自标注


## 如何运行


### 1.环境配置

直接人为配置环境：
```bash
cd annotation/video2tasks
python -m pip install -e ".[qwen3vl]"
```

或者，也可以直接使用现存的镜像：
aliyun上海地区的annotation-v2镜像
标注环境直接安装在基环境中
镜像地址：yuanli-ai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/ai-infra/wudi_rlinf:annotation-v2


### 2.参数配置

编辑 `config.observe_and_pickup.yaml` 中的数据路径、run 路径和模型路径，然后启动：

主要配置如下：

```yaml
datasets:
  - root: "/path/to/InfiData"       # 数据集根目录
    subset: "observe_and_pickup"     # 实际读取 root/subset
    format: "infidata"
    video_key: "cam_high"            # 用于标注的视频视角

run:
  base_dir: "/path/to/runs"          # 独立运行结果目录
  run_id: "memory_v1"                # 更换实验时使用新的 run_id

annotation:
  targets:
    - "memory"                       # 可选 memory、subtask，或同时配置两者

worker:
  backend: "qwen3vl"
  qwen3vl:
    model_path: "Qwen/Qwen3-VL-8B-Instruct"
    device_map: "auto"

windowing:
  window_sec: 16.0                   # 每个视频窗口长度
  step_sec: 8.0                      # 窗口滑动步长
  frames_per_window: 16              # 每个窗口均匀抽取的关键帧数

infidata:
  write_back: true                   # 是否写回输入数据集
  update_parquet_subtasks: false     # 是否逐帧写回 subtask
  update_parquet_memory_summaries: true
  parquet_memory_column: "summary"   # memory 写入的 parquet 列名

memory:
  use_subtask_context: true          # 将已有 subtask 提供给 memory 模型
  align_to_subtasks: false           # memory 是否强制对齐 subtask 边界

visualization:
  enabled: true                      # 是否生成标注可视化视频
  output_dir: ""                     # 空值表示 runs/<subset>/<run_id>/visualizations
```

标注目标：

```yaml
targets: ["memory"]             # 仅标 memory
targets: ["subtask"]            # 仅标 subtask
targets: ["subtask", "memory"]  # 同时标注
```


### 3.实际运行

开两个终端，分别运行：

```bash
v2t-server --config config.observe_and_pickup.yaml
v2t-worker --config config.observe_and_pickup.yaml
```

server 和 worker 必须使用同一配置。修改 Python 代码后需要重启两个进程。



## 核心算法


### 1. 总体输入输出

输入：

- `meta/episodes.jsonl`：帧数、FPS、视频路径、task。
- `meta/segments.jsonl`：已有 subtask，作为模型上下文。
- `cam_high` mp4：视觉输入。

输出：

- `runs/<subset>/<run_id>/samples/<episode>/windows.jsonl`：窗口级 VLM 原始 JSON。
- `runs/.../memory_segments.jsonl`：聚合结果。
- `meta/memory_segments.jsonl`：写回数据集。
- episode parquet `summary`：逐帧 memory。
- `visualizations/*.mp4`：标注可视化。

### 2. 分窗和关键帧

默认配置：

```text
window_sec=16
step_sec=8
frames_per_window=16
```

每个窗口使用 `numpy.linspace(start_frame, end_frame, 16)` 均匀选关键帧。关键帧缩放为 720x480 PNG 后发送给 worker。

episode 0 共 148 帧，小于 16 秒，因此只有一个窗口，关键帧为：

```text
[0, 9, 19, 29, 39, 49, 58, 68, 78, 88, 98, 107, 117, 127, 137, 147]
```

### 3. 模型输入

Qwen3-VL 接收：

- 16 张有序关键帧。
- task 文本。
- 与窗口相交的已有 subtask。
- memory 提示词。

模型返回：

```json
{
  "thought": "...",
  "transitions": [6],
  "summaries": ["memory before frame 6", "memory from frame 6"],
  "change_event_types": [["initial_observation"], ["object_picked"]]
}
```

`transitions` 是关键帧序号，不是原视频帧号。episode 0 的 transition `6` 映射到原视频帧 `58`。

### 4. 校验和重试

窗口结果必须满足：

- transitions 为升序、唯一整数，范围为 `1..15`。
- summaries 数量等于 `len(transitions)+1`。
- change_event_types 数量与 summaries 相同。
- summary 和 event tag 非空。

worker 单次取到任务后最多生成 2 次；仍非法时返回空结果，server 最多重新入队 5 次。非法结果不会写入窗口结果。

聚合后再次检查：

- 第一段从帧 0 开始。
- 相邻段无空缺、无重叠。
- 最后一段结束于 `num_frames-1`。
- 每段 summary 非空。

覆盖不完整时删除窗口结果并重新标注，不生成 `.DONE`。

### 5. 窗口聚合

- transition 先映射为原视频帧。
- 重叠窗口的邻近 transition 在 `2.5 * fps` 范围内聚类。
- 使用 Hann 中心权重计算最终切点，降低窗口边缘预测的权重。
- 每段 summary 从对应窗口候选中按出现次数选择。

episode 0 最终得到：

```text
0-57    第一条 memory
58-147  第二条 memory
```

真实窗口输入输出见 `annotation/example_output/`。



## 关键代码

| 文件 | 作用 |
|---|---|
| `src/video2tasks/server/windowing.py` | 分窗、关键帧、切点映射和聚合 |
| `src/video2tasks/prompt.py` | memory 提示词和 JSON 约束 |
| `src/video2tasks/worker/runner.py` | 模型调用、本地校验和重试 |
| `src/video2tasks/server/app.py` | 任务队列、最终覆盖检查和写回 |
| `src/video2tasks/validation.py` | JSON 结构与逐帧覆盖校验 |
| `src/video2tasks/vlm/qwen3vl.py` | Qwen3-VL 推理后端 |
