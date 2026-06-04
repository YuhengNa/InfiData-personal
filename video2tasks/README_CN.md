<div align="center">

# 🎬 Video2Tasks

**多任务机器人视频 → 单任务片段 + 自动指令标注 → VLA 训练数据**

[![Python 3.8+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[English](README.md) | [中文文档](README_CN.md)

</div>

---

> InfiData、RMBench、memory summary、parquet 写回和可视化的项目对接说明，请参阅 [自动标注管线使用说明](PIPELINE_GUIDE_CN.md)。

## 📖 概览

### 🎯 解决什么问题？

训练 **VLA（Vision-Language-Action）模型**（如 [π₀ (pi-zero)](https://www.physicalintelligence.company/blog/pi0)）时，你需要的是**带指令标注的单任务视频片段**。然而，真实的机器人演示视频往往包含**多个连续任务**且**没有任何标注**：

```
输入:  包含多个任务的长视频，无标注
           ┃
           ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  🎬 Video2Tasks                                             │
     │  • VLM 驱动的任务边界检测                                     │
     │  • 自动生成自然语言指令标注                                   │
     │  • 分布式处理支持大规模数据集                                 │
     └─────────────────────────────────────────────────────────────┘
           ┃
           ▼
输出: 单任务片段 + 指令标注，可直接用于 VLA 训练

  segment_001.mp4         segment_002.mp4         segment_003.mp4
  "Pick up the fork"      "Place the fork"        "Pick up the spoon"
```

**Video2Tasks = 任务切分 + 指令标注 → VLA 训练数据生产线**

### 🔧 工作原理

本工具采用**分布式 Client-Server 架构**，使用视觉语言模型（如 Qwen3-VL）分析视频帧，智能检测任务边界，并为每个片段生成自然语言指令。

| 组件 | 描述 |
|------|------|
| **Server** | 读取视频、分窗抽帧、管理任务队列并聚合结果 |
| **Worker** | 调用 VLM 推理，检测任务切换点并生成指令标注 |

---

## 📊 输出示例

### VLM 逐窗口推理过程

VLM 会分析每个重叠的帧窗口，并提供详细的任务切换推理：

<details>
<summary>🔍 点击查看多个窗口的 VLM 推理过程</summary>

**Window 0** - 检测 bag → mask 切换:
```json
{
  "task_id": "LongData601-1189::1765279974654_w0",
  "window_id": 0,
  "vlm_json": {
    "thought": "Frames 0-2: The person is standing, hands open, wearing gloves, facing the room. No object interaction yet. Frame 3: The person reaches toward a white tote bag on the sofa. Frame 4: The person grasps the tote bag and begins lifting it. Frames 5-11: The person continues manipulating the tote bag, opening it, adjusting its straps, and handling its contents. This is a continuous interaction with the same object (the tote bag). Frame 12: The person reaches into the bag and pulls out a white object with a black strap (likely a mask or headgear). The interaction from frame 12 onward shifts from the tote bag to the white object (mask). Therefore, a switch occurs at frame 12.",
    "transitions": [12],
    "instructions": ["Pick up and manipulate the tote bag", "Retrieve and adjust the white face mask"]
  }
}
```

**Window 3** - 检测多物体切换:
```json
{
  "task_id": "LongData601-1189::1765279974654_w3",
  "window_id": 3,
  "vlm_json": {
    "thought": "Frames 0-2: The robot's left hand reaches for and grasps a small black object from the left table. The right hand holds a white tote bag. Frames 3-5: The left hand places the black object into the tote bag. Frames 6-7: The left hand releases the black object into the bag and then reaches back to pick up another small black object. This is a clear switch: the robot completes interaction with the first black object and starts interacting with a second, distinct black object. Frame 15: The robot reaches for the white kettle on the left table. This marks a new interaction with a different object (the kettle). Therefore, switches are detected at frame 6 (first→second black object) and frame 15 (bag→kettle).",
    "transitions": [6, 15],
    "instructions": ["Place the first black object into the tote bag", "Place the second black object into the tote bag", "Pick up the white kettle"]
  }
}
```

**Window 8** - 未检测到切换（连续任务）:
```json
{
  "task_id": "LongData601-1189::1765279974654_w8",
  "window_id": 8,
  "vlm_json": {
    "thought": "Frames 0-2: The robot is holding a small white object (possibly a charger or adapter) in both hands and appears to be manipulating or connecting it to a cable. Frames 3-15: The robot continues to hold the same object and cable, moving it toward an electrical outlet near the TV stand. There is no clear release of the object or grasping of a new distinct object. The entire sequence involves manipulating a single object. No switch occurs.",
    "transitions": [],
    "instructions": ["Move the charger toward the power outlet"]
  }
}
```

**Window 14** - 复杂多物体序列:
```json
{
  "task_id": "LongData601-1189::1765279974654_w14",
  "window_id": 14,
  "vlm_json": {
    "thought": "Frames 0-2: The person is interacting with objects on the left table — moving a small white cup and adjusting a black rectangular object. Frame 3: The right hand moves away from the black object and reaches toward the ashtray. This marks the end of interaction with the black object. Frame 4: The right hand grasps the ashtray. This is a clear switch to a new object. Frames 5-7: The person moves the ashtray toward the trash can. Frame 11: The right hand reaches down to pick up a pair of white slippers from the floor. This is a clear switch from ashtray to slippers. Switches occur at frame 3 (black object→ashtray) and frame 11 (ashtray→slippers).",
    "transitions": [3, 11],
    "instructions": ["Move the black rectangular object and cup", "Pick up the ashtray", "Pick up the white slippers", "Place the slippers on the rack"]
  }
}
```

</details>

### 最终切分结果

一个 4501 帧的视频自动切分成 16 个单任务片段：

```json
{
  "video_id": "1765279974654",
  "nframes": 4501,
  "segments": [
    {"seg_id": 0,  "start_frame": 0,    "end_frame": 373,  "instruction": "Pick up and manipulate the tote bag"},
    {"seg_id": 1,  "start_frame": 373,  "end_frame": 542,  "instruction": "Retrieve and adjust the white face mask"},
    {"seg_id": 2,  "start_frame": 542,  "end_frame": 703,  "instruction": "Open and place items into the bag"},
    {"seg_id": 3,  "start_frame": 703,  "end_frame": 912,  "instruction": "Place the first black object into the tote bag"},
    {"seg_id": 4,  "start_frame": 912,  "end_frame": 1214, "instruction": "Place the second black object into the tote bag"},
    {"seg_id": 5,  "start_frame": 1214, "end_frame": 1375, "instruction": "Place the white cup on the table"},
    {"seg_id": 6,  "start_frame": 1375, "end_frame": 1524, "instruction": "Move the cup to the right table"},
    {"seg_id": 7,  "start_frame": 1524, "end_frame": 1784, "instruction": "Connect the power adapter to the cable"},
    {"seg_id": 8,  "start_frame": 1784, "end_frame": 2991, "instruction": "Plug the device into the power strip"},
    {"seg_id": 9,  "start_frame": 2991, "end_frame": 3135, "instruction": "Interact with black object on coffee table"},
    {"seg_id": 10, "start_frame": 3135, "end_frame": 3238, "instruction": "Adjust the ashtray"},
    {"seg_id": 11, "start_frame": 3238, "end_frame": 3359, "instruction": "Interact with the white mug"},
    {"seg_id": 12, "start_frame": 3359, "end_frame": 3478, "instruction": "Move the black rectangular object and cup"},
    {"seg_id": 13, "start_frame": 3478, "end_frame": 3711, "instruction": "Pick up the ashtray"},
    {"seg_id": 14, "start_frame": 3711, "end_frame": 4095, "instruction": "Move the white slippers from the shoe rack"},
    {"seg_id": 15, "start_frame": 4095, "end_frame": 4501, "instruction": "Raise the window blind"}
  ]
}
```

> 🎯 每个片段只包含**一个任务**，并自动生成自然语言指令 —— 直接用于 VLA 训练！

---

## 💡 为什么选择这套架构？

<table>
<tr>
<td width="50%">

### 🧠 分布式架构

这不是一个死循环脚本。FastAPI 作为调度中心，Worker 只负责推理。

**你可以在一台 4090 上跑 Server，再挂 10 台机器跑 Worker 并行处理海量数据。**

这是工业级的思路。

</td>
<td width="50%">

### 🛡️ 工程化容错

- ⏱️ Inflight 超时重发
- 🔄 失败重试上限
- 📍 `.DONE` 断点续传标记

这些机制是大规模任务稳定跑完的关键。

</td>
</tr>
<tr>
<td width="50%">

### 🎯 智能切分算法

不是简单把图片丢给模型。`build_segments_via_cuts` 对多窗口结果做**加权投票**，并引入 **Hanning Window** 处理窗口边缘权重。

解决了"窗口边缘识别不稳"的经典问题。

</td>
<td width="50%">

### ✍️ 专业 Prompt 设计

`prompt_switch_detection` 明确区分：
- **True Switch**：切换到新物体
- **False Switch**：同一物体不同操作

贴合 Manipulation 数据集的痛点，**显著降低过切**。

</td>
</tr>
</table>

---

## ✨ 特性

| 特性 | 描述 |
|------|------|
| 🎥 **视频分窗** | 可配置的视频窗口抽样参数 |
| 🧩 **可插拔后端** | 支持 Qwen3-VL / 远程 API / 自定义 VLM |
| 📊 **智能聚合** | 加权投票 + Hanning Window 自动聚合分段结果 |
| 🔄 **分布式处理** | 支持多 Worker 水平扩展 |
| ⚙️ **YAML 配置** | 简洁的声明式配置管理 |
| 🧪 **跨平台** | 推荐 Linux + GPU；Windows/CPU 可用 dummy 后端 |

---

## 🏗️ 架构图

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│     Server      │────────▶│   Job Queue     │◀────────│     Worker      │
│    (FastAPI)    │         │                 │         │     (VLM)       │
│                 │         │                 │         │                 │
└────────┬────────┘         └─────────────────┘         └────────┬────────┘
         │                                                       │
         ▼                                                       ▼
┌─────────────────┐                                     ┌─────────────────┐
│   Video Files   │                                     │    VLM Model    │
└─────────────────┘                                     └─────────────────┘
```

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/ly-geming/video2tasks.git
cd video2tasks

# 安装核心依赖
pip install -e .

# 如果使用 Qwen3-VL（需要 GPU）
pip install -e ".[qwen3vl]"
```

### 配置

```bash
# 复制示例配置
cp config.example.yaml config.yaml

# 根据需要修改配置
vim config.yaml  # 或使用你喜欢的编辑器
```

### 运行

**终端 1 - 启动服务器：**
```bash
v2t-server --config config.yaml
```

**终端 2 - 启动 Worker：**
```bash
v2t-worker --config config.yaml
```

> 💡 **提示：** 可以启动多个 Worker 来并行处理视频！

---

## ⚙️ 配置说明

查看 [`config.example.yaml`](config.example.yaml) 了解所有可用选项：

| 配置项 | 描述 |
|--------|------|
| `datasets` | 视频数据集路径和子集 |
| `run` | 输出目录配置 |
| `server` | 主机、端口和队列设置 |
| `worker` | VLM 后端选择和模型路径 |
| `windowing` | 帧采样参数 |

---

## 🔌 VLM 后端

### Dummy 后端（默认）

轻量级后端，用于测试和 Windows/CPU 环境。返回模拟结果，不加载重型模型。

```yaml
worker:
  backend: dummy
```

### Qwen3-VL 后端

使用 Qwen3-VL-32B-Instruct（或其他变体）进行完整推理。

**要求：**
- 🐧 Linux + NVIDIA GPU
- 💾 24GB+ 显存（32B 模型）
- 🔥 PyTorch + CUDA 支持

```yaml
worker:
  backend: qwen3vl
  model_path: /path/to/model
```

### 远程 API 后端

如不想本地部署模型，可配置远程 API：

```yaml
worker:
  backend: remote_api
  api_url: http://your-api-server/infer
```

<details>
<summary>📡 API 请求/响应格式</summary>

**请求体：**
```json
{
  "prompt": "...",
  "images_b64_png": ["...", "..."]
}
```

**响应格式（两种皆可）：**
```json
{
  "transitions": [6],
  "instructions": ["Place the fork", "Place the spoon"],
  "thought": "..."
}
```

或者：
```json
{
  "vlm_json": {
    "transitions": [6],
    "instructions": ["Place the fork", "Place the spoon"],
    "thought": "..."
  }
}
```

</details>

### 自定义后端

实现 `VLMBackend` 接口来添加你自己的 VLM：

```python
from video2tasks.vlm.base import VLMBackend

class MyBackend(VLMBackend):
    def infer(self, images, prompt):
        # 你的推理逻辑
        return {"transitions": [], "instructions": []}
```

---

## 📁 项目结构

```
video2tasks/
├── 📂 src/video2tasks/
│   ├── config.py              # 配置模型
│   ├── prompt.py              # Prompt 模板
│   ├── 📂 server/             # FastAPI 服务端
│   │   ├── app.py
│   │   └── windowing.py
│   ├── 📂 worker/             # Worker 实现
│   │   └── runner.py
│   ├── 📂 vlm/                # VLM 后端
│   │   ├── dummy.py
│   │   ├── qwen3vl.py
│   │   └── remote_api.py
│   └── 📂 cli/                # CLI 入口
│       ├── server.py
│       └── worker.py
├── 📄 config.example.yaml
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 README_CN.md
└── 📄 LICENSE
```

---

## 🧪 测试与验证

```bash
# 验证配置文件
v2t-validate --config config.yaml

# 运行测试
pytest
```

---

## 💻 系统要求

<table>
<tr>
<th>最低配置（Dummy 后端）</th>
<th>推荐配置（Qwen3-VL）</th>
</tr>
<tr>
<td>

- Python 3.8+
- 4GB 内存
- 任意操作系统

</td>
<td>

- Python 3.8+
- Linux + NVIDIA GPU
- 24GB+ 显存
- CUDA 11.8+ / 12.x

</td>
</tr>
</table>

---

## 🤝 贡献

欢迎贡献代码！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 基于 [FastAPI](https://fastapi.tiangolo.com/) 构建
- VLM 支持来自 [Transformers](https://huggingface.co/docs/transformers/)
- 灵感来源于机器人视频分析研究

---

<div align="center">

**⭐ 如果觉得有用，请给个 Star！⭐**

</div>
