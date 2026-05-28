<div align="center">

# 🎬 Video2Tasks

**Split Multi-Task Robot Videos into Single-Task Segments with Auto-Generated Instructions for VLA Training**

[![Python 3.8+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[English](README.md) | [中文文档](README_CN.md)

</div>

---

## 📖 Overview

### 🎯 What Problem Does This Solve?

When training **VLA (Vision-Language-Action) models** like [π₀ (pi-zero)](https://www.physicalintelligence.company/blog/pi0), you need **single-task video segments with instruction labels**. However, real-world robot demonstration videos often contain **multiple consecutive tasks** without any annotation:

```
Input:  Long video with multiple tasks, NO labels
           ┃
           ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  🎬 Video2Tasks                                             │
     │  • VLM-powered task boundary detection                      │
     │  • Auto-generate natural language instructions              │
     │  • Distributed processing for large-scale datasets          │
     └─────────────────────────────────────────────────────────────┘
           ┃
           ▼
Output: Single-task segments + instruction labels, READY for VLA training

  segment_001.mp4         segment_002.mp4         segment_003.mp4
  "Pick up the fork"      "Place the fork"        "Pick up the spoon"
```

**Video2Tasks = Task Segmentation + Instruction Labeling → VLA Training Data Pipeline**

### 🔧 How It Works

This tool uses a **distributed client-server architecture** with VLMs (like Qwen3-VL) to analyze video frames, intelligently detect task boundaries, and generate natural language instructions for each segment.

| Component | Description |
|-----------|-------------|
| **Server** | Manages job queues, video frame extraction, and result aggregation |
| **Worker** | Runs VLM inference to detect task transitions and generate instructions |

---

## 📊 Output Example

### VLM Window-by-Window Reasoning

The VLM analyzes each overlapping frame window and provides detailed reasoning about task transitions:

<details>
<summary>🔍 Click to see VLM reasoning for multiple windows</summary>

**Window 0** - Detecting bag → mask transition:
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

**Window 3** - Detecting multiple object switches:
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

**Window 8** - No switch detected (continuous task):
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

**Window 14** - Complex multi-object sequence:
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

### Final Segmentation Output

A 4501-frame video automatically split into 16 single-task segments:

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

> 🎯 Each segment contains exactly ONE task with auto-generated natural language instruction — ready for VLA training!

---

## 💡 Why This Architecture?

<table>
<tr>
<td width="50%">

### 🧠 Distributed Architecture

Not just a single script. FastAPI acts as the orchestrator, Workers handle inference only.

**Run Server on one 4090, then connect 10 machines running Workers to process massive datasets in parallel.**

This is production-grade thinking.

</td>
<td width="50%">

### 🛡️ Production-Ready Resilience

- ⏱️ Inflight timeout & re-dispatch
- 🔄 Configurable retry limits
- 📍 `.DONE` checkpoint markers for resume

Critical mechanisms for running large-scale tasks to completion.

</td>
</tr>
<tr>
<td width="50%">

### 🎯 Smart Segmentation Algorithm

Not just throwing images at a model. `build_segments_via_cuts` performs **weighted voting** across overlapping windows with **Hanning Window** edge weighting.

Solves the classic "unstable edge detection" problem.

</td>
<td width="50%">

### ✍️ Domain-Specific Prompts

`prompt_switch_detection` explicitly distinguishes:
- **True Switch**: Transition to a new object
- **False Switch**: Different operation on the same object

Tailored for manipulation datasets, **significantly reducing over-segmentation**.

</td>
</tr>
</table>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Video Windowing** | Configurable video window sampling parameters |
| 🤖 **Pluggable Backends** | Support for Qwen3-VL, Remote API, or custom VLM implementations |
| 📊 **Smart Aggregation** | Automatic segment generation with weighted voting & Hanning window |
| 🔄 **Distributed Processing** | Scale horizontally with multiple workers |
| ⚙️ **YAML Config** | Simple, declarative configuration management |
| 🖥️ **Cross-Platform** | Linux/GPU recommended; Windows/CPU with dummy backend |

---

## 🏗️ Architecture

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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ly-geming/video2tasks.git
cd video2tasks

# Install with core dependencies
pip install -e .

# Or install with Qwen3-VL support (requires GPU)
pip install -e ".[qwen3vl]"
```

### Configuration

```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit with your paths and settings
vim config.yaml  # or your preferred editor
```

### Running

**Terminal 1 - Start the Server:**
```bash
v2t-server --config config.yaml
```

**Terminal 2 - Start a Worker:**
```bash
v2t-worker --config config.yaml
```

> 💡 **Tip:** You can start multiple workers to process videos in parallel!

---

## ⚙️ Configuration

See [`config.example.yaml`](config.example.yaml) for all available options:

| Section | Description |
|---------|-------------|
| `datasets` | Video dataset paths and subsets |
| `run` | Output directory configuration |
| `server` | Host, port, and queue settings |
| `worker` | VLM backend selection and model paths |
| `windowing` | Frame sampling parameters |

---

## 🔌 VLM Backends

### Dummy Backend (Default)

Lightweight backend for testing and Windows/CPU environments. Returns mock results without loading heavy models.

```yaml
worker:
  backend: dummy
```

### Qwen3-VL Backend

Full inference using Qwen3-VL-32B-Instruct (or other variants).

**Requirements:**
- 🐧 Linux with NVIDIA GPU
- 💾 24GB+ VRAM (for 32B model)
- 🔥 PyTorch with CUDA support

```yaml
worker:
  backend: qwen3vl
  model_path: /path/to/model
```

### Remote API Backend

Use an external API endpoint for inference:

```yaml
worker:
  backend: remote_api
  api_url: http://your-api-server/infer
```

<details>
<summary>📡 API Request/Response Format</summary>

**Request:**
```json
{
  "prompt": "...",
  "images_b64_png": ["...", "..."]
}
```

**Response:**
```json
{
  "transitions": [6],
  "instructions": ["Place the fork", "Place the spoon"],
  "thought": "..."
}
```

</details>

### Custom Backend

Implement the `VLMBackend` interface to add your own:

```python
from video2tasks.vlm.base import VLMBackend

class MyBackend(VLMBackend):
    def infer(self, images, prompt):
        # Your inference logic
        return {"transitions": [], "instructions": []}
```

---

## 📁 Project Structure

```
video2tasks/
├── 📂 src/video2tasks/
│   ├── config.py              # Configuration models
│   ├── prompt.py              # Prompt templates
│   ├── 📂 server/             # FastAPI server
│   │   ├── app.py
│   │   └── windowing.py
│   ├── 📂 worker/             # Worker implementation
│   │   └── runner.py
│   ├── 📂 vlm/                # VLM backends
│   │   ├── dummy.py
│   │   ├── qwen3vl.py
│   │   └── remote_api.py
│   └── 📂 cli/                # CLI entrypoints
│       ├── server.py
│       └── worker.py
├── 📄 config.example.yaml
├── 📄 pyproject.toml
├── 📄 README.md
├── 📄 README_CN.md
└── 📄 LICENSE
```

---

## 🧪 Testing

```bash
# Validate configuration
v2t-validate --config config.yaml

# Run tests
pytest
```

---

## 💻 Requirements

<table>
<tr>
<th>Minimum (Dummy Backend)</th>
<th>Recommended (Qwen3-VL)</th>
</tr>
<tr>
<td>

- Python 3.8+
- 4GB RAM
- Any OS

</td>
<td>

- Python 3.8+
- Linux + NVIDIA GPU
- 24GB+ VRAM
- CUDA 11.8+ / 12.x

</td>
</tr>
</table>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- VLM support via [Transformers](https://huggingface.co/docs/transformers/)
- Inspired by robotic video analysis research

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

WARNING!!
thanks for the great using tips from YuanJingYi (Sun Yat-sen University）
PLEASE name your video like this:
<img width="386" height="143" alt="348e206ad4948edee65c82d8c12ae671" src="https://github.com/user-attachments/assets/272bad75-872d-4321-9e24-e59f211ae880" />
and put each video in each folder like this
<img width="355" height="139" alt="1be04121f3312610400b559daa5bd7b3" src="https://github.com/user-attachments/assets/c65e841f-893e-411d-8e33-3a52cef95a1b" />



</div>
