**InfiData —— Multi-Context Datasets for Robot Learning 社区共建仓库**

---

## 项目概述

InfiData 是一个面向社区共建的 **π₀.₇ 风格** 机器人学习数据集项目。

与传统模仿学习数据集（仅包含图像 + 本体感知 + 动作）不同，本项目重点标准化**丰富的上下文标注**，包括：

- 任务指令（task）与子任务指令（subtask）
- 子目标图像（subgoal images）
- Episode 元数据（quality 1-5、speed_bin、mistake、success 等）
- 控制模式（control_mode）、机器人类型（robot_type）等

目标是构建可用于训练 VLA 策略、世界模型、高层语言规划器以及跨具身模型的高质量、多源数据集。

---

## 核心理念

一个训练样本应包含：

- 多视角观测历史 + 本体感知（qpos）
- 动作序列（或 50 步 action chunk）
- **task** + **subtask** 指令
- **subgoal images**（真实未来帧或世界模型生成）
- Episode 元数据（quality、speed_bin、mistake）
- robot_type、control_mode、domain（real/sim）等

关键设计：
- 子目标图像提供语言难以表达的空间精度
- 元数据允许模型从次优/失败数据中学习
- 子任务指令支持逐步指导与高层策略训练
- 训练时可随机丢弃部分上下文，实现灵活推理

---
## 仓库结构
InfiData/
├── schemas/                    # JSON Schema 数据格式规范
├── configs/                    # 配置字典（机器人、任务、数据源）
├── scripts/                    # 转换、标注、验证脚本
│   ├── convert/
│   ├── annotate/
│   └── validate/
├── data/                       # Parquet 数据（按来源分）
├── videos/                     # MP4 视频文件
├── subgoals/                   # 子目标图像
├── meta/                       # episodes.jsonl、segments.jsonl、stats.json 等
├── examples/                   # ALOHA mini 示例
├── docs/
├── CONTRIBUTING.md
├── LICENSE
└── README.md
---

## 数据类别（共8类）

本数据集需要以下 **8 个类别** 的数据：

| 类别 | 是否含机器人动作 | 主要用途 | 优先级 |
|------|------------------|----------|--------|
| 1. 机器人演示数据 | 是 | VLA 动作训练（核心） | ★★★★★ |
| 2. 机器人自主执行数据 | 是 | 鲁棒性、失败学习 | ★★★★ |
| 3. 人工干预与指导数据 | 部分 | 高层语言策略、子任务指令 | ★★★★ |
| 4. 仿真机器人数据 | 是 | 任务多样性、自动标注 | ★★★★ |
| 5. 人类第一人称视频 | 否 | 世界模型、子目标生成 | ★★★ |
| 6. 人类第三人称视频 | 否 | 任务序列、可视化子目标 | ★★★ |
| 7. 网页多模态数据 | 否 | VLM 预训练、对齐 | ★★★ |
| 8. 派生上下文与标注数据 | 间接 | π₀.₇ prompt 构建（核心） | ★★★★★ |

### 类别1：机器人演示数据（最重要）

**开源资源**：
- **ALOHA-Cosmos-Policy、DROID、Open X-Embodiment (OXE)、BridgeData V2、RoboSet 等**

**自采建议**：强烈推荐采集自己实验室机器人的高质量遥操作演示（尤其是目标任务）。

**格式**：Parquet（每 episode 一个文件，每行一个 timestep）+ MP4 视频指针。

**标注规范**：
- task：从目录/文件名自动提取
- subtask：先自动均匀切分 → VLM 伪标注 → 人工审核
- quality：成功演示默认 5 分
- mistake：成功演示默认 false
- speed_bin：`round(T / 500) * 500`

### 类别2：机器人自主执行数据

**描述**：  
由已训练策略在评估、RL 微调或实际部署中生成的 episode。包含成功、失败和次优 rollout。使用带丰富元数据的次优数据来提升模型鲁棒性。

**格式**：  
与类别 1 一致（Parquet + MP4 指针），额外增加字段：  
- `policy_name`、`policy_version`  
- `rollout_type`（eval / rl / deployment）  
- `completion_score`（0-1 任务完成度）  
- `failure_reason`、`intervention_count`、`reward`（可选）

**可用开源数据集**：  
- DROID（部分评估数据）  
- RoboMimic / Offline RL 数据集  
- OXE 中包含的 non-expert 数据

**自采需求（必须）**：  
- 在实验室机器人上运行当前策略，保留**所有失败 episode**（不要丢弃）  
- 记录策略版本、checkpoint、干预时间戳  
- 重点采集边缘案例和长时序失败场景

**标注规范**：  
- `success` / `quality`：根据完成度和平滑度打分  
- `mistake`：自动检测抓取失败、碰撞、超时、重复尝试等  
- `failure_reason`：人工或 VLM 补充描述

### 类别3：人工干预与指导数据

**描述**：  
人类在机器人执行过程中提供逐步语言指令、纠正或遥操作接管的 episode。用于训练能自动生成 subtask 指令的高层语言策略。

**格式**：  
与类别 1 一致，额外字段：  
- `human_instruction`（本 timestep 的指令文本）  
- `intervention_type`（verbal_coaching / teleop_takeover / correction 等）  
- `target_subtask`  

另存独立标注文件：`annotations/coaching/episode_XXXX.json`

**可用开源数据集**：  
- RoboVQA（238 小时视频-文本对）  
- DROID 中带操作员笔记的部分

**自采需求（必须）**：  
- 记录人类语音/文字指令 + 时间戳  
- 区分语言指导与物理接管  
- 与实验室具体任务和环境强绑定

**标注规范**：  
- ASR 转录指令 → 对齐到 frame_index  
- 映射到目标 subtask  
- 标记干预是否成功纠正

### 类别4：仿真机器人数据

**描述**：  
仿真环境中生成的轨迹，具备完整状态、动作、物体信息和自动成功信号。用于扩展任务多样性和自动生成标签。

**格式**：  
与类别 1 一致，额外字段：  
- `domain: "sim"`  
- `simulator`（isaac / mujoco / robosuite 等）  
- `object_states`（JSON 物体位姿、状态）

**可用开源数据集**：  
- **InterData-A1**（多任务，已有 Parquet 结构）  
- LIBERO（语言条件基准）  
- RoboMimic（HDF5 基准）

**自采需求**：  
- 构建目标任务的数字孪生  
- 生成 domain randomization 数据（用于 sim-to-real）  
- 包含失败和边缘案例轨迹

**标注规范**：  
- 大部分字段可自动从仿真器导出（success、mistake、object_states）  
- 保留 `domain: sim` 和 `robot_type` 防止域混淆

### 类别5：人类第一人称视频数据

**描述**：  
头戴相机等第一人称人类视频。不含机器人动作，用于训练世界模型、子目标生成、物体可供性学习和类人 subtask 语言。

**格式**：  
- `video.path` + `frame_index`  
- `camera_view: "egocentric"`  
- `task`、`subtask`、`narration`、`objects`、`action_verb`、`target_object`

**可用开源数据集**：  
- **Ego4D**（3670 小时，规模最大）  
- **EPIC-KITCHENS-100**（厨房场景，密集叙述）

**自采需求**：  
- 使用与机器人相同环境和物体的第一人称演示  
- 边做边自然语言叙述

**标注规范**：  
- 从叙述或 VLM 生成 subtask  
- Clip 结束帧作为 subgoal 参考

### 类别6：人类第三人称视频数据

**描述**：  
教学视频、固定相机录制的人类执行任务视频。用于任务序列建模和可视化子目标学习。

**格式**：  
类似类别 5，`camera_view: "third_person"`，额外添加 `caption`、`scene_type`、`source_url`（网页来源仅存 URL + 元数据）。

**可用开源数据集**：  
- Ego4D / EPIC-KITCHENS 的第三人称视角  
- RoboVQA  
- YouTube 教学视频（注意版权，仅存元数据）

**自采需求**：  
- 固定相机录制与机器人工作空间一致的任务演示  
- 优先带叙述版本

**标注规范**：  
- ASR + VLM 生成 caption 和 subtask 分段  
- 人工审核关键边界

### 类别7：网页多模态数据

**描述**：  
来自互联网的图像、视频、文本、VQA、字幕等。用于 VLM 预训练和视觉-语言对齐，是 π₀.₇ 的辅助非机器人数据。

**格式**：  
- `modality`（image/video/text）  
- `task_type`（vqa / captioning / object_localization 等）  
- `question`、`answer`、`caption`、`object_boxes` 等

**可用开源数据集**：  
- Open Images（900 万图像，丰富标注）  
- COCO Captions  
- 各类 VQA 数据集

**自采需求**：  
- 实验室物体属性照片（颜色、材质、状态）  
- 机器人场景 VQA 对（“哪个抽屉是打开的？”）

**标注规范**：  
- 优先使用现有标注  
- VLM 生成机器人相关描述  
- 人工抽查过滤无关数据

### 类别8：派生上下文与标注数据（π₀.₇ 风格核心）

**描述**：  
从上述 1-7 类数据派生而来，是让数据集真正成为“π₀.₇-style”的关键。包含丰富的 prompt 上下文。

**主要子类别**：  
1. **Subtask 分段标注**（`meta/segments.jsonl` + 独立 json）  
2. **Episode 元数据**（quality、mistake、speed_bin 等）  
3. **Subgoal 图像**（real_future / segment_end / generated）  
4. **训练 Prompt 样本**（observation history + context + 50 步 action chunk）  
5. **机器人规格**（`configs/robots/*.yaml`）  
6. **数据集统计**（`meta/stats.json` 用于归一化）

**Subgoal 生成策略**（遵循论文）：  
- 75%：当前帧后 0-4 秒内随机未来帧  
- 25%：当前 subtask 结束帧  

**标注流程优先级**：  
- **最高**：subtask 分段（自动 → VLM → 人工）  
- **高**：quality（1-5）、mistake 标签  
- **自动**：speed_bin、real_future subgoal、prompt 样本生成

**标注状态**：`auto_placeholder` → `vlm_pseudo` → `human_labeled` → `reviewed` → `approved`

---

## 统一数据格式

所有机器人 episode 数据（类别1-4）必须符合 `schemas/robot_episode.schema.json` 定义的结构。

核心字段包括：
- `observation.state`、`action`
- `task`、`subtask`
- `video.{cam}.path` + `video.{cam}.frame_index`
- `subgoal.real_future.{cam}.*`
- `quality`、`mistake`、`speed_bin`、`robot_type`、`control_mode` 等

---

## 标注流程与规范

1. **自动阶段**：转换脚本自动填充 task、speed_bin、占位 subtask、quality=5 等
2. **VLM 辅助阶段**：使用视觉语言模型生成 subtask、mistake 伪标签
3. **人工审核阶段**：重点审核 subtask 分段、quality、mistake
4. 标注状态字段：`auto_placeholder` → `vlm_pseudo` → `human_labeled` → `reviewed` → `approved`

---

## 快速开始

```bash
# 1. 转换 ALOHA 示例
python scripts/convert/convert_aloha_cosmos.py \
  --aloha_root /path/to/ALOHA-Cosmos-Policy/... \
  --out_root examples/aloha_mini \
  --num_episodes 5

# 2. 检查数据
python scripts/validate/inspect.py --root examples/aloha_mini
```

---

## 路线图

- [x] ALOHA mini prototype
- [ ] 全量 ALOHA 转换器
- [ ] InterData-A1 转换器
- [ ] 浏览器 subtask 标注工具
- [ ] 更多数据集转换器
- [ ] 完整 validation pipeline

---

## 贡献指南

欢迎提交：
- 新数据集转换脚本
- 机器人配置（configs/robots/）
- 任务定义（configs/tasks/）
- 标注数据
- 代码改进

详见 [CONTRIBUTING.md](CONTRIBUTING.md)
