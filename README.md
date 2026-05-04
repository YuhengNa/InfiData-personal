# InfiData Open Dataset

**π₀.₇-style 富上下文机器人数据集共建仓库**

本仓库提供统一的 **π₀.₇-style** 数据规范、转换工具、标注流程和示例，帮助社区构建高质量、多源、富上下文的机器人学习数据集。

## 核心理念（基于 π₀.₇ 论文）
- 不仅包含 observation + action，还包含 **task / subtask / subgoal images / episode metadata**（quality、speed_bin、mistake 等）。
- 支持多机器人、多域（real/sim）、高质量+次优数据混合训练。

## 仓库结构
- `schemas/` — JSON Schema 校验规范
- `configs/` — 数据源、机器人、任务字典
- `scripts/` — 转换、标注、验证工具
- `examples/aloha_mini/` — ALOHA-Cosmos-Policy 小样本示例
- `data/`, `videos/`, `subgoals/`, `meta/` — 数据存储目录（大文件不提交 GitHub）

## Quick Start

```bash
# 1. 转换 ALOHA 示例
python scripts/convert/convert_aloha_cosmos_mini.py \
  --aloha_root /path/to/ALOHA-Cosmos-Policy/preprocessed/fold_shirt_demos_1500_steps_each_pt1/train \
  --out_root examples/aloha_mini \
  --num_episodes 3 \
  --task "fold shirt"

# 2. 检查数据
python scripts/validate/inspect_mini_dataset.py --root examples/aloha_mini --episode 0
```

## 下一步计划
- 完善全量 ALOHA 转换器
- 添加 InterData-A1 转换器
- 开发浏览器 subtask 标注工具
- 加入 validation CI

欢迎贡献新数据集转换器、标注数据、机器人配置等！

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。