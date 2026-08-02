# RLDS State-Action 数据筛选（S1/S2/S3）

实现脚本：`scripts/quality/filter_rlds_state_action.py`

本工具对应 `docs/Qwen数据筛选方案.txt` 的第一部分，只执行：

- S1：Median Filter + Savitzky–Golay 平滑，并联合判断 residual 与 acceleration/jerk；
- S2：在物理语义和 layout 可比时，以 cross-correlation 寻找延迟并计算 Directional Agreement；
- S3：按维度拟合 q01/q99，并使用可配置的 alpha 扩展范围寻找极端值。

工具只生成筛选清单和人工复核材料，不修改源 RLDS。
`quality_runs/` 已加入 `.gitignore`，运行结果保存在仓库目录中但不会提交到 Git。
运行时会显示 RLDS 读取、S1 State/Action 阈值拟合和逐 episode 筛选的进度、速度与 ETA。

## 实现约定

1. S1 和 S3 先基于本次扫描的数据拟合数据集级阈值，再逐 episode 判断。
2. S1 使用 MAD 鲁棒尺度，并以每维 q01-q99 范围的 0.2% 作为最小尺度，按
   `center + z_threshold × scale` 进行单侧上限检测。Acceleration 的首尾帧和
   Jerk 的前后两帧没有完整中心差分邻域，不参与阈值拟合和检测。
3. 从 `state_action_schema_json` 的 layout 自动识别 gripper。gripper 的正常开合是离散/双峰信号，
   S1/S3 不对其执行突变或极值规则，但其中的 NaN/Inf 仍直接视为异常。
4. S3 拟合前将所有 NaN/Inf 排除出分位数计算，记录每维有限样本数和无效阈值
   维度。超过一百万行时，从整个 split 的全局行索引中等概率无放回抽样。
5. S2 默认 DA 阈值为 0.6，搜索 ±10 帧延迟。计算方向时把绝对值不超过
   `motion_epsilon=1e-5` 的差分视为静止；负延迟表示 state 领先 action，按反因果风险标记。
   S2 默认跳过 state/action 任一侧标记为 gripper 的维度。
6. action 为 delta 时先积分。state/action layout 不同或物理语义不明确时，S2 会标为 skipped，不进行错误比较。
   `action_is_delta` 仅接受布尔值或字符串 `"true"`/`"false"`。根据当前项目
   的原生 RLDS 均为 absolute source target 的约定，`"unknown"`、缺失或非法值
   默认按 absolute 处理，并在输出中记录 `action_semantics_source=default_absolute`。
7. S2 命中建议 episode 级排除；S1/S3 命中只建议复核或过滤相应帧。

## 环境

本次使用的独立环境：

```bash
source /data/wudi/.venvs/infidata-quality/bin/activate
```

若需重新创建：

```bash
uv venv /data/wudi/.venvs/infidata-quality --python 3.12
uv pip install --python /data/wudi/.venvs/infidata-quality/bin/python \
  'numpy<2.3' scipy matplotlib tensorflow-cpu tensorflow-datasets
```

## 算法测试是什么

`scripts/quality/test_rlds_state_action_filter.py` 是不读取真实 RLDS 的快速回归测试。它构造结果
已知的合成信号，检查：

- S1 能命中人为注入的尖峰，只检测异常大的指标，排除有限差分边界，并在跳过
  gripper 正常开合的同时保留 NaN/Inf；
- S2 能恢复已知的 3 帧 Action→State 延迟、识别反向趋势，并拒绝不兼容 layout；
- S3 能命中极端值，同时不误筛 gripper。

保留该文件是为了防止以后修改阈值、平滑方法或重构代码时破坏核心逻辑。它只验证算法
行为，不代表真实数据质量，也不会产生或提交 `quality_runs`。

## 运行

完整扫描 realworld_piper_2 train：

```bash
python scripts/quality/filter_rlds_state_action.py \
  --dataset-dir /data/wudi/RLDS/realworld_piper_2/realworld_piper_infidata/1.0.0 \
  --split train \
  --output-dir /data/wudi/InfiData-personal/quality_runs/realworld_piper_2_train
```

小规模 smoke test 可以增加 `--max-episodes 1`，但小样本拟合的 S1/S3
阈值不适合用作最终筛选结论。

## 筛选全部 RLDS 仓库

`scripts/quality/run_all_rlds_state_action.sh` 不包含变量、循环、条件判断或自动发现逻辑。
其中按数据集分组，明确列出了当前 49 个 RLDS version 的 49 条完整 Python 命令。
每条命令都写明输入目录、`train` split 和独立输出目录。

先查看命令清单：

```bash
less /data/wudi/InfiData-personal/scripts/quality/run_all_rlds_state_action.sh
```

可以从文件中复制任意一条 Python 命令，单独筛选对应的 RLDS version。需要顺序执行
全部 49 条命令时：

```bash
cd /data/wudi/InfiData-personal
bash scripts/quality/run_all_rlds_state_action.sh
```

如需后台执行：

```bash
nohup bash scripts/quality/run_all_rlds_state_action.sh \
  > quality_runs/all_rlds.log 2>&1 &
```

随后用 `tail -f quality_runs/all_rlds.log` 查看进度。当前命令清单刻意不实现断点跳过：
如任务中断，直接从文件中找到尚未完成的数据集命令，并单独执行即可。若要筛选
`unseen_test`，复制对应命令后将 `--split train` 和输出目录末尾的 `train` 改为
`unseen_test`。

全量 RLDS 声明规模约 46 TB，运行会产生大量 BOS 读取并持续较长时间。当前单数据集
筛选器会在内存中保留该 dataset version 的数值 State/Action；每条命令结束后内存会被
释放，但超大单库仍需监控内存。不要并发启动多个全量筛选任务。

## 输出

- `summary.json`：运行配置和规则命中数量；
- `episodes.jsonl`：每个 episode 的结论、S1/S2/S3 详情和建议动作；
- `flagged_frames.jsonl`：S1/S3 命中的帧、维度、数值和阈值证据；
- `s1_thresholds.json`、`s3_thresholds.json`：可复现的数据集级阈值；
- `review/*.png`：命中 episode 的 state/action 曲线和异常帧标线。

## 首次测试结果

数据集：
`/data/wudi/RLDS/realworld_piper_2/realworld_piper_infidata/1.0.0`

| 测试 | Episodes | S1 | S2 | S3 |
|---|---:|---:|---:|---:|
| seen_test（校准后） | 45 | 0 | 0 | 0 |
| train 全量 | 902 | 3 | 0 | 0 |

全量 train 无 TFRecord 解码失败或 state/action shape 异常。

人工复核：

- Episode 123，frame 271，joint dim 5：短时反向运动，连续且幅度不大，更像正常快速动作，倾向假阳性；
- Episode 18，frames 1035–1036，joint dim 7：约两帧的明显速度突增，建议结合视频复核；
- Episode 205，frame 454，joint dim 11：单帧步长显著高于邻帧，建议结合视频复核。

三条候选的 action 均与 state 保持严格的 1 帧领先关系，S2 DA 接近 1。
因此目前没有证据将它们直接认定为 packet loss 或错配数据，工具将其保留为人工复核候选。

注意：该数据集的 action 表示为
`next_step_absolute_joint_position_with_gripper`，S2 的严格一致也可能部分来自
action 的构造方式。S2 在此数据集上能验证转换后的时序关系，但不是完全独立的传感器交叉验证。
