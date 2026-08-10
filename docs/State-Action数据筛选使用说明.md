# RLDS State-Action 质量筛选与数据落盘对接文档

本文档是两个正式功能的交接入口：

1. 对原始 RLDS 的 `train` split 执行 State-Action S1/S2/S3 质量筛选，生成可审计记录；
2. 按已确认的 P0–P5 策略将筛选记录转为最终决策，生成一份独立、可训练的 RLDS 副本。

整个流程不会删除或改写原始 `/data/wudi/RLDS` 数据。

## 1. 代码与数据位置

两个功能所需的正式代码和入口脚本均在本仓库内：

| 文件 | 用途 |
|---|---|
| `scripts/quality/filter_rlds_state_action.py` | 单个 RLDS split 的 S1/S2/S3 筛选 |
| `scripts/quality/run_all_rlds_state_action.sh` | 顺序执行当前 44 个 RLDS 配置的正式筛选 |
| `scripts/quality/select_robomind_quality_candidates.py` | 应用 RoboMIND P3–P5 特殊策略 |
| `scripts/quality/build_final_rlds_decisions.py` | 生成并强校验 44 个配置的最终 episode 决策 |
| `scripts/quality/materialize_filtered_rlds.py` | 按决策重写 train TFRecord，复制其他 split 和 metadata |
| `scripts/quality/run_final_rlds_materialization.sh` | 重建最终决策并创建/续传筛选后 RLDS 的总入口 |

数据与运行产物不纳入 Git：

- 原始 RLDS：`/data/wudi/RLDS`
- S1/S2/S3 记录：`/data/wudi/InfiData-personal/quality_runs/all_rlds`
- 最终决策：`/data/wudi/InfiData-personal/quality_runs/final_selection`
- 筛选后 RLDS：`/data/wudi/RLDS_State_Action_Filtered_20260807`

`quality_runs/` 已在 `.gitignore` 中。新 clone 的仓库不包含筛选记录，需先执行第 3 节生成。

## 2. 环境

当前可用环境：

```bash
source /data/wudi/.venvs/infidata-quality/bin/activate
```

重建环境时需安装：

```bash
uv venv /data/wudi/.venvs/infidata-quality --python 3.12
uv pip install --python /data/wudi/.venvs/infidata-quality/bin/python \
  'numpy<2.3' scipy matplotlib tensorflow-cpu tensorflow-datasets tqdm
```

## 3. 功能一：State-Action 筛选

### 3.1 检查内容

- **S1 Sudden Change**：使用 Median Filter、Savitzky–Golay 和中心差分计算 residual/acceleration/jerk，以数据集级单侧鲁棒上限检测突变。
- **S2 State-Action Alignment**：只在物理语义和 layout 可比时计算方向一致性与时延；未明确标记 delta 的原生 action 默认按 absolute 处理。
- **S3 Extreme Value**：按维度估计 q01/q99，用 `alpha=1.5` 扩展范围检测极端值。

Gripper 的正常离散开合不参与 S1/S3 幅值检查，也不参与 S2；但任意维度中的 NaN/Inf 仍然标记为异常。详细定义见 `docs/Qwen数据筛选方案.txt`。

### 3.2 运行单个数据集

```bash
/data/wudi/.venvs/infidata-quality/bin/python \
  /data/wudi/InfiData-personal/scripts/quality/filter_rlds_state_action.py \
  --dataset-dir /data/wudi/RLDS/realworld_piper_2/realworld_piper_infidata/1.0.0 \
  --split train \
  --output-dir /data/wudi/InfiData-personal/quality_runs/all_rlds/realworld_piper_2/realworld_piper_infidata/1.0.0/train
```

正式运行时会显示校准和逐 episode 筛选的进度、速度与 ETA。

### 3.3 运行全部 44 个配置

```bash
cd /data/wudi/InfiData-personal
bash scripts/quality/run_all_rlds_state_action.sh
```

该脚本不使用自动发现、循环或条件分支，而是显式列出 44 条已验收命令和对应 gripper 参数。运行时不要并发启动多个全量筛选任务。

每个配置的主要产物：

- `summary.json`：扫描总数、命中数和运行配置；
- `episodes.jsonl`：逐 episode 的 S1/S2/S3 结论与证据；
- `flagged_frames.jsonl`：异常帧、维度、数值和阈值；
- `s1_thresholds.json` / `s3_thresholds.json`：数据集级阈值；
- `review/*.png`：人工复核曲线。

## 4. 最终删除策略

不得直接用原始 `flagged_any` 删除数据。最终决策使用已确认的 P0–P5：

| 策略 | 数据集 | 最终判定 |
|---|---|---|
| P0 | AgiBot、realworld_piper_2、realworld_piper_task_split | 使用 S1、S2、S3 去重并集 |
| P1 | DROID | S1 只看 state；保留 S2、S3 |
| P2 | EgoVerse_full、RoboCOIN | 不使用 S1；只使用 S2、S3 |
| P3 | RoboMIND master–puppet | S1 只看 state；不使用 S2；保留 S3 |
| P4 | RoboMIND Franka 仿真 | S1 至少命中 3 帧且占比至少 1%；保留 S2、S3 |
| P5 | RoboMIND Tiankung s38 | S1/S2 只看双臂 `0-6,19-25`；S2 要求 `active_samples>=20`；保留 S3 |

当前固定验收口径是 44 个配置、355,654 条 episode，删除 31,585（8.88%），保留 324,069（91.12%）。当数据集版本或策略变化时，应同步评审并更新 `build_final_rlds_decisions.py` 中的期望数量，不应绕过强校验。

## 5. 功能二：按筛选记录落盘

### 5.1 正式运行

确认第 3 节的 44 个正式记录已完成后，直接运行：

```bash
/data/wudi/InfiData-personal/scripts/quality/run_final_rlds_materialization.sh
```

该入口会顺序执行：

1. 重建 RoboMIND P3–P5 决策；
2. 重建全部 P0–P5 决策并强校验数量；
3. 将筛选后的独立 RLDS 写入 `/data/wudi/RLDS_State_Action_Filtered_20260807`。

这是长时间任务，建议在 `tmux` 内运行。同一条命令可以安全重跑：已完成的 train shard 会根据 checkpoint 校验后跳过，未完成文件通过 `.incomplete` 临时文件重写，不会当作成功产物。

### 5.2 只做落盘预检

不带 `--execute` 时只校验 manifest、源数据和空间，不写数据：

```bash
/data/wudi/.venvs/infidata-quality/bin/python \
  /data/wudi/InfiData-personal/scripts/quality/materialize_filtered_rlds.py \
  --source-root /data/wudi/RLDS \
  --output-root /data/wudi/RLDS_State_Action_Filtered_20260807
```

### 5.3 落盘行为

- 原始 RLDS 始终只读；
- 只对 `train` TFRecord 进行 episode 级过滤，保留的 serialized Example 原样复制，不解码/重编码图像；
- 其他 split 和 metadata 逐字节复制；
- 更新 `dataset_info.json` 中 train 的 shard length 和字节数；
- train 被全部删除的配置不进入训练清单。

运行中会显示全局字节进度、速度和 ETA。完成后根目录应包含：

- `_QUALITY_FILTER_BATCH_SUMMARY.json`：全局落盘汇总；
- `_TRAIN_DATASETS.json`：最终可训练配置清单；
- 每个配置的 `_QUALITY_FILTER_SUCCESS.json`、`dataset_info.json` 和 `_SUCCESS`。

## 6. 下游使用

训练代码应显式指向新副本：

```bash
export RLDS_DATA_DIR=/data/wudi/RLDS_State_Action_Filtered_20260807
```

可训练数据集应以 `${RLDS_DATA_DIR}/_TRAIN_DATASETS.json` 为准，不要再从目录名自动推断。

## 7. 对接验收清单

1. `run_all_rlds_state_action.sh` 的 44 条命令全部成功；
2. `quality_runs/all_rlds` 下每个配置都有 `summary.json` 和 `episodes.jsonl`；
3. 最终决策校验为 44 / 355,654 / 31,585 / 324,069；
4. 落盘根目录存在 `_QUALITY_FILTER_BATCH_SUMMARY.json` 和 `_TRAIN_DATASETS.json`；
5. 每个需要训练的配置均有 `_QUALITY_FILTER_SUCCESS.json` 和 `_SUCCESS`；
6. 原始 `/data/wudi/RLDS` 的目录和文件未被改写。
