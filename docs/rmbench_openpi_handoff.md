# RMBench -> InfiData -> openpi 交付对接指南

本文档面向只拿到 InfiData 仓库的对接方，说明如何使用仓库内已经转换好的 RMBench 示例数据，在 openpi 中跑通一次训练 smoke test。

目标不是训练出有效策略，而是证明数据格式、LeRobot 加载、openpi transform、norm stats、训练 loop 和 checkpoint 保存可以端到端跑通。

## 1. 数据说明

InfiData 仓库中包含两份 RMBench 示例：

```text
examples/rmbench_mini/
examples/rmbench_lerobot_mini/
```

`examples/rmbench_mini` 是 InfiData 中间格式，包含 parquet、meta 和三路 mp4。

`examples/rmbench_lerobot_mini` 是面向 openpi 的 LeRobot 格式，包含 3 个 episode、1995 帧，字段包括：

| 字段 | 说明 |
| --- | --- |
| `observation.images.cam_high` | 头部相机图像 |
| `observation.images.cam_left_wrist` | 左腕相机图像 |
| `observation.images.cam_right_wrist` | 右腕相机图像 |
| `observation.state` | 14 维机器人状态 |
| `action` | 14 维动作 |
| `task` | episode 级任务语言 |
| `subtask` | RMBench 片段级语义目标 |
| `memory` | 当前与 `subtask` 相同，后续可替换为长期记忆标注 |

openpi 的 LeRobot loader 默认从 `$HF_LEROBOT_HOME/<repo_id>` 读取数据。本示例使用的 `repo_id` 是：

```text
wudi/rmbench_battery_try_demo3
```

因此需要把数据放到：

```text
$HF_LEROBOT_HOME/wudi/rmbench_battery_try_demo3
```

推荐使用大盘目录：

```bash
export HF_LEROBOT_HOME=/home/wudi/wudi_data/lerobot
mkdir -p "$HF_LEROBOT_HOME/wudi"
rsync -a examples/rmbench_lerobot_mini/ \
  "$HF_LEROBOT_HOME/wudi/rmbench_battery_try_demo3/"
```

如果对接方需要从 InfiData 中间格式重新生成 LeRobot 数据，可以运行：

```bash
export HF_LEROBOT_HOME=/home/wudi/wudi_data/lerobot
python scripts/convert2openpi/convert_rmbench.py \
  --infidata_root examples/rmbench_mini \
  --repo_id wudi/rmbench_battery_try_demo3 \
  --output_root "$HF_LEROBOT_HOME/wudi/rmbench_battery_try_demo3" \
  --num_episodes 3 \
  --fps 30 \
  --use_images \
  --overwrite
```

## 2. openpi 需要修改的地方

需要修改 openpi 仓库中的：

```text
src/openpi/training/config.py
```

修改内容分两部分：

1. 增加一个 `LeRobotRMBenchDataConfig`，把 LeRobot 数据字段映射成 openpi 训练时需要的标准字段。
2. 在 `_CONFIGS` 中增加 `debug_rmbench` 和 `pi0_rmbench_smoke` 两个训练配置。

### 2.1 增加 RMBench DataConfig

在 `LeRobotDROIDDataConfig` 或其他 LeRobot data config 后面加入：

```python
@dataclasses.dataclass(frozen=True)
class LeRobotRMBenchDataConfig(DataConfigFactory):
    """Data config for RMBench datasets converted through InfiData into LeRobot format."""

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=False)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=False)],
        )
        if self.use_delta_joint_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(_transforms.make_bool_mask(14))],
                outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(14))],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
```

这里复用 `aloha_policy.AlohaInputs(adapt_to_pi=False)`，原因是 RMBench 转换后的数据与 ALOHA 训练入口相同：三路相机、14 维 state、14 维 action。`adapt_to_pi=False` 表示不使用 ALOHA 专属的关节翻转和夹爪空间变换。

`action` 是下一帧绝对关节向量，因此配置中用 `DeltaActions(make_bool_mask(14))` 将 14 维动作都转换成相对当前 state 的 delta action，符合 openpi pi0 训练习惯。

### 2.2 增加训练配置

在 `_CONFIGS` 列表中加入：

```python
TrainConfig(
    name="debug_rmbench",
    model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
    data=LeRobotRMBenchDataConfig(
        repo_id="wudi/rmbench_battery_try_demo3",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=6,
    num_workers=0,
    num_train_steps=5,
    log_interval=1,
    save_interval=5,
    keep_period=None,
    overwrite=True,
    exp_name="rmbench_debug",
    wandb_enabled=False,
),
TrainConfig(
    name="pi0_rmbench_smoke",
    model=pi0_config.Pi0Config(),
    data=LeRobotRMBenchDataConfig(
        repo_id="wudi/rmbench_battery_try_demo3",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=6,
    num_workers=0,
    num_train_steps=5,
    log_interval=1,
    save_interval=5,
    keep_period=None,
    overwrite=True,
    exp_name="rmbench_pi0_smoke",
    wandb_enabled=False,
),
```

`debug_rmbench` 使用真实数据管线和 openpi dummy 模型，适合最先验证数据能否端到端跑起来。

`pi0_rmbench_smoke` 使用真实 pi0 模型结构，适合 GPU 资源充足时进一步做模型形状级 smoke test。

## 3. 运行前检查

在 openpi 仓库中执行：

```bash
cd /path/to/openpi
export HF_LEROBOT_HOME=/home/wudi/wudi_data/lerobot

python scripts/train.py --help | grep rmbench
```

如果能看到 `debug_rmbench` 和 `pi0_rmbench_smoke`，说明 config 已注册成功。

可以先用 LeRobot 直接读取数据：

```bash
python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "wudi/rmbench_battery_try_demo3",
    root="/home/wudi/wudi_data/lerobot/wudi/rmbench_battery_try_demo3",
)
s = ds[0]
print("len:", len(ds))
print("state:", s["observation.state"].shape)
print("action:", s["action"].shape)
print("cam_high:", s["observation.images.cam_high"].shape)
print("memory_eq_subtask:", s["memory"] == s["subtask"])
PY
```

预期输出应包含：

```text
len: 1995
state: torch.Size([14])
action: torch.Size([14])
cam_high: torch.Size([3, 240, 320])
memory_eq_subtask: True
```

## 4. 计算 normalization stats

openpi 训练非 fake 数据前需要先计算 norm stats：

```bash
cd /path/to/openpi

env HF_LEROBOT_HOME=/home/wudi/wudi_data/lerobot \
  python scripts/compute_norm_stats.py \
  --config-name debug_rmbench \
  --max-frames 60
```

输出路径应为：

```text
assets/debug_rmbench/wudi/rmbench_battery_try_demo3/norm_stats.json
```

`--max-frames 60` 只是为了快速 smoke test。正式训练时可以去掉，让脚本统计全部帧。

## 5. 跑通训练 smoke test

推荐先跑 `debug_rmbench`。如果机器上 JAX 能看到多张 GPU，`batch_size` 必须能被 `jax.device_count()` 整除。为了交付验证稳定，建议直接指定单卡：

```bash
cd /path/to/openpi

env HF_LEROBOT_HOME=/home/wudi/wudi_data/lerobot \
  CUDA_VISIBLE_DEVICES=2 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.2 \
  python scripts/train.py debug_rmbench \
  --exp-name rmbench_debug_single_gpu \
  --overwrite \
  --batch-size 1 \
  --num-train-steps 2 \
  --save-interval 2 \
  --model.action-horizon 10
```

本地已验证该命令可以跑通，关键信息如下：

```text
Initialized data loader:
[0].images['base_0_rgb']: (1, 224, 224, 3)@float32
[0].images['left_wrist_0_rgb']: (1, 224, 224, 3)@float32
[0].images['right_wrist_0_rgb']: (1, 224, 224, 3)@float32
[0].state: (1, 32)@float32
[1]: (1, 10, 32)@float32

Step 0: loss=2.0797
Step 1: loss=1.8069
```

checkpoint 会写到：

```text
checkpoints/debug_rmbench/rmbench_debug_single_gpu/1
```

## 6. 可选：pi0 结构 smoke test

如果希望用真实 pi0 模型结构做进一步检查，可以运行：

```bash
env HF_LEROBOT_HOME=/home/wudi/wudi_data/lerobot \
  python scripts/compute_norm_stats.py \
  --config-name pi0_rmbench_smoke \
  --max-frames 60

env HF_LEROBOT_HOME=/home/wudi/wudi_data/lerobot \
  CUDA_VISIBLE_DEVICES=2 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.4 \
  python scripts/train.py pi0_rmbench_smoke \
  --exp-name rmbench_pi0_smoke \
  --overwrite \
  --batch-size 1 \
  --num-train-steps 2 \
  --save-interval 2 \
  --model.action-horizon 10
```

如果使用 base checkpoint 微调，还需要把 `weight_loader` 改为对应的 openpi checkpoint，例如：

```python
weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params")
```

这一步可能需要网络或提前下载好的 openpi base checkpoint。

## 7. 常见问题

### 7.1 找不到数据

确认目录结构是：

```text
$HF_LEROBOT_HOME/
└── wudi/
    └── rmbench_battery_try_demo3/
        ├── data/
        └── meta/
```

不要多套一层 `examples/rmbench_lerobot_mini`。

### 7.2 batch size 报错

如果出现：

```text
Batch size ... must be divisible by the number of devices
```

要么设置 `CUDA_VISIBLE_DEVICES` 只暴露一张卡，要么把 `--batch-size` 调整为 GPU 数量的整数倍。

### 7.3 GPU OOM

优先使用：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU>
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.2
```

也可以降低：

```bash
--batch-size 1
--model.action-horizon 10
```

### 7.4 没有 norm stats

如果训练时报：

```text
Normalization stats not found
```

先运行：

```bash
python scripts/compute_norm_stats.py --config-name debug_rmbench --max-frames 60
```

## 8. 交付结论

只要完成以上 openpi 代码修改，并把 InfiData 仓库中的 `examples/rmbench_lerobot_mini` 放到 `$HF_LEROBOT_HOME/wudi/rmbench_battery_try_demo3`，就可以用 `debug_rmbench` 在 openpi 中跑通真实 RMBench 转换数据的训练链路。

该链路已经本地验证：数据加载、三路图像、语言 prompt、state/action、norm stats、loss 计算、反向传播和 checkpoint 保存均可运行。
