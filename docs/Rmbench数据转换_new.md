# RMBench 数据转换说明

本文档简要说明 RMBench 原始格式、转换后的 InfiData 格式、最终适配 openpi07 训练的 LeRobot 格式，以及对应转换脚本。

## 1. RMBench 原始格式

完整 task 目录结构：

```text
RMBench/data/<task_name>/demo_clean/
├── data/
│   ├── episode0.hdf5
│   ├── episode1.hdf5
│   └── ...
├── instructions/
│   ├── episode0.json
│   ├── episode1.json
│   └── ...
└── language_annotation.json
```

转换脚本实际使用的主要输入字段如下；其中前三类来自 HDF5，后两类来自 HDF5 同级目录下的 JSON 文件：

| 原始位置                        | 文件类型     | 含义                            | 转换去向                         |
| ------------------------------- | ------------ | ------------------------------- | -------------------------------- |
| `/joint_action/vector`          | HDF5 dataset | 机器人关节向量，shape `[T, 14]` | `observation.state` 和 `action`  |
| `/observation/head_camera/rgb`  | HDF5 dataset | 头部相机图像                    | `cam_high`                       |
| `/observation/left_camera/rgb`  | HDF5 dataset | 左腕相机图像                    | `cam_left_wrist`                 |
| `/observation/right_camera/rgb` | HDF5 dataset | 右腕相机图像                    | `cam_right_wrist`                |
| `instructions/episode*.json`    | JSON 文件    | episode 级任务文本              | `task`                           |
| `language_annotation.json`      | JSON 文件    | 分段级语言标注                  | `subtask`、`meta/segments.jsonl` |

动作定义：

```text
observation.state[t] = /joint_action/vector[t]
action[t] = /joint_action/vector[t + 1]
```



## 2. InfiData 格式

每个 episode 一个 parquet：

```text
<out_root>/<task_name>/
├── data/rmbench/chunk-000/episode_000000.parquet
├── meta/
│   ├── episodes.jsonl
│   ├── segments.jsonl
│   ├── tasks.json
│   ├── robots.json
│   └── stats.json
└── videos/rmbench/
    ├── episode_000000_cam_high.mp4
    ├── episode_000000_cam_left_wrist.mp4
    └── episode_000000_cam_right_wrist.mp4
```

`meta/` 文件：

| 文件 | 内容 |
|---|---|
| `episodes.jsonl` | 每个 episode 的来源、帧数、fps、parquet 路径、视频路径等 |
| `segments.jsonl` | subtask 分段边界和语言标注 |
| `tasks.json` | task 文本与 RMBench task 名映射 |
| `robots.json` | robot 类型、state/action 维度、相机列表 |
| `stats.json` | state/action 的 min/max/mean/std，以及 skipped episode |

`videos/rmbench/`：

| 文件 | 内容 |
|---|---|
| `episode_******_cam_high.mp4` | `/observation/head_camera/rgb` 导出视频 |
| `episode_******_cam_left_wrist.mp4` | `/observation/left_camera/rgb` 导出视频 |
| `episode_******_cam_right_wrist.mp4` | `/observation/right_camera/rgb` 导出视频 |

parquet 每行是一个 timestep：

| 字段 | 类型/维度 | 说明 | 备注 |
|---|---|---|---|
| `episode_index` | int | 转换后的 episode id | 真实索引 |
| `source_episode_index` | int | RMBench 原始 episode id | 真实索引 |
| `frame_index` | int | 当前帧 | 真实索引 |
| `timestamp` | float | `frame_index / fps` | 由 fps 计算 |
| `observation.state` | float[14] | 当前关节状态 | 真实数据 |
| `action` | float[14] | 下一帧关节目标 | 真实数据 |
| `task` | str | episode 级任务文本 | 真实数据，来自 instructions |
| `subtask` | str | 当前帧所属分段语言标注 | 真实数据，来自 language_annotation |
| `robot_type` | str | `robotwin_dual_arm_sim` | 固定元信息 |
| `control_mode` | str | `joint` | 固定元信息 |
| `quality` | int | 当前固定为 `5` | ==固定假设，后续可细化质量标注== |
| `speed_bin` | int | 按 episode 长度分桶 | 自动计算 |
| `mistake` | bool | 当前为 `False` | ==固定假设，后续可标失败/错误片段== |
| `success` | bool | 当前为 `True` | ==固定假设，后续可接入真实成功标注== |
| `source_dataset` | str | `RMBench` | 固定元信息 |
| `domain` | str | `sim` | 固定元信息 |
| `rmbench_task_name` | str | RMBench task 名 | 真实 task 目录名 |
| `video.cam_*.path` | str | 三路相机 mp4 相对路径 | 由 HDF5 图像导出 |
| `video.cam_*.frame_index` | int | 对应视频帧号 | 真实帧索引 |
| `subgoal.real_future.cam_*.path` | str | 未来帧所在视频 | 自动未来帧指针 |
| `subgoal.real_future.cam_*.frame_index` | int | 当前帧后约 2 秒的未来帧 | 自动指针，不是人工语义标注 |

说明：

- `subtask` 来自 RMBench `language_annotation.json`，不是 fake data

- `subgoal.real_future.*` 是未来帧指针，不是人工语义目标文本。

- 缺文件、相机帧不足、语言标注覆盖不完整时不会填 fake data；`--strict` 下会直接报错。

- 目前唯一缺少的是memory字段，这个需要标注

  

## 3. LeRobot/openpi07 格式

输出目录：

```text
$HF_LEROBOT_HOME/<org>/rmbench_<task_name>/
├── data/chunk-000/episode_000000.parquet
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
└── README.md
```

parquet中的主要 features：

| 字段                                 | 类型/维度      | 说明                                         |
| ------------------------------------ | -------------- | -------------------------------------------- |
| `observation.state`                  | float32[14]    | 状态                                         |
| `action`                             | float32[14]    | 动作                                         |
| `observation.images.cam_high`        | image[3, H, W] | 头部相机                                     |
| `observation.images.cam_left_wrist`  | image[3, H, W] | 左腕相机                                     |
| `observation.images.cam_right_wrist` | image[3, H, W] | 右腕相机                                     |
| `subtask`                            | string         | 分段语义                                     |
| `memory`                             | string         | 当前等于 `subtask`，后续可替换为长期记忆标注 |
| `task`                               | metadata       | episode 级任务文本                           |

说明：

+ parquet中，memory和task是为了适配pi07额外加的，其他内容是pi05版本也具有的
+ 即，目前在最终训练时，还是把subtask & memory信息放在了每timestamp帧中，而不是放在meta中





## 4. 转换脚本

执行路径均在/mnt/workspace/wudi/InfiData-personal

### 4.1 RMBench -> InfiData

单个 task：

```bash
python scripts/convert/convert_rmbench_mini.py \
  --rmbench_root /mnt/data/wudi/RMBench/data/data \
  --task_name battery_try \
  --out_root /mnt/data/InfiData/test \
  --num_episodes 999999 \
  --strict
```

所有 task：

```bash
python scripts/convert/convert_all_rmbench_tasks.py \
  --rmbench_root /mnt/data/wudi/RMBench/data/data \
  --out_root /mnt/data/InfiData/test \
  --strict \
  --skip_existing
```

常用参数：

| 参数              | 说明                             |
| ----------------- | -------------------------------- |
| `--rmbench_root`  | RMBench `data` 根目录            |
| `--task_name`     | 单个 task 名                     |
| `--task_config`   | 默认 `demo_clean`                |
| `--out_root`      | 输出目录                         |
| `--num_episodes`  | 每个 task 最多转换多少个 episode |
| `--strict`        | 遇到异常直接报错                 |
| `--skip_existing` | 批量转换时跳过已完成 task        |



### 4.2 InfiData -> LeRobot/openpi07

单个 task：

```bash
export HF_LEROBOT_HOME=/mnt/data/lerobot

python scripts/convert2openpi/convert_rmbench.py \
  --infidata_root /mnt/data/InfiData/Rmbench_infi/observe_and_pickup \
  --repo_id wudi/observe_and_pickup \
  --output_root "$HF_LEROBOT_HOME/wudi/observe_and_pickup" \
  --fps 30 \
  --use_images \
  --overwrite
```

所有 task：

```bash
cd /path/to/InfiData-personal

export HF_LEROBOT_HOME=/mnt/data/lerobot
INFI_ROOT=/mnt/data/InfiData/Rmbench_infi/
ORG=wudi

for task_dir in "$INFI_ROOT"/*; do
  [ -d "$task_dir" ] || continue
  task_name=$(basename "$task_dir")

  python scripts/convert2openpi/convert_rmbench.py \
    --infidata_root "$task_dir" \
    --repo_id "$ORG/rmbench_${task_name}" \
    --output_root "$HF_LEROBOT_HOME/$ORG/rmbench_${task_name}" \
    --fps 30 \
    --use_images \
    --overwrite
done
```

注意：

- `HF_LEROBOT_HOME` 是 LeRobot 数据根目录，openpi07 会按 `$HF_LEROBOT_HOME/<repo_id>` 查找数据。

- Openpi 对 lerobot版本有要求，需要旧版 LeRobot API：

  ```python
  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
  ```

  对应 LeRobot 版本为 openpi07 lock 中的：

  ```text
  lerobot 0.1.0
  git rev 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
  pip install \
    "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
  ```

  若环境正确，应能通过：
  
  ```bash
  "$OPENPI_PYTHON" -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; print('ok')"
  ```



+ 当前推荐使用：

  ```bash
  --use_images
  ```

  即把图像直接写入 LeRobot parquet。该格式已验证可被 openpi07 读取





## 5. 训练对接

在得到上面所述的转换后的Lerobot数据后，依然需要对Openpi code进行以下的修改，从而将数据对接到训练中

需要修改 openpi 仓库中的：

```text
src/openpi/training/config.py
```

修改内容分两部分：

1. 增加一个 `LeRobotRMBenchDataConfig`，把 LeRobot 数据字段映射成 openpi 训练时需要的标准字段。
2. 在 `_CONFIGS` 中增加 `debug_rmbench` 和 `pi0_rmbench_smoke` 两个训练配置。

### 5.1 增加 RMBench DataConfig

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

### 5.2 增加训练配置

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

### 5.3. 运行前检查

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

### 5.4. 计算 normalization stats

openpi 训练数据前需要先计算 norm stats：

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

### 5.5. 跑通训练 smoke test

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

### 5.6. 可选：pi0 结构 smoke test

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



## 6.aliyun资料存档

aliyun服务器上的数据的对应位置：

##### Rmbench源数据：

/mnt/workspace/wudi/RMBench/data/data

##### Rmbench infidata格式中间层数据

/mnt/workspace/InfiData/Rmbench_infi

##### rmbench pi07 lerobot格式数据

/mnt/workspace/lerobot/wudi/rmbench_battery_try_test
