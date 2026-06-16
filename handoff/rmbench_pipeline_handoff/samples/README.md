# Episode 0 样例

三个目录对应同一条 `observe_and_pickup/episode0`。

目录层级、文件名和元数据组织均与实际数据一致。为控制体积，只保留 episode 0，
相关 `episodes`、`segments`、`stats` 和总数均已同步裁剪为 1 条、148 帧。

## raw

```text
raw/
└── observe_and_pickup/
    └── demo_clean/
        ├── data/
        │   └── episode0.hdf5
        ├── instructions/
        │   └── episode0.json
        └── language_annotation.json
```

HDF5 关键字段：

```text
/joint_action/vector                  (149, 14)
/observation/head_camera/rgb          (149,)
/observation/left_camera/rgb          (149,)
/observation/right_camera/rgb         (149,)
```

## infidata

```text
infidata/
└── observe_and_pickup/
    ├── data/rmbench/chunk-000/
    │   └── episode_000000.parquet
    ├── meta/
    │   ├── episodes.jsonl
    │   ├── segments.jsonl
    │   ├── memory_segments.jsonl
    │   ├── tasks.json
    │   ├── robots.json
    │   └── stats.json
    ├── videos/rmbench/
    │   ├── episode_000000_cam_high.mp4
    │   ├── episode_000000_cam_left_wrist.mp4
    │   └── episode_000000_cam_right_wrist.mp4
    └── README.md
```

parquet 共 148 行，每行一个 timestep。`summary` 为逐帧 memory。

## lerobot

```text
lerobot/
└── observe_and_pickup/
    ├── data/chunk-000/
    │   └── episode_000000.parquet
    ├── meta/
    │   ├── info.json
    │   ├── tasks.jsonl
    │   ├── episodes.jsonl
    │   └── episodes_stats.jsonl
    └── README.md
```

LeRobot parquet 共 148 行，三路图像以内嵌 PNG 保存。`memory` 与 InfiData 的逐帧 `summary` 完全一致。
