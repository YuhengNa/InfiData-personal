# RMBench LeRobot Mini

This dataset was converted from an InfiData RMBench dataset for LeRobot/openpi training.

## Contents

- LeRobot repo id: `wudi/observe_and_pickup`
- Converted episodes: 1
- FPS: 30
- Robot type: robotwin_dual_arm_sim
- Cameras: cam_high, cam_left_wrist, cam_right_wrist
- Images are stored as LeRobot `image` features inside parquet files.

## Features

- `observation.state`
- `action`
- `observation.images.cam_high`
- `observation.images.cam_left_wrist`
- `observation.images.cam_right_wrist`
- `subtask`
- `memory`

When `meta/memory_segments.jsonl` is present, `memory` is read from its `summary` field for the covered frame range.
Episodes with missing memory coverage are skipped by default, or fail immediately when converted with `--strict`.
By default, a final missing tail of at most 2 frames is filled by extending the last real memory segment.
