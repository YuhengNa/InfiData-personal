# rmbench_mini

This is an InfiData-style prototype dataset converted from RMBench.

## Contents

- Number of converted episodes: 1
- Source dataset: RMBench
- RMBench task: observe_and_pickup
- RMBench config: demo_clean
- Robot: RoboTwin dual-arm simulation
- Control mode: joint
- FPS: 30
- Cameras: cam_high, cam_left_wrist, cam_right_wrist

## Field notes

- `observation.state` is RMBench `/joint_action/vector` at timestep `t`.
- `action` is the next-step RMBench `/joint_action/vector` at timestep `t + 1`.
- `task` is read from RMBench `instructions/episode*.json`.
- `subtask` and `meta/segments.jsonl` are converted from RMBench `language_annotation.json`.
- `subgoal.real_future.*` is a future-frame pointer, not a human semantic goal label.
- `video.cam_*.path` and `video.cam_*.frame_index` point to exported mp4 files.

## Important notes

This converter does not create placeholder task, subtask, state, action, or camera data.
Episodes with missing required RMBench files or inconsistent frame counts are skipped by default, or fail immediately with `--strict`.
Subtasks are derived from `language_annotation.json`, and segment records use `annotation_status=human_labeled` with `source_annotation_status=rmbench_language_annotation`.
