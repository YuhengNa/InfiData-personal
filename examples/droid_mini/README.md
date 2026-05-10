# droid_mini

This is a small InfiData-style prototype dataset converted from DROID (LeRobot v3 format).

## Contents

- Number of converted episodes: 3
- Source dataset: DROID
- Robot: Franka
- Control mode: joint
- FPS: 15
- Cameras mapped to InfiData keys:
  - cam_high <- observation.images.exterior_1_left
  - cam_left_wrist <- observation.images.wrist_left
  - cam_right_wrist <- observation.images.exterior_2_left

## Important note

This is a prototype for data pipeline validation.

The fields `quality`, `success`, and `mistake` are initialized from `is_episode_successful`.
The `subtask` segments are auto-generated placeholders and should be replaced by human or VLM-assisted annotations before real training.
