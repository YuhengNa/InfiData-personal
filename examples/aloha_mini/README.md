# my_pi07_dataset_mini

This is a small π0.7-style prototype dataset converted from ALOHA-Cosmos-Policy.

## Contents

- Number of converted episodes: 3
- Source dataset: ALOHA-Cosmos-Policy
- Robot: ALOHA 2 bimanual ViperX
- Control mode: joint
- FPS: 25
- Cameras: cam_high, cam_left_wrist, cam_right_wrist

## Important note

This is a prototype for presentation and data pipeline validation.

The fields `quality`, `success`, and `mistake` are initialized using the assumption that ALOHA-Cosmos-Policy contains successful demonstrations.
The `subtask` segments are automatically generated placeholders and should be replaced by human or VLM-assisted annotations before real training.

## π0.7-style fields

Each parquet row corresponds to one timestep and includes:

- observation.state
- observation.qvel
- observation.effort
- action
- action_relative
- task
- subtask
- robot_type
- control_mode
- quality
- speed_bin
- mistake
- success
- video paths and frame indices
- real-future subgoal frame pointers
