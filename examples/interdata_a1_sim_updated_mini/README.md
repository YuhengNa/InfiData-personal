# interdata_a1_sim_updated_mini

This is a small InfiData-style dataset converted from InterData-A1 `sim_updated`
(LeRobot v2.1).

## Contents

- Converted task root: `/mnt/data/szeluresearch/ELUBrain/InterData-A1/extracted_lerobot/sim_updated/articulation_tasks/franka/close_the_electriccooker/close_the_electriccooker`
- Number of episodes: 3
- Robot: Franka single arm
- Domain: simulation
- Control mode: joint
- FPS: 30
- State: 7 joint positions + 1 gripper position
- Action: 7 joint targets + 1 gripper target
- Future visual target offset: 2 seconds

## Camera mapping

- `images.rgb.head` -> `video.cam_high.*`
- `images.rgb.hand` -> `video.cam_right_wrist.*`

InterData-A1 has one generic hand camera. `cam_right_wrist` is only an InfiData
schema compatibility alias and does not assert physical right-arm placement.

## Derived and preserved fields

- `observation.qvel` is a finite difference of `observation.state`.
- `action_relative = action - observation.state`, matching the InternVLA-A1
  delta-action convention.
- Camera calibration, EE/TCP poses, and `master_actions.*` are preserved as
  source-specific extension columns.
- `subgoal.real_future.*` points to a future frame in the same source video.

## Annotation warning

InterData-A1 does not provide InfiData `subtask`, `quality`, `mistake`, or
`success` annotations. This prototype writes:

- `subtask = task`
- `quality = 3`
- `mistake = false`
- `success = true`

These values use `auto_placeholder` provenance fields and are present only to
satisfy the current required InfiData schema. They must be replaced or reviewed
before quality-conditioned or failure-recovery training.
