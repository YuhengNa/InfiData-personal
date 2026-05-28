# biplay_mini

This is a BiPlay prototype dataset converted from raw HDF5 into InfiData format.

## Contents

- Number of converted episodes: 1
- Source dataset root: /mnt/data/szeluresearch/ELUBrain/BiPlay/aloha_pen_uncap_diverse_raw
- Robot: ALOHA 2 bimanual ViperX
- Control mode: joint
- FPS: 25
- Cameras: cam_high, cam_left_wrist, cam_low, cam_right_wrist

## Important note

This build only converts raw HDF5 subsets that are locally available.
The current BiPlay workspace contains raw HDF5 for aloha_pen_uncap_diverse_raw; other subsets are present only as TFRecord release packages.
The `subtask` segments are automatically generated placeholders.
