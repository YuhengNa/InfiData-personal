# Remaining RoboMIND / RoboCOIN RLDS Run Commands

Generated from `progress.md`, wrapper scripts, and the current RLDS output directories.

Completion rule used here:

- RoboCOIN is complete only if
  `/mnt/workspace/RLDS/RoboCOIN/<repo>/robocoin_infidata/1.0.0/dataset_info.json` exists.
- RoboMIND full is complete only if
  `/mnt/workspace/RLDS/RoboMIND_full/<repo>/robomind_full_infidata/1.0.0/dataset_info.json` exists.
- Existing output directories without `dataset_info.json` are treated as unfinished and are listed below.

Notes:

- RoboCOIN has 22 RLDS repos but 21 wrapper scripts. `unitree_g1_s28_a28_fps30__episodes_1158.sh` converts two RLDS repos: `cam_high` and `cam_high,cam_left_wrist,cam_right_wrist`.
- RoboMIND `franka_sim_simulation_franka_joint_position_h5_sim_franka_3rgb_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external__episodes_14488.sh` is marked unfinished in `progress.md`, but it now has `dataset_info.json`, so it is not listed below.

## RoboCOIN

17 wrapper scripts remain unfinished, covering 18 RLDS repos.

```bash
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/Agilex_Cobot_Magic_s26_a26_fps30__episodes_8284.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/Galaxea_R1_Lite_s14_a14_fps30__episodes_3650.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/Realman_RMC-AIDA-L_s28_a28_fps30__episodes_693.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/Unitree_G1_Dex3_phecda_s28_a28_fps30__episodes_1411.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/agilex_cobot_decoupled_magic_s26_a26_fps30__episodes_23712.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/agilex_cobot_decoupled_magic_s14_a14_fps50__episodes_3397.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/aloha_s26_a26_fps30__episodes_4879.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/alpha_bot_2_s28_a28_fps30__episodes_857.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/discover_robotics_aitbot_mmk2_s36_a36_fps30__episodes_5747.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/galaxea_r1_lite_s14_a14_fps30__episodes_5167.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/galaxea_r1_lite_s16_a18_fps30__episodes_970.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/leju_robot_s54_a54_fps30__episodes_394.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/ruantong_a2d_s17_a17_fps30__episodes_1719.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/ruantong_a2d_s41_a34_fps30__episodes_6459.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/unitree_g1_s28_a28_fps30__episodes_1158.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/unknown_s30_a30_fps30__episodes_891.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/yinhe_s49_a16_fps30__episodes_5452.sh
===
```

bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboCOIN/galaxea_r1_lite_s14_a14_fps30__episodes_5167.sh


## RoboMIND Full

8 wrapper scripts remain unfinished, covering 8 RLDS repos.

```bash
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboMIND_full/franka_fr3_dual_master_puppet_joint_position_h5_franka_fr3_dual_real_s16_a16_fps30_cam_high_cam_left_cam_right_cam_top__episodes_1774.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboMIND_full/franka_sim_simulation_franka_joint_position_h5_simulation_sim_s8_a8_fps30_cam_handeye_cam_left_external_cam_right_external__episodes_158.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboMIND_full/franka_sim_simulation_franka_joint_position_none_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external__episodes_222.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboMIND_full/tienkung_humanoid_master_puppet_joint_position_h5_tienkung_gello_1rgb_real_s16_a16_fps30_cam_top__episodes_6626.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboMIND_full/tienkung_humanoid_master_puppet_joint_position_h5_tienkung_prod1_gello_1rgb_real_s16_a16_fps30_cam_top__episodes_2959.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboMIND_full/tienkung_humanoid_master_puppet_joint_position_h5_tienkung_xsens_1rgb_real_s14_a14_fps30_cam_top__episodes_6126.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboMIND_full/tienkung_humanoid_tiangong_joint_position_h5_sim_tienkung_1rgb_sim_s38_a38_fps30_cam_chest_cam_head__episodes_3965.sh
bash /mnt/workspace/wudi/InfiData-personal/scripts/convert2openpi/RoboMIND_full/tienkung_humanoid_tiangong_joint_position_none_real_s38_a38_fps30_cam_chest_cam_head__episodes_146.sh
===
```
