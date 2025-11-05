[1mdiff --git a/source/direct_pm01_walk/direct_pm01_walk/tasks/direct/direct_pm01_walk/direct_pm01_walk_env.py b/source/direct_pm01_walk/direct_pm01_walk/tasks/direct/direct_pm01_walk/direct_pm01_walk_env.py[m
[1mindex e672b48..8f7f958 100644[m
[1m--- a/source/direct_pm01_walk/direct_pm01_walk/tasks/direct/direct_pm01_walk/direct_pm01_walk_env.py[m
[1m+++ b/source/direct_pm01_walk/direct_pm01_walk/tasks/direct/direct_pm01_walk/direct_pm01_walk_env.py[m
[36m@@ -236,22 +236,22 @@[m [mclass DirectPm01WalkEnv(DirectRLEnv):[m
         print("joint_symmetry_penalty: %.3f \t weighted: %.3f" % (-joint_symmetry_penalty.mean().item(), -joint_symmetry_penalty.mean().item() * weight))[m
 [m
         left_leg_sum_penalty = joint_sum_l2(self, joint_names=["j00_hip_pitch_l", "j03_knee_pitch_l", "j04_ankle_pitch_l"])[m
[31m-        weight = 0.2[m
[32m+[m[32m        weight = 0.1[m
         reward -= left_leg_sum_penalty * weight[m
         print("left_leg_sum_penalty: %.3f \t weighted: %.3f" % (-left_leg_sum_penalty.mean().item(), -left_leg_sum_penalty.mean().item() * weight))[m
 [m
         left_leg_equal_penalty = joint_equal_l2(self, joint_name_a="j00_hip_pitch_l", joint_name_b="j04_ankle_pitch_l")[m
[31m-        weight = 0.2[m
[32m+[m[32m        weight = 0.1[m
         reward -= left_leg_equal_penalty * weight[m
         print("left_leg_equal_penalty: %.3f \t weighted: %.3f" % (-left_leg_equal_penalty.mean().item(), -left_leg_equal_penalty.mean().item() * weight))[m
 [m
         right_leg_sum_penalty = joint_sum_l2(self, joint_names=["j06_hip_pitch_r", "j09_knee_pitch_r", "j10_ankle_pitch_r"])[m
[31m-        weight = 0.2[m
[32m+[m[32m        weight = 0.1[m
         reward -= right_leg_sum_penalty * weight[m
         print("right_leg_sum_penalty: %.3f \t weighted: %.3f" % (-right_leg_sum_penalty.mean().item(), -right_leg_sum_penalty.mean().item() * weight))[m
 [m
         right_leg_equal_penalty = joint_equal_l2(self, joint_name_a="j06_hip_pitch_r", joint_name_b="j10_ankle_pitch_r")[m
[31m-        weight = 0.2[m
[32m+[m[32m        weight = 0.1[m
         reward -= right_leg_equal_penalty * weight[m
         print("right_leg_equal_penalty: %.3f \t weighted: %.3f" % (-right_leg_equal_penalty.mean().item(), -right_leg_equal_penalty.mean().item() * weight))[m
 [m
[1mdiff --git a/source/direct_pm01_walk/direct_pm01_walk/tasks/direct/direct_pm01_walk/rewards/rewards.py b/source/direct_pm01_walk/direct_pm01_walk/tasks/direct/direct_pm01_walk/rewards/rewards.py[m
[1mindex 8bed862..b51e5d8 100644[m
[1m--- a/source/direct_pm01_walk/direct_pm01_walk/tasks/direct/direct_pm01_walk/rewards/rewards.py[m
[1m+++ b/source/direct_pm01_walk/direct_pm01_walk/tasks/direct/direct_pm01_walk/rewards/rewards.py[m
[36m@@ -195,6 +195,7 @@[m [mdef get_gait_phase_reward(env):[m
 [m
     # 当前脚的世界坐标高度[m
     zL, zR = body_pos[:, l_id, 2], body_pos[:, r_id, 2][m
[32m+[m[32m    print('zL:', zL[0].item(), ' zR:', zR[0].item())[m
 [m
     # 当前步态相位（假设随时间线性增加）[m
     phase = env.gait_phase[m
[36m@@ -202,8 +203,8 @@[m [mdef get_gait_phase_reward(env):[m
 [m
     # 理想的脚高度曲线：sin(phase) 对应的目标高度[m
     # 左脚：在 sin>0 时高，右脚相反[m
[31m-    target_L = 0.1 + 0.1 * torch.clamp(phase_sin, min=0.0)   # 正半周抬高到 +0.2m[m
[31m-    target_R = 0.1 + 0.1 * torch.clamp(-phase_sin, min=0.0)  # 负半周抬高到 +0.2m[m
[32m+[m[32m    target_L = 0.2 * torch.clamp(phase_sin, min=0.0)   # 正半周抬高到 +0.2m[m
[32m+[m[32m    target_R = 0.2 * torch.clamp(-phase_sin, min=0.0)  # 负半周抬高到 +0.2m[m
 [m
     # 实际脚高度与目标高度的偏差[m
     err_L = (zL - target_L).pow(2)[m
