from isaaclab.utils.math import quat_apply
import torch

def flat_orientation_l2(env):
    """计算与世界Z轴的偏离程度，鼓励身体保持直立"""
    base_quat = env.robot.data.root_quat_w   # (num_envs, 4)

    num_envs = base_quat.shape[0]
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=torch.float32).repeat(num_envs, 1)
    up_vector = quat_apply(base_quat, world_up)  # (num_envs, 3)

    # 惩罚身体倾斜：up_vector 应该尽量接近 [0, 0, 1]
    deviation = 1.0 - up_vector[:, 2]  # z 分量偏离 1
    l2 = deviation**2  # L2 penalty
    return l2

def fall_penalty(env):
    """计算跌倒惩罚，基于身体与地面的高度差"""
    base_pos = env.robot.data.root_pos_w  # (num_envs, 3)
    height = base_pos[:, 2]  # z 轴高度

    # 假设跌倒阈值为0.5米
    fall_threshold = 0.5
    # 给一个固定的惩罚值200
    penalty = torch.where(height < fall_threshold, torch.tensor(10.0, device=env.device), torch.tensor(0.0, device=env.device))
    return penalty