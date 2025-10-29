# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .direct_pm01_walk_env_cfg import DirectPm01WalkEnvCfg


class DirectPm01WalkEnv(DirectRLEnv):
    cfg: DirectPm01WalkEnvCfg

    def __init__(self, cfg: DirectPm01WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        action_scale = 1.0
        self.robot.set_joint_effort_target(self.actions * action_scale)


    def _get_observations(self) -> dict:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        obs = torch.cat([base_lin_vel, base_ang_vel, joint_pos, joint_vel], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return termination and timeout flags for each environment."""
        # 1. 只根据 episode 长度判断超时
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 2. 当前不检测摔倒 / 越界
        terminated = torch.zeros_like(time_out, device=self.device, dtype=torch.bool)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments to default state (minimal version)."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # 调用父类逻辑（清理 buffers）
        super()._reset_idx(env_ids)

        # 默认状态
        joint_pos = torch.zeros_like(self.robot.data.joint_pos[env_ids])
        joint_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids])
        root_state = self.robot.data.default_root_state[env_ids].clone()

        # 将每个环境放到对应的 origin（env_spacing 控制）
        root_state[:, :3] = self.scene.env_origins[env_ids]

        # 写入仿真
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

