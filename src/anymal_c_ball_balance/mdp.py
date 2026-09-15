from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity import mdp, velocity_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg

# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def ball_pos_rel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Ball position relative to the robot root, in world frame."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    return ball.data.root_link_pos_w - robot.data.root_link_pos_w


def ball_lin_vel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Ball linear velocity."""
    return env.scene["ball"].data.root_link_lin_vel_w


def ball_ang_vel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Ball angular velocity."""
    return env.scene["ball"].data.root_link_ang_vel_w


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def ball_centered(
    env: ManagerBasedRlEnv,
    std: float = 0.15,
) -> torch.Tensor:
    """Reward the robot for keeping its root above the ball center."""
    robot_pos = env.scene["robot"].data.root_link_pos_w
    ball_pos = env.scene["ball"].data.root_link_pos_w

    error = robot_pos[:, :2] - ball_pos[:, :2]
    error_sq = torch.sum(error**2, dim=-1)

    return torch.exp(-error_sq / std**2)


def feet_on_ball(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    ball_radius: float,
    foot_radius: float = 0.03,
    std: float = 0.04,
) -> torch.Tensor:
    """Reward all four foot spheres for staying on the ball surface."""
    robot = env.scene[asset_cfg.name]
    ball = env.scene["ball"]

    foot_pos = robot.data.site_pos_w[:, asset_cfg.site_ids]
    ball_pos = ball.data.root_link_pos_w[:, None, :]

    distance = torch.linalg.vector_norm(foot_pos - ball_pos, dim=-1)

    # ANYmal foot sites are at the center of the spherical foot collision geom.
    target_distance = ball_radius + foot_radius
    error = distance - target_distance

    reward_per_foot = torch.exp(-((error / std) ** 2))

    return reward_per_foot.mean(dim=-1)


def target_root_height(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float = 0.08,
) -> torch.Tensor:
    """Reward desired root height relative to ball center."""
    robot_z = env.scene["robot"].data.root_link_pos_w[:, 2]
    ball_z = env.scene["ball"].data.root_link_pos_w[:, 2]

    relative_height = robot_z - ball_z
    error = relative_height - target_height

    return torch.exp(-((error / std) ** 2))


def ball_speed_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize horizontal ball velocity."""
    vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
    return torch.sum(vel**2, dim=-1)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def fell_from_ball(
    env: ManagerBasedRlEnv,
    max_offset: float,
    min_height: float,
    max_tilt_deg: float = 60.0,
) -> torch.Tensor:
    """Off the ball: root too low above the ball centre, root horizontally off
    the ball, or tilted past ``fall_tilt_deg``."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    robot_pos = robot.data.root_link_pos_w
    ball_pos = ball.data.root_link_pos_w

    offset = torch.linalg.vector_norm(
        robot_pos[:, :2] - ball_pos[:, :2],
        dim=-1,
    )

    relative_height = robot_pos[:, 2] - ball_pos[:, 2]

    # MuJoCo/mjlab quaternion: w, x, y, z.
    quat = robot.data.root_link_quat_w

    # Dot(body_z, world_z).
    upright_cos = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    min_cos = math.cos(math.radians(max_tilt_deg))

    return (
        (offset > max_offset) | (relative_height < min_height) | (upright_cos < min_cos)
    )


def feet_touched_ground(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    return found.reshape(found.shape[0], -1).any(dim=-1)


def nonfoot_ball_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    found = env.scene.sensors[sensor_name].data.found
    assert found is not None

    return (found > 0).any(dim=-1)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def reset_ball_balance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ball_radius: float,
    robot_height_offset: float,
    xy_noise: float = 0.01,
    joint_noise: float = 0.02,
) -> None:
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    origins = env.scene.env_origins[env_ids]
    num_envs = len(env_ids)

    #
    # Ball
    #
    ball_state = ball.data.default_root_state[env_ids].clone()
    ball_state[:, :3] = origins
    ball_state[:, 2] += ball_radius

    # Unit quaternion.
    ball_state[:, 3:7] = 0.0
    ball_state[:, 3] = 1.0

    # Zero linear/angular velocity.
    ball_state[:, 7:] = 0.0

    ball.write_root_state_to_sim(
        ball_state,
        env_ids=env_ids,
    )

    #
    # Robot
    #

    robot_state = robot.data.default_root_state[env_ids].clone()

    robot_state[:, :3] += origins

    # Ball top + normal ANYmal root height correction.
    robot_state[:, 2] += 2.0 * ball_radius + robot_height_offset

    if xy_noise > 0:
        noise = torch.empty(
            (num_envs, 2),
            device=env.device,
        ).uniform_(-xy_noise, xy_noise)

        robot_state[:, :2] += noise

    robot_state[:, 7:] = 0.0

    robot.write_root_state_to_sim(
        robot_state,
        env_ids=env_ids,
    )

    #
    # Joints
    #

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    if joint_noise > 0:
        joint_pos += torch.empty_like(joint_pos).uniform_(
            -joint_noise,
            joint_noise,
        )

    joint_vel.zero_()

    robot.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
    )

    robot.reset(env_ids=env_ids)
    ball.reset(env_ids=env_ids)
