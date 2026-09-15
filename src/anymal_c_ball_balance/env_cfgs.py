"""ANYbotics ANYmal C velocity environment configurations."""

import mujoco
from anymal_c_velocity.env_cfgs import anymal_c_flat_env_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from . import mdp
from .ball import BALL_RADIUS, get_ball_spec

FOOT_SITES = ("LF", "RF", "LH", "RH")

# À calibrer visuellement au début.
ROBOT_HEIGHT_OFFSET = -0.10

# Hauteur du root par rapport au centre du ballon.
TARGET_ROOT_HEIGHT = BALL_RADIUS + 0.54 + ROBOT_HEIGHT_OFFSET


def anymal_c_ball_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create ANYmal C ball balance velocity configuration."""
    cfg = anymal_c_flat_env_cfg(play=play)

    # -----------------------------------------------------------------------
    # Scene
    # -----------------------------------------------------------------------

    cfg.scene.entities["ball"] = EntityCfg(spec_fn=get_ball_spec)

    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 3.0)
    cfg.episode_length_s = 10.0
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)

    # -----------------------------------------------------------------------
    # Commands
    #
    # Phase 1: no locomotion command.
    # -----------------------------------------------------------------------
    twist = cfg.commands["twist"]

    twist.heading_command = False
    twist.ranges.heading = None

    # Non-zero ranges required by the Viser command GUI.
    twist.ranges.lin_vel_x = (-0.1, 0.1)
    twist.ranges.lin_vel_y = (-0.1, 0.1)
    twist.ranges.ang_vel_z = (-0.1, 0.1)

    # But every training environment receives the zero command.
    twist.rel_standing_envs = 1.0

    # Disable the other command modes.
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.init_velocity_prob = 0.0

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------
    for group_name in ("actor", "critic"):
        terms = cfg.observations[group_name].terms

        terms["ball_pos_rel"] = ObservationTermCfg(func=mdp.ball_pos_rel_w)
        terms["ball_lin_vel"] = ObservationTermCfg(func=mdp.ball_lin_vel_w)
        terms["ball_ang_vel"] = ObservationTermCfg(func=mdp.ball_ang_vel_w)

    # Terrain-height information is not useful for the balancing task.
    cfg.observations["critic"].terms.pop("foot_height", None)

    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name != "foot_height_scan"
    )

    nonfoot_ball_contact = ContactSensorCfg(
        name="nonfoot_ball_contact",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision.*",
        ),
        secondary=ContactMatch(
            mode="geom",
            entity="ball",
            pattern="ball_sphere",
        ),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (
        *(cfg.scene.sensors or ()),
        nonfoot_ball_contact,
    )

    # -----------------------------------------------------------------------
    # Remove walking-specific rewards
    # -----------------------------------------------------------------------

    for reward_name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        cfg.rewards.pop(reward_name, None)

    # Keep the useful regularizers from the velocity task:
    #
    # pose
    # upright
    # dof_pos_limits
    # action_rate_l2
    # body_ang_vel
    # angular_momentum

    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # -----------------------------------------------------------------------
    # Ball rewards
    # -----------------------------------------------------------------------

    # Rewards
    cfg.rewards["ball_centered"] = RewardTermCfg(
        func=mdp.ball_centered,
        weight=3.0,
        params={"std": 0.15},
    )

    cfg.rewards["feet_on_ball"] = RewardTermCfg(
        func=mdp.feet_on_ball,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
            "ball_radius": BALL_RADIUS,
            "foot_radius": 0.03,
            "std": 0.05,
        },
    )

    cfg.rewards["target_height"] = RewardTermCfg(
        func=mdp.target_root_height,
        weight=1.0,
        params={
            "target_height": TARGET_ROOT_HEIGHT,
            "std": 0.10,
        },
    )

    cfg.rewards["ball_speed"] = RewardTermCfg(
        func=mdp.ball_speed_l2,
        weight=-0.05,
    )

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)

    # No perturbations in the first training phase.
    cfg.events.pop("push_robot", None)

    cfg.events["reset_ball_balance"] = EventTermCfg(
        func=mdp.reset_ball_balance,
        mode="reset",
        params={
            "ball_radius": BALL_RADIUS,
            "robot_height_offset": ROBOT_HEIGHT_OFFSET,
            "xy_noise": 0.01,
            "joint_noise": 0.02,
        },
    )

    # -----------------------------------------------------------------------
    # Terminations
    # -----------------------------------------------------------------------

    cfg.terminations["fell_from_ball"] = TerminationTermCfg(
        func=mdp.fell_from_ball,
        params={
            "max_offset": 0.55,
            "min_height": BALL_RADIUS + 0.25,
            "max_tilt_deg": 60.0,
        },
    )

    # feet_ground_contact already exists in your ANYmal velocity config.
    cfg.terminations["feet_touched_ground"] = TerminationTermCfg(
        func=mdp.feet_touched_ground,
        params={
            "sensor_name": "feet_ground_contact",
        },
    )

    cfg.terminations["nonfoot_ball_contact"] = TerminationTermCfg(
        func=mdp.nonfoot_ball_contact,
        params={
            "sensor_name": "nonfoot_ball_contact",
        },
    )

    return cfg
