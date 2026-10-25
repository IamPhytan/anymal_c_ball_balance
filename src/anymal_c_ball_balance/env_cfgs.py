"""ANYbotics ANYmal C velocity environment configurations."""

from anymal_c_velocity.env_cfgs import anymal_c_flat_env_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

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
    cfg.episode_length_s = 60.0 if play else 10.0
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)

    # -----------------------------------------------------------------------
    # Commands
    #
    # Phase 1: no locomotion command.
    # -----------------------------------------------------------------------
    twist = cfg.commands["twist"]

    twist.heading_command = False
    twist.ranges.heading = None

    twist.rel_standing_envs = 0.25

    # Disable the other command modes.
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.init_velocity_prob = 0.0

    twist.resampling_time_range = (6.0, 8.0)

    # -----------------------------------------------------------------------
    # Velocity curriculum
    # -----------------------------------------------------------------------

    cfg.curriculum.pop("command_vel", None)
    if not play:
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=velocity_mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x": (-0.15, 0.15),
                        "lin_vel_y": (-0.10, 0.10),
                        "ang_vel_z": (-0.10, 0.10),
                    },
                    {
                        "step": 3000 * 24,
                        "lin_vel_x": (-0.20, 0.25),
                        "lin_vel_y": (-0.10, 0.10),
                        "ang_vel_z": (-0.10, 0.10),
                    },
                    {
                        "step": 6000 * 24,
                        "lin_vel_x": (-0.20, 0.40),
                        "lin_vel_y": (-0.12, 0.12),
                        "ang_vel_z": (-0.12, 0.12),
                    },
                    {
                        "step": 10000 * 24,
                        "lin_vel_x": (-0.25, 0.55),
                        "lin_vel_y": (-0.15, 0.15),
                        "ang_vel_z": (-0.12, 0.12),
                    },
                    {
                        "step": 14000 * 24,
                        "lin_vel_x": (-0.30, 0.70),
                        "lin_vel_y": (-0.15, 0.15),
                        "ang_vel_z": (-0.15, 0.15),
                    },
                    {
                        "step": 18000 * 24,
                        "lin_vel_x": (-0.30, 0.85),
                        "lin_vel_y": (-0.15, 0.15),
                        "ang_vel_z": (-0.15, 0.15),
                    },
                    {
                        "step": 22000 * 24,
                        "lin_vel_x": (-0.35, 1.00),
                        "lin_vel_y": (-0.15, 0.15),
                        "ang_vel_z": (-0.15, 0.15),
                    },
                ],
            },
        )
    else:
        # Full command range for evaluation
        twist.ranges.lin_vel_x = (-0.35, 1.00)
        twist.ranges.lin_vel_y = (-0.15, 0.15)
        twist.ranges.ang_vel_z = (-0.15, 0.15)

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------
    for group_name in ("actor", "critic"):
        terms = cfg.observations[group_name].terms

        terms["ball_pos_rel"] = ObservationTermCfg(func=mdp.ball_pos_rel_b)
        terms["ball_lin_vel"] = ObservationTermCfg(func=mdp.ball_lin_vel_b)
        terms["ball_ang_vel"] = ObservationTermCfg(func=mdp.ball_ang_vel_b)

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
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        cfg.rewards.pop(reward_name, None)

    # Velocity tracking
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.20

    cfg.rewards["track_angular_velocity"].weight = 0.5
    cfg.rewards["track_angular_velocity"].params["std"] = 0.5

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
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["pose"].params["std_walking"][".*KFE"] = 0.45

    # -----------------------------------------------------------------------
    # Ball rewards
    # -----------------------------------------------------------------------

    # Rewards
    cfg.rewards["ball_centered"] = RewardTermCfg(
        func=mdp.ball_centered,
        weight=2.0,
        params={"std": 0.25},
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
        weight=1.5,
        params={
            "target_height": TARGET_ROOT_HEIGHT,
            "std": 0.10,
        },
    )

    cfg.rewards["ball_speed"] = RewardTermCfg(
        func=mdp.ball_speed_l2,
        weight=0.0,
    )

    KNEE_BODIES = (
        "LF_SHANK",
        "RF_SHANK",
        "LH_SHANK",
        "RH_SHANK",
    )
    cfg.rewards["knee_ball_proximity"] = RewardTermCfg(
        func=mdp.knee_ball_proximity_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=KNEE_BODIES,
            ),
            "ball_radius": BALL_RADIUS,
            "min_clearance": 0.12,
        },
    )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    cfg.metrics["velocity_tracking_error_xy"] = MetricsTermCfg(
        func=mdp.velocity_tracking_error_xy
    )

    cfg.metrics["yaw_tracking_error"] = MetricsTermCfg(func=mdp.yaw_tracking_error)

    cfg.metrics["ball_center_error"] = MetricsTermCfg(func=mdp.ball_center_error)

    cfg.metrics["knee_clearance"] = MetricsTermCfg(
        func=mdp.knee_clearance_proxy,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=KNEE_BODIES,
            ),
            "ball_radius": BALL_RADIUS,
        },
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
