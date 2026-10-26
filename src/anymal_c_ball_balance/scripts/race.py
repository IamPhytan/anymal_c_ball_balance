"""Race multiple ANYmal C robots balancing on balls."""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from enum import IntEnum
from pathlib import Path

import anymal_c_ball_balance  # noqa: F401
import mujoco
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationManager
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViserPlayViewer

TASK = "Mjlab-BallBalance-Anymal-C"


class RaceState(IntEnum):
    RUNNING = 0
    FINISHED = 1
    FAILED = 2
    TIMED_OUT = 3


@dataclass(frozen=True)
class RaceConfig:
    """Race options"""

    checkpoints: tuple[str, ...]
    """One shared checkpoint, or one checkpoint per robot."""
    num_envs: int = 4
    """Number of race participants."""
    distance: float = 10.0
    """Race distance, in meters"""
    speed: float = 0.5
    """Forward velocity command, in meters per second."""
    max_time: float = 120.0
    """Maximum race duration, in simulated seconds."""
    spacing: float = 3.0
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    """Lateral distance between starting positions, in meters."""
    log_root: str = "logs/rsl_rl"
    """Root directory under which experiment logs are written."""
    seed: int = 42
    device: str | None = None


class RaceVelocityCommand(UniformVelocityCommand):
    def create_gui(
        self,
        name,
        server,
        get_env_idx,
        on_change=None,
        request_action=None,
    ):
        # Race commands are configured before simulation starts.
        # Skip the interactive joystick and its sliders.
        pass


@dataclass(kw_only=True)
class RaceVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return RaceVelocityCommand(self, env)


def add_race_markings(env_cfg, race_cfg: RaceConfig):
    previous_spec_fn = env_cfg.scene.spec_fn

    def configure_scene(spec):
        # Preserve any existing scene customization.
        if previous_spec_fn is not None:
            previous_spec_fn(spec)

        # Cover all starting lanes with some extra margin.
        half_width = (race_cfg.num_envs - 1) * race_cfg.spacing / 2 + 1.5

        markings = (
            ("race_start", 0.0, (1.0, 1.0, 1.0, 1.0)),
            ("race_finish", race_cfg.distance, (0.1, 0.9, 0.2, 1.0)),
        )

        for name, x, color in markings:
            # MuJoCo box sizes are half-extents.
            # Raise the marking slightly above the ground.
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(x, 0.0, 0.006),
                size=(0.04, half_width, 0.005),
                rgba=color,
                contype=0,
                conaffinity=0,
            )

    env_cfg.scene.spec_fn = configure_scene


def make_race_cfg(cfg: RaceConfig) -> ManagerBasedRlEnvCfg:
    env_cfg = load_env_cfg(TASK, play=True)

    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.max_time + 10.0
    env_cfg.curriculum = {}

    # Flat terrain
    assert env_cfg.scene.terrain is not None
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None

    # Same command for all participants
    original_twist = env_cfg.commands["twist"]
    # Preserve the original configuration, including nested dataclasses.
    twist = RaceVelocityCommandCfg(
        **{
            field.name: getattr(original_twist, field.name)
            for field in fields(original_twist)
            if field.init
        }
    )
    env_cfg.commands["twist"] = twist

    twist.heading_command = False
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = (cfg.speed, cfg.speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)

    twist.rel_standing_envs = 0.0
    twist.rel_world_envs = 1.0
    twist.rel_forward_envs = 0.0
    twist.init_velocity_prob = 0.0
    twist.resampling_time_range = (cfg.max_time + 10.0,) * 2

    env_cfg.events.pop("push_robot", None)
    env_cfg.observations["actor"].enable_corruption = False

    # Inline start
    reset = env_cfg.events["reset_ball_balance"]
    reset.params["xy_noise"] = 0.0
    reset.params["joint_noise"] = 0.02

    env_cfg.viewer.distance = max(12.0, cfg.num_envs * cfg.spacing)

    add_race_markings(env_cfg, cfg)

    return env_cfg


class RaceEnv(ManagerBasedRlEnv):
    def __init__(
        self,
        env_cfg: ManagerBasedRlEnvCfg,
        race_cfg: RaceConfig,
        device: str,
        render_mode: str | None = None,
    ) -> None:
        # Evaluate failure conditions separately so fallen robots are never automatically reset during the race.
        failure_cfg = deepcopy(
            {
                name: term
                for name, term in env_cfg.terminations.items()
                if term is not None and not term.time_out
            }
        )
        env_cfg.terminations = {}

        super().__init__(cfg=env_cfg, device=device, render_mode=render_mode)

        self.failure_manager = TerminationManager(failure_cfg, self)
        self.distance = race_cfg.distance
        self.max_time = race_cfg.max_time

        # Place independent simulation worlds in parallel lanes.
        origins = self.scene.env_origins
        origins.zero_()
        origins[:, 1] = (
            torch.arange(self.num_envs, device=self.device) - (self.num_envs - 1) / 2
        ) * race_cfg.spacing

        # 0 = racing, 1 = arrived, 2 = forfeit, 3 = exceeded time
        self.status = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.finish_times = torch.full(
            (self.num_envs,),
            float("inf"),
            device=self.device,
        )
        self.progress = torch.zeros(self.num_envs, device=self.device)

        self.elapsed = 0.0
        self.finished = False
        self.last_step = None

    def reset(self, *, seed=None, env_ids=None, options=None):
        # A manual reset always restarts the entire race.
        result = super().reset(seed=seed)

        self.failure_manager.reset()
        self.status.fill_(RaceState.RUNNING)
        self.finish_times.fill_(float("inf"))
        self.progress.zero_()

        self.elapsed = 0.0
        self.finished = False
        self.last_step = None

        self.start_x = self.scene["robot"].data.root_link_pos_w[:, 0].clone()

        return result

    def step(self, actions):
        # Freeze the final scene once every participant is done.
        if self.finished:
            return self.last_step

        previous = self.progress.clone()
        active = self.status == RaceState.RUNNING

        result = super().step(actions)
        self.elapsed += self.step_dt

        current = self.scene["robot"].data.root_link_pos_w[:, 0] - self.start_x

        # Stop updating recorded progress for eliminated participants.
        self.progress[active] = current[active]

        # Reuse the task's original non-timeout failure conditions.
        self.failure_manager.compute()
        failed = active & self.failure_manager.terminated

        # Fall at the end counts as a fall
        arrived = active & ~failed & (current >= self.distance)

        # Compute finish times
        if arrived.any():
            delta = (current - previous).clamp_min(1e-8)
            fraction = ((self.distance - previous) / delta).clamp(0.0, 1.0)

            crossing_time = self.elapsed - self.step_dt + fraction * self.step_dt
            self.finish_times[arrived] = crossing_time[arrived]

        self.status[failed] = RaceState.FAILED
        self.status[arrived] = RaceState.FINISHED

        if self.elapsed >= self.max_time:
            self.status[self.status == RaceState.RUNNING] = 3

        self.finished = bool((self.status != RaceState.RUNNING).all().item())
        self.last_step = result

        if self.finished:
            print("\n" + self.scoreboard(), flush=True)

        return result

    def scoreboard(self):
        status = self.status.cpu().tolist()
        times = self.finish_times.cpu().tolist()
        progress = self.progress.cpu().tolist()

        order = sorted(
            range(self.num_envs),
            key=lambda i: (
                0 if status[i] == RaceState.FINISHED else 1,
                times[i] if status[i] == RaceState.FINISHED else -progress[i],
            ),
        )

        labels = ["Running", "Finished", "Failed", "Timed out"]
        lines = [
            f"Simulated time : {self.elapsed:.2f} s",
            "",
            "| Robot | Distance | Status | Finish time |",
            "|---|---:|---|---:|",
        ]

        for i in order:
            arrival = f"{times[i]:.3f} s" if status[i] == RaceState.FINISHED else "—"
            lines.append(
                f"| {i + 1} | {progress[i]:.2f} m | {labels[status[i]]} | {arrival} |"
            )

        return "\n".join(lines)


class RaceViewer(ViserPlayViewer):
    def setup(self) -> None:
        super().setup()

        # Show all envs simultaneously
        self._scene.show_only_selected = False
        self._scene.request_update()

        self.board = self._server.gui.add_markdown(self.env.unwrapped.scoreboard())

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()

        race: RaceEnv = self.env.unwrapped
        self.board.content = race.scoreboard()

        if race.finished:
            self.pause()


def run_race(cfg: RaceConfig):
    configure_torch_backends()

    if cfg.num_envs < 1:
        raise ValueError("num_envs must be strictly positive.")

    if len(cfg.checkpoints) not in (1, cfg.num_envs):
        raise ValueError(
            f"Provide one shared checkpoint or {cfg.num_envs} individual checkpoints."
        )

    if min(cfg.distance, cfg.speed, cfg.max_time, cfg.spacing) <= 0:
        raise ValueError(
            "distance, speed, max_time and spacing must be strictly positive."
        )

    if cfg.video and cfg.video_length < 1:
        raise ValueError("video_length must be strictly positive.")

    for value in (cfg.video_height, cfg.video_width):
        if value is not None and value < 1:
            raise ValueError("Video dimensions must be strictly positive.")

    for checkpoint in cfg.checkpoints:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = make_race_cfg(cfg)
    agent_cfg = load_rl_cfg(TASK)

    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    if cfg.video:
        # Include every participant in the offscreen rendering.
        env_cfg.viewer.env_idx = 0
        env_cfg.viewer.max_extra_envs = cfg.num_envs - 1

        # Use a fixed camera overlooking the entire track.
        env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.WORLD
        env_cfg.viewer.lookat = (
            cfg.distance / 2,
            0.0,
            0.5,
        )
        env_cfg.viewer.elevation = -25.0
        env_cfg.viewer.distance = max(
            env_cfg.viewer.distance,
            cfg.distance * 1.5,
            cfg.num_envs * cfg.spacing * 1.5,
        )

    render_mode = "rgb_array" if cfg.video else None

    # Match play.py: save videos next to the checkpoint.
    # For multiple checkpoints, use the first one's directory.
    log_dir = Path(cfg.checkpoints[0]).resolve().parent
    video_folder = log_dir / "videos" / "race"

    race_env = None

    try:
        race_env = RaceEnv(
            env_cfg=env_cfg,
            race_cfg=cfg,
            device=device,
            render_mode=render_mode,
        )

        if cfg.video:
            print(f"[INFO]: Recording video to {video_folder}")

            race_env = VideoRecorder(
                race_env,
                video_folder=video_folder,
                step_trigger=lambda step: step == 0,
                video_length=cfg.video_length,
                name_prefix="race",
                disable_logger=False,
            )

        env = RslRlVecEnvWrapper(
            race_env,
            clip_actions=agent_cfg.clip_actions,
        )

        # Use the task's registered runner
        runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner

        runners = []
        policies = []
        for checkpoint in cfg.checkpoints:
            print(f"[INFO]: Loading checkpoint: {checkpoint}")
            runner = runner_cls(
                env,
                asdict(agent_cfg),
                device=device,
            )
            runner.load(
                checkpoint,
                load_cfg={"actor": True},
                strict=True,
                map_location=device,
            )
            runners.append(runner)
            policies.append(runner.get_inference_policy(device=device))

        if len(policies) == 1:
            # A single policy controls the entire observation batch.
            policy = policies[0]
        else:

            def policy(obs):
                # Evaluate each policy on the full batch and select
                # the action for its corresponding participant.
                return torch.stack(
                    [
                        participant_policy(obs)[i]
                        for i, participant_policy in enumerate(policies)
                    ]
                )

        with torch.inference_mode():
            RaceViewer(env, policy).run()

    finally:
        if race_env is not None:
            race_env.close()


def main():
    cfg = tyro.cli(RaceConfig)
    run_race(cfg)


if __name__ == "__main__":
    main()
