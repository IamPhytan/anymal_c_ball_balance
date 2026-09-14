from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import anymal_c_ball_balance_env_cfg
from .rl_cfg import anymal_c_ball_balance_ppo_cfg

register_mjlab_task(
    task_id="Mjlab-BallBalance-Anymal-C",
    env_cfg=anymal_c_ball_balance_env_cfg(),
    play_env_cfg=anymal_c_ball_balance_env_cfg(play=True),
    rl_cfg=anymal_c_ball_balance_ppo_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
