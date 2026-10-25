"""RL configuration for ANYmal C Ball Balance task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def anymal_c_ball_balance_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for ANYmal C Ball Balance task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
        ),
        experiment_name="anymal_c_ball_balance",
        max_iterations=30_000,
        wandb_project="mjlab-anymal-c-ball",
    )
