set dotenv-load
set shell := ["bash", "-c"]

ball-compile:
    uv run src/anymal_c_ball_balance/ball.py

# Ball Balance
ball-static:
    uv run play Mjlab-BallBalance-Anymal-C --agent zero --viewer viser

ball-random:
    uv run play Mjlab-BallBalance-Anymal-C --agent random --viewer viser

ball-train:
    uv run train Mjlab-BallBalance-Anymal-C --env.scene.num-envs 4096 --agent.max-iterations 30_000

ball-play CHECKPOINT *ARGS:
    uv run play Mjlab-BallBalance-Anymal-C --viewer viser --checkpoint-file {{CHECKPOINT}} {{ARGS}}

