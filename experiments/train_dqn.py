#!/usr/bin/env python3
"""Train a compact DQN policy against FIREOPS-SIM in plain PyTorch.

This is the from-scratch counterpart to the Stable-Baselines3 PPO script: an
explicit replay buffer, target network, Huber loss, and epsilon-greedy
exploration, written to be read. PyTorch is imported only after argument
validation, so the base simulator never requires the deep-learning stack.

Example:
    python experiments/train_dqn.py --mode robust --timesteps 20000 \
        --output models/dqn_robust
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments._common import (  # noqa: E402
    ExperimentConfigurationError,
    create_environment,
    public_artifact_path,
    reset_environment,
    sha256_file,
)
from experiments.dqn_core import (  # noqa: E402
    ReplayBuffer,
    flatten_observation,
    linear_epsilon,
)

TRAINING_MODES = ("nominal", "robust")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=TRAINING_MODES, default="nominal")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("models/dqn_nominal"))
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--target-sync", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=256)
    arguments = parser.parse_args(argv)
    if arguments.timesteps <= 0:
        raise ExperimentConfigurationError("timesteps must be positive")
    if arguments.warmup >= arguments.buffer_size:
        raise ExperimentConfigurationError("warmup must be smaller than buffer size")
    return arguments


def episode_reliability(mode: str, rng: Any) -> float:
    """Nominal trains at full reliability; robust randomizes per episode."""

    if mode == "nominal":
        return 1.0
    return float(rng.uniform(0.25, 1.0))


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)

    try:
        import numpy as np
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise SystemExit(
            "PyTorch is required for DQN training. Install the optional stack with: "
            'python -m pip install -e "./backend[ppo]"'
        ) from exc

    torch.manual_seed(arguments.seed)
    rng = np.random.default_rng(arguments.seed)
    environment = create_environment(reliability=1.0, seed=arguments.seed)

    reliability = episode_reliability(arguments.mode, rng)
    observation, _ = reset_environment(
        environment, reliability=reliability, seed=arguments.seed
    )
    state = flatten_observation(observation)
    action_count = int(environment.action_space.n)

    class QNetwork(nn.Module):
        def __init__(self, inputs: int, actions: int, hidden: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, actions),
            )

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return self.layers(batch)

    online = QNetwork(state.size, action_count, arguments.hidden_size)
    target = QNetwork(state.size, action_count, arguments.hidden_size)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=arguments.learning_rate)
    loss_function = nn.SmoothL1Loss()
    buffer = ReplayBuffer(arguments.buffer_size, seed=arguments.seed + 1)

    episode_rewards: list[float] = []
    episode_reward = 0.0
    updates = 0

    for step in range(arguments.timesteps):
        epsilon = linear_epsilon(step, decay_steps=max(1, arguments.timesteps // 2))
        if rng.random() < epsilon:
            action = int(rng.integers(0, action_count))
        else:
            with torch.no_grad():
                q_values = online(torch.from_numpy(state).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())

        observation, reward, terminated, truncated, _ = environment.step(action)
        done = bool(terminated or truncated)
        next_state = flatten_observation(observation)
        buffer.push(state, action, float(reward), next_state, done)
        episode_reward += float(reward)
        state = next_state

        if done:
            episode_rewards.append(round(episode_reward, 4))
            episode_reward = 0.0
            reliability = episode_reliability(arguments.mode, rng)
            observation, _ = reset_environment(
                environment, reliability=reliability, seed=None
            )
            state = flatten_observation(observation)

        if len(buffer) >= max(arguments.warmup, arguments.batch_size):
            states, actions, rewards, next_states, dones = buffer.sample(
                arguments.batch_size
            )
            state_batch = torch.from_numpy(states)
            action_batch = torch.from_numpy(actions).unsqueeze(1)
            reward_batch = torch.from_numpy(rewards)
            next_batch = torch.from_numpy(next_states)
            done_batch = torch.from_numpy(dones)

            predicted = online(state_batch).gather(1, action_batch).squeeze(1)
            with torch.no_grad():
                future = target(next_batch).max(dim=1).values
                expected = reward_batch + arguments.discount * future * (1.0 - done_batch)
            loss = loss_function(predicted, expected)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            optimizer.step()
            updates += 1
            if updates % arguments.target_sync == 0:
                target.load_state_dict(online.state_dict())

    output = arguments.output.with_suffix(".pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(online.state_dict(), output)

    metadata = {
        "algorithm": "dqn",
        "mode": arguments.mode,
        "seed": arguments.seed,
        "timesteps": arguments.timesteps,
        "episodes_completed": len(episode_rewards),
        "last_10_episode_rewards": episode_rewards[-10:],
        "hyperparameters": {
            "learning_rate": arguments.learning_rate,
            "discount": arguments.discount,
            "batch_size": arguments.batch_size,
            "buffer_size": arguments.buffer_size,
            "warmup": arguments.warmup,
            "target_sync": arguments.target_sync,
            "hidden_size": arguments.hidden_size,
        },
        "observation_features": int(state.size),
        "torch_version": torch.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": public_artifact_path(output),
        "artifact_sha256": sha256_file(output),
        "disclaimer": (
            "Synthetic research artifact. A short training run demonstrates the "
            "pipeline, not general robustness, and is never real emergency guidance."
        ),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved DQN weights to {public_artifact_path(output)}")
    print(f"Saved metadata to {public_artifact_path(metadata_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
